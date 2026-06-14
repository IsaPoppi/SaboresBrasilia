from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Annotated, Literal
from groq import Groq
from dotenv import load_dotenv
import json, os, httpx, random, logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chatbot Restaurantes Brasília")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_groq_api_key = os.getenv("GROQ_API_KEY")
if not _groq_api_key:
    raise RuntimeError("GROQ_API_KEY não encontrada. Crie um arquivo .env com GROQ_API_KEY=sua_chave.")
groq_client = Groq(api_key=_groq_api_key, timeout=15.0)

# ── Schema de saída do LLM ───────────────────────────────────────────────────
class ModelOutput(BaseModel):
    cuisine: Annotated[
        str | None,
        "tipo de cozinha ou estabelecimento mencionado pelo usuário em linguagem natural "
        "(ex: japonês, pizza, churrasco, italiano, vegetariano, burger, chinês, "
        "árabe, seafood, mexicano, café, cafeteria, padaria, indiano, coreano, "
        "lanche, frango) — null se não especificado"
    ] = None
    cuisine_osm_regex: Annotated[
        str | None,
        "regex para filtrar a tag 'cuisine' do OpenStreetMap, em inglês, com termos separados por '|'. "
        "Exemplos: 'japanese|sushi|temaki', 'pizza|italian', 'barbecue|churrascaria|brazilian', "
        "'burger|american', 'chinese|asian', 'arab|lebanese|middle_eastern', 'seafood|fish', "
        "'vegetarian|vegan', 'indian', 'korean', 'mexican', 'sandwich|burger|fastfood'. "
        "Deve ser null quando amenity='cafe' (cafés raramente usam a tag cuisine no OSM) "
        "ou quando nenhuma culinária foi especificada"
    ] = None
    amenity: Annotated[
        Literal["restaurant", "cafe"],
        "tipo de estabelecimento no OSM: use 'cafe' para cafés, cafeterias, padarias e confeitarias; "
        "use 'restaurant' para todos os demais"
    ] = "restaurant"
    neighborhood: Annotated[
        Literal[
            "asa sul", "asa norte", "lago sul", "lago norte",
            "sudoeste", "noroeste", "águas claras", "taguatinga", "guará"
        ] | None,
        "bairro mencionado em lowercase "
        "(ex: asa sul, asa norte, lago sul, lago norte, sudoeste, noroeste, "
        "águas claras, taguatinga, guará) — null se não mencionado"
    ] = None
    friendly_message: Annotated[
        str,
        "mensagem de 1 frase dizendo que está buscando os lugares"
    ] = "Buscando restaurantes para você..."

INTENT_PROMPT = """Você extrai intenções de busca de restaurantes em Brasília.
Responda SOMENTE com JSON válido, sem markdown, sem explicações.

Campos obrigatórios:
•⁠  ⁠"cuisine": string em português (ex: "japonês", "pizza", "churrasco") ou null
•⁠  ⁠"cuisine_osm_regex": string com termos em inglês separados por "|" para busca no OpenStreetMap.
  SEMPRE preencha quando cuisine não for null. Exemplos:
    "japonês" → "japanese|sushi|temaki"
    "pizza" → "pizza|italian"
    "churrasco" → "barbecue|churrascaria|brazilian"
    "churrascarias" → "barbecue|churrascaria|brazilian|grill"
    "burger" → "burger|american|hamburger"
    "chinês" → "chinese|asian"
    "árabe" → "arab|lebanese|middle_eastern"
    "frutos do mar" → "seafood|fish"
    "vegetariano" → "vegetarian|vegan"
    "indiano" → "indian"
    "coreano" → "korean"
    "mexicano" → "mexican"
    "italiano" → "italian|pizza|pasta"
  Coloque null SOMENTE se cuisine for null ou amenity for "cafe".
•⁠  ⁠"amenity": "cafe" para cafés/padarias, "restaurant" para demais
•⁠  ⁠"neighborhood": um de ["asa sul","asa norte","lago sul","lago norte","sudoeste","noroeste","águas claras","taguatinga","guará"] ou null
•⁠  ⁠"friendly_message": mensagem curta dizendo que está buscando

Exemplo de saída para "quero comer japonês na asa norte":
{"cuisine":"japonês","cuisine_osm_regex":"japanese|sushi|temaki","amenity":"restaurant","neighborhood":"asa norte","friendly_message":"Buscando restaurantes japoneses na Asa Norte..."}"""


# ── REVISÃO/VALIDAÇÃO DOS RESULTADOS ─────────────────────────────────────────
# Prompt usado para a 2ª passada do LLM: ele avalia CADA candidato antes de
# devolvermos ao usuário, decidindo se o lugar realmente é um restaurante/café
# E se combina com o que foi pedido, além de escrever uma descrição melhor.
REVIEW_PROMPT = """Você é um avaliador de qualidade de um buscador de restaurantes em Brasília.
Recebe o pedido original do usuário e uma lista de candidatos (nome, categoria/cuisine, bairro, endereço).

Para CADA candidato, decida:
1.⁠ ⁠"keep": true APENAS se o lugar for de fato um estabelecimento gastronômico
   (restaurante, café, bar, lanchonete, padaria, confeitaria) E fizer sentido para o pedido.
   Use false quando:
     - claramente NÃO é comida (mercado, supermercado, farmácia, posto, loja, banco, hotel sem restaurante);
     - o usuário pediu uma culinária específica e este lugar NÃO corresponde a ela.
2.⁠ ⁠"relevance": número de 0 a 100 indicando o quão bem o lugar atende ao pedido
   (100 = exatamente a culinária/tipo pedido; valores baixos = pouco relevante).
3.⁠ ⁠"description": UMA a DUAS frases naturais em português descrevendo o lugar.
   Use os dados fornecidos para deixá-la informativa: tipo de cozinha/categorias,
   bairro e, QUANDO vierem preenchidos, a faixa de preço (price) e o horário (opening_hours).
   Ex.: "Restaurante japonês na Asa Norte, com faixa de preço $$ — aberto agora."
   Cite um campo SOMENTE se ele estiver presente nos dados; se vier vazio/null, ignore-o.
   NUNCA invente pratos, notas, avaliações, telefones ou qualquer informação que não foi dada.

Responda SOMENTE com JSON válido, sem markdown, no formato:
{"results":[{"index":0,"keep":true,"relevance":90,"description":"..."}]}
Inclua exatamente um objeto para cada candidato recebido, usando o mesmo "index"."""


# Filtro heurístico barato (roda ANTES do LLM): elimina de cara o que
# obviamente não é um lugar de comida, economizando tokens e reduzindo ruído.
_NON_FOOD_HINTS = (
    "grocery", "supermarket", "convenience store", "pharmacy", "drugstore",
    "gas station", "bank", "atm", "hotel", "hostel", "gym", "fitness",
    "mercado", "supermercado", "farmácia", "farmacia", "drogaria",
    "posto", "banco", "academia", "loja",
)

def _looks_like_food_place(r: dict) -> bool:
    """Heurística conservadora: só rejeita quando a categoria/nome aponta
    claramente para algo que não é comida. Em dúvida, mantém (o LLM decide)."""
    blob = f"{(r.get('cuisine') or '')} {(r.get('name') or '')}".lower()
    return not any(hint in blob for hint in _NON_FOOD_HINTS)


# ── Mapa de fallback: se o LLM não gerar cuisine_osm_regex, usamos este dicionário
CUISINE_REGEX_MAP: dict[str, str] = {
    "japonês": "japanese|sushi|temaki",
    "sushi": "sushi|japanese|temaki",
    "pizza": "pizza|italian",
    "italiano": "italian|pizza|pasta",
    "churrasco": "barbecue|churrascaria|brazilian|grill",
    "churrascarias": "barbecue|churrascaria|brazilian|grill",
    "burger": "burger|hamburger|american",
    "hamburguer": "burger|hamburger|american",
    "chinês": "chinese|asian",
    "árabe": "arab|lebanese|middle_eastern",
    "libanês": "arab|lebanese|middle_eastern",
    "frutos do mar": "seafood|fish",
    "seafood": "seafood|fish",
    "vegetariano": "vegetarian|vegan",
    "vegano": "vegan|vegetarian",
    "indiano": "indian",
    "coreano": "korean",
    "mexicano": "mexican",
    "frango": "chicken|frango|brazilian",
    "lanche": "sandwich|burger|fastfood",
    "brasileiro": "brazilian|regional",
    "regional": "regional|brazilian",
}


# ── Bounding boxes dos bairros ───────────────────────────────────────────────
NEIGHBORHOODS = {
    "asa sul":      (-15.830, -47.930, -15.790, -47.880),
    "asa norte":    (-15.780, -47.920, -15.740, -47.870),
    "lago sul":     (-15.880, -47.910, -15.820, -47.840),
    "lago norte":   (-15.760, -47.910, -15.710, -47.850),
    "sudoeste":     (-15.820, -47.960, -15.785, -47.925),
    "noroeste":     (-15.760, -47.960, -15.730, -47.920),
    "águas claras": (-15.870, -48.060, -15.820, -48.000),
    "taguatinga":   (-15.860, -48.080, -15.810, -48.030),
    "guará":        (-15.840, -47.990, -15.800, -47.940),
}
BRASILIA_BBOX = (-15.890, -48.100, -15.580, -47.600)


# ── MUDANÇA 1: Utilitário bbox → círculo ────────────────────────────────────
# Foursquare e HERE Maps usam ponto central + raio em vez de bounding box.
def bbox_to_circle(bbox: tuple) -> tuple[float, float, int]:
    """Converte bounding box em (center_lat, center_lng, radius_metros)."""
    lat_min, lng_min, lat_max, lng_max = bbox
    center_lat = (lat_min + lat_max) / 2
    center_lng = (lng_min + lng_max) / 2
    # 1° lat ≈ 111 km; 1° lon em lat -15° ≈ 107 km
    radius_m = int(max(
        abs(lat_max - lat_min) * 111_000,
        abs(lng_max - lng_min) * 107_000,
    ) / 2)
    return center_lat, center_lng, max(500, min(radius_m, 10_000))


# ── MUDANÇA 2: Foursquare Places API v3 (fonte primária) ────────────────────
#
# Por que Foursquare em vez de continuar só com OSM?
#   - Tem dados de redes grandes (Outback, Madero, Spoleto…) que o OSM não tem
#   - Usa ML para mapear "japonês" → Japanese Restaurant sem precisar de regex
#   - Free tier: 1.000 chamadas/dia (suficiente para um projeto acadêmico)
#   - Cadastro: https://foursquare.com/developers → criar app → copiar API Key
#     e adicionar ao .env como FOURSQUARE_API_KEY=<sua_chave>
#
# Alternativas igualmente boas:
#   - HERE Places API (250.000 chamadas/mês grátis, ótima cobertura no Brasil)
#     https://developer.here.com/products/places
#   - SerpApi Google Maps (dados reais do Google, free tier 100/mês)
#     https://serpapi.com/google-maps-api  ← opção "API não oficial" do Google
#
async def query_foursquare(
    cuisine: str | None,
    bbox: tuple,
    amenity: str = "restaurant",
) -> list:
    api_key = os.getenv("FOURSQUARE_API_KEY")
    if not api_key:
        logger.info("FOURSQUARE_API_KEY não configurada — pulando para OSM")
        return []

    center_lat, center_lng, radius = bbox_to_circle(bbox)

    # Foursquare entende termos em português quando a busca está geolocalizada no Brasil
    query_term = cuisine if cuisine else ("café" if amenity == "cafe" else "restaurante")

    params: dict = {
        "query": query_term,
        "ll": f"{center_lat},{center_lng}",
        "radius": radius,
        "limit": 15,
        "fields": "name,geocodes,location,categories,price,tel,website,hours",
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                "https://api.foursquare.com/v3/places/search",
                params=params,
                headers={"Authorization": api_key, "Accept": "application/json"},
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Foursquare erro: {e}")
            return []

    logger.info(f"Foursquare retornou {len(data.get('results', []))} resultados")

    seen: set[str] = set()
    results = []
    for place in data.get("results", []):
        name = place.get("name")
        if not name or name in seen:
            continue
        seen.add(name)

        geocodes = place.get("geocodes", {}).get("main", {})
        lat = geocodes.get("latitude")
        lng = geocodes.get("longitude")
        if not lat or not lng:
            continue

        cats       = place.get("categories", [])
        cuisine_name = cats[0].get("name", "Variada") if cats else "Variada"
        # Todas as categorias (para dar mais contexto real ao revisor)
        all_categories = ", ".join(
            c.get("name") for c in cats if c.get("name")
        ) or cuisine_name
        location   = place.get("location", {})

        neighborhood_list = location.get("neighborhood") or []
        neighborhood = (
            neighborhood_list[0]
            if neighborhood_list
            else location.get("locality", "Brasília")
        )

        address = location.get("formatted_address") or " ".join(
            filter(None, [location.get("address"), location.get("postcode")])
        ) or None

        hours_obj = place.get("hours") or {}
        if "open_now" in hours_obj:
            opening_hours = "Aberto agora" if hours_obj["open_now"] else "Fechado agora"
        else:
            opening_hours = None

        results.append({
            "name": name,
            "description": f"Culinária {cuisine_name}",
            "cuisine": cuisine_name,
            "categories": all_categories,
            "priceRange": fsq_price(place.get("price")),
            "neighborhood": neighborhood,
            "lat": lat,
            "lng": lng,
            "address": address,
            "phone": place.get("tel"),
            "website": place.get("website"),
            "opening_hours": opening_hours,
        })

    random.shuffle(results)
    return results[:8]  # candidatos extras: a revisão depois corta no top 5


def fsq_price(tier: int | None) -> str:
    return {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}.get(tier, "$$")


# Palavras-chave para busca por nome quando a tag cuisine nao e suficiente
CUISINE_NAME_KEYWORDS: dict[str, str] = {
    "barbecue":     "churrasco|churrascaria|picanha|espeto|brasa",
    "churrascaria": "churrasco|churrascaria|picanha|espeto|brasa",
    "brazilian":    "churrasco|churrascaria|picanha|espeto|brasa",
    "burger":     "burger|hamburguer|hamburger|smash|lanche",
    "hamburger":  "burger|hamburguer|hamburger|smash|lanche",
    "american":   "burger|hamburguer|hamburger|smash|lanche",
}

def _name_filter_for_regex(cuisine_regex: str | None) -> str | None:
    if not cuisine_regex:
        return None
    for key, name_regex in CUISINE_NAME_KEYWORDS.items():
        if key in cuisine_regex.lower():
            return name_regex
    return None


# ── Busca na Overpass API (OSM) — mantida como fallback ─────────────────────
async def query_overpass(cuisine_regex: str | None, bbox: tuple, amenity: str = "restaurant") -> list:
    lat_min, lng_min, lat_max, lng_max = bbox
    cuisine_filter = f'["cuisine"~"{cuisine_regex}",i]' if cuisine_regex else ""
    name_regex     = _name_filter_for_regex(cuisine_regex)
    name_filter    = f'["name"~"{name_regex}",i]' if name_regex else ""
    area = f'({lat_min},{lng_min},{lat_max},{lng_max})'

    query = f"""[out:json][timeout:25];
(
  node["amenity"="{amenity}"]{cuisine_filter}{area};
  way["amenity"="{amenity}"]{cuisine_filter}{area};
  relation["amenity"="{amenity}"]{cuisine_filter}{area};
  node["amenity"="{amenity}"]{name_filter}{area};
  way["amenity"="{amenity}"]{name_filter}{area};
  relation["amenity"="{amenity}"]{name_filter}{area};
);
out center 50;"""

    logger.info(f"Overpass amenity={amenity!r} cuisine_filter={cuisine_filter!r} bbox={bbox}")

    async with httpx.AsyncClient(
        headers={"User-Agent": "RestauranteBrasilia/1.0", "Accept": "application/json"}
    ) as client:
        try:
            resp = await client.get(
                "https://overpass-api.de/api/interpreter",
                params={"data": query},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Overpass erro HTTP: {e.response.status_code}")
            return []
        except httpx.RequestError as e:
            logger.error(f"Overpass erro de conexão: {e}")
            return []

    elements = data.get("elements", [])
    logger.info(f"Overpass retornou {len(elements)} elementos")

    seen_names: set[str] = set()
    results = []
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name or name in seen_names:
            continue
        seen_names.add(name)

        lat = el.get("lat") or el.get("center", {}).get("lat")
        lng = el.get("lon") or el.get("center", {}).get("lon")
        if not lat or not lng:
            continue

        cuisine_tag = tags.get("cuisine", "Variada").replace(";", ", ").title()
        neighborhood = (
            tags.get("addr:suburb")
            or tags.get("addr:neighbourhood")
            or tags.get("addr:city_district")
            or "Brasília"
        )
        address = " ".join(filter(None, [
            tags.get("addr:street"),
            tags.get("addr:housenumber"),
        ])) or None

        # Pontua por relevância: 2 = culinária pedida é a principal, 1 = secundária
        import re as _re
        score = 0
        if cuisine_regex:
            raw_cuisine = tags.get("cuisine", "")
            primary = raw_cuisine.split(";")[0].strip()
            if _re.search(cuisine_regex, primary, _re.IGNORECASE):
                score = 2
            elif _re.search(cuisine_regex, raw_cuisine, _re.IGNORECASE):
                score = 1

        results.append({
            "name": name,
            "description": build_description(tags),
            "cuisine": cuisine_tag,
            "priceRange": osm_price(tags.get("price_range") or tags.get("level:cost")),
            "neighborhood": neighborhood,
            "lat": lat,
            "lng": lng,
            "address": address,
            "phone": tags.get("phone") or tags.get("contact:phone"),
            "website": tags.get("website") or tags.get("contact:website"),
            "opening_hours": tags.get("opening_hours"),
            "_score": score,
        })

# Se tem culinária pedida, prioriza os que têm culinária como principal
    if cuisine_regex:
        primary_results = [r for r in results if r["_score"] == 2]
        results = primary_results if primary_results else results

    # Ordena: culinária principal primeiro, depois secundária
    results.sort(key=lambda x: x.pop("_score"), reverse=True)
    return results[:8]  # candidatos extras: a revisão depois corta no top 5


def build_description(tags: dict) -> str:
    parts = []
    cuisine = tags.get("cuisine", "").replace(";", " e ").title()
    if cuisine:
        parts.append(f"Culinária {cuisine}")
    if tags.get("outdoor_seating") == "yes":
        parts.append("área externa")
    if tags.get("wheelchair") == "yes":
        parts.append("acessível")
    return ". ".join(parts) if parts else "Restaurante em Brasília."


def osm_price(tag: str | None) -> str:
    return {"1": "$", "2": "$$", "3": "$$$", "4": "$$$$"}.get(str(tag), "$$")


# Monta a mensagem de resposta já com nome + descrição de cada recomendação,
# para que o texto do chat traga as sugestões (não só a lista estruturada).
def build_reply_message(friendly: str, restaurants: list) -> str:
    if not restaurants:
        return friendly
    linhas = []
    for r in restaurants:
        nome = r.get("name", "Restaurante")
        desc = (r.get("description") or "").strip()
        bairro = (r.get("neighborhood") or "").strip()
        local = f" ({bairro})" if bairro and bairro.lower() != "brasília" else ""
        linhas.append(f"• {nome}{local} — {desc}" if desc else f"• {nome}{local}")
    intro = (
        f"Encontrei {len(restaurants)} "
        f"{'opção' if len(restaurants) == 1 else 'opções'} pra você:"
    )
    return intro + "\n\n" + "\n".join(linhas)


# ── Revisão dos candidatos com o LLM ─────────────────────────────────────────
# Recebe os candidatos crus (Foursquare/OSM) e devolve apenas os que passam na
# validação, já com descrições reescritas e ordenados por relevância.
# Estratégia em camadas:
#   1) filtro heurístico (descarta não-comida óbvio);
#   2) LLM avalia keep + relevance + description de cada um;
#   3) mantém só os aprovados, ordena por relevância, corta no top 5.
# Em caso de falha do LLM, degrada para a lista já filtrada pela heurística.
def review_restaurants(
    user_query: str,
    cuisine_input: str,
    amenity: str,
    candidates: list,
) -> list:
    if not candidates:
        return []

    # Camada 1: heurística barata
    prefiltered = [r for r in candidates if _looks_like_food_place(r)]
    if not prefiltered:
        logger.info("Revisão: todos os candidatos reprovados na heurística de não-comida")
        return []

    # Payload enxuto para o LLM (só o necessário para julgar)
    payload = [
        {
            "index": i,
            "name": r.get("name"),
            "cuisine": r.get("categories") or r.get("cuisine"),
            "neighborhood": r.get("neighborhood"),
            "address": r.get("address"),
            "price": r.get("priceRange"),
            "opening_hours": r.get("opening_hours"),
        }
        for i, r in enumerate(prefiltered)
    ]
    user_content = (
        f'Pedido do usuário: "{user_query}"\n'
        f'Culinária desejada: {cuisine_input or "não especificada"}\n'
        f'Tipo de estabelecimento: {amenity}\n'
        f'Candidatos:\n{json.dumps(payload, ensure_ascii=False)}'
    )

    # Camada 2: avaliação pelo LLM
    try:
        review_resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": REVIEW_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=1300,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        verdicts = json.loads(review_resp.choices[0].message.content).get("results", [])
    except Exception as e:
        logger.error(f"Revisão LLM falhou: {e} — usando lista pós-heurística")
        return prefiltered[:5]

    # Camada 3: aplica os vereditos
    reviewed = []
    for v in verdicts:
        idx = v.get("index")
        if not isinstance(idx, int) or not (0 <= idx < len(prefiltered)):
            continue
        if not v.get("keep", False):
            continue
        r = dict(prefiltered[idx])
        desc = (v.get("description") or "").strip()
        if desc:
            r["description"] = desc
        try:
            r["_relevance"] = float(v.get("relevance", 0))
        except (TypeError, ValueError):
            r["_relevance"] = 0.0
        reviewed.append(r)

    reviewed.sort(key=lambda x: x.get("_relevance", 0), reverse=True)
    for r in reviewed:
        r.pop("_relevance", None)

    logger.info(f"Revisão: {len(reviewed)}/{len(prefiltered)} candidatos aprovados pelo LLM")
    return reviewed[:5]


# ── Modelos ──────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

class Restaurant(BaseModel):
    name:          Annotated[str,        "nome do estabelecimento"]
    description:   Annotated[str,        "descrição gerada a partir das tags OSM"]
    cuisine:       Annotated[str,        "tipo de cozinha"]
    priceRange:    Annotated[str,        "faixa de preço ($, $$, $$$, $$$$)"]
    neighborhood:  Annotated[str,        "bairro"]
    lat:           Annotated[float,      "latitude do estabelecimento"]
    lng:           Annotated[float,      "longitude do estabelecimento"]
    address:       Annotated[str | None, "endereço formatado (rua + número)"] = None
    phone:         Annotated[str | None, "telefone de contato"]               = None
    website:       Annotated[str | None, "site do estabelecimento"]           = None
    opening_hours: Annotated[str | None, "horário de funcionamento"]          = None

class ChatResponse(BaseModel):
    message: str
    restaurants: list[Restaurant]


# ── Endpoint principal ───────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    last_user_msg = next(
        (m.content for m in reversed(request.messages) if m.role == "user"), ""
    )

    if not last_user_msg:
        raise HTTPException(status_code=400, detail="Mensagem vazia.")
    if len(last_user_msg) > 400:
        raise HTTPException(status_code=400, detail="Mensagem muito longa. Use até 400 caracteres.")

    history = [{"role": "system", "content": INTENT_PROMPT}]
    for m in request.messages:
        history.append({"role": m.role, "content": m.content})

    # 1. LLM extrai intenção
    try:
        intent_resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=history,
            max_tokens=200,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        intent = ModelOutput(**json.loads(intent_resp.choices[0].message.content))
    except Exception as e:
        logger.error(f"Groq erro: {e}")
        raise HTTPException(status_code=503, detail="Serviço de IA temporariamente indisponível. Tente novamente.")

    logger.info(f"Intent parsed: {intent}")

    cuisine_input = (intent.cuisine or "").lower().strip()
    hood          = (intent.neighborhood or "").lower().strip()
    amenity       = intent.amenity
    friendly      = intent.friendly_message

    # Se o LLM não gerou cuisine_osm_regex mas temos cuisine, usa o mapa de fallback
    cuisine_regex = intent.cuisine_osm_regex
    if not cuisine_regex and cuisine_input:
        cuisine_regex = CUISINE_REGEX_MAP.get(cuisine_input)
        if cuisine_regex:
            logger.info(f"cuisine_osm_regex gerado pelo mapa de fallback: {cuisine_regex}")

    # 2. Define bounding box
    bbox = BRASILIA_BBOX
    for key, coords in NEIGHBORHOODS.items():
        if key in hood:
            bbox = coords
            break

    # ── MUDANÇA 3: Nova estratégia de busca em cascata ───────────────────────
    #
    # ANTES (problema):
    #   OSM com filtro → sem resultado → OSM SEM filtro → aparecia Coco Bambu
    #   para quem pediu "japonês" porque o fallback ignorava a culinária pedida.
    #
    # AGORA (corrigido):
    #   Passo A — Foursquare (fonte primária, dados mais completos)
    #   Passo B — OSM COM filtro de culinária (se Foursquare não retornou nada)
    #   Passo C — OSM SEM filtro SOMENTE se o usuário NÃO pediu culinária específica
    #   Passo D — 404 com mensagem útil se ainda não achou nada

    # Passo A: Foursquare
    restaurants = await query_foursquare(cuisine_input or None, bbox, amenity)

    # Passo B: OSM com filtro (Foursquare indisponível ou sem resultado)
    if not restaurants:
        logger.info("Foursquare sem resultados, tentando Overpass com filtro...")
        restaurants = await query_overpass(cuisine_regex, bbox, amenity)

    # Passo C: OSM sem filtro — APENAS quando o usuário não pediu culinária específica.
    # Isso impede que restaurantes irrelevantes (ex.: Coco Bambu) apareçam
    # em buscas por "japonês", "árabe", etc.
    if not restaurants and not cuisine_input:
        logger.info("OSM sem resultados gerais, tentando sem filtro de culinária...")
        restaurants = await query_overpass(None, bbox, amenity)

    # ── Revisão/validação: avalia cada candidato (é mesmo restaurante? combina
    # com a busca?), reescreve a descrição e filtra antes de entregar ao usuário.
    if restaurants:
        restaurants = review_restaurants(last_user_msg, cuisine_input, amenity, restaurants)

    # Passo D: Sem resultados (nada encontrado OU nada aprovado na revisão)
    if not restaurants:
        if cuisine_input:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Não encontrei restaurantes de {cuisine_input} nessa região. "
                    "Tente um bairro diferente ou outra culinária."
                ),
            )
        raise HTTPException(
            status_code=404,
            detail="Nenhum restaurante encontrado na região. Tente outro bairro.",
        )

    reply = build_reply_message(friendly, restaurants)
    return ChatResponse(message=reply, restaurants=restaurants)


@app.get("/health")
async def health():
    return {"status": "ok"}