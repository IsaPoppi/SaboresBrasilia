from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Annotated, Literal
from groq import Groq
from dotenv import load_dotenv
import json, os, httpx, random, logging
import instructor




load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chatbot Restaurantes Brasília")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def root():
    return {"message": "API funcionando!"}

_groq_api_key = os.getenv("GROQ_API_KEY")
if not _groq_api_key:
    raise RuntimeError("GROQ_API_KEY não encontrada. Crie um arquivo .env com GROQ_API_KEY=sua_chave.")
groq_client = Groq(api_key=_groq_api_key, timeout=15.0)
groq_client = instructor.from_groq(
   client=groq_client
)

#MUDANÇA 3 — Dicionários manuais removidos; o LLM agora gera diretamente
# cuisine_osm_regex e amenity como campos do ModelOutput.

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

# ── Prompt para extrair intenção ─────────────────────────────────────────────
INTENT_PROMPT = f"""Você extrai intenções de busca de restaurantes e cafés em Brasília.
Responda SOMENTE com JSON válido seguindo este schema, sem markdown:
{json.dumps(ModelOutput.model_json_schema(), ensure_ascii=False, indent=2)}"""


#MUDANÇA 1 
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


# ── Busca na Overpass API ────────────────────────────────────────────────────
async def query_overpass(cuisine_regex: str | None, bbox: tuple, amenity: str = "restaurant") -> list:
    lat_min, lng_min, lat_max, lng_max = bbox
    cuisine_filter = f'["cuisine"~"{cuisine_regex}",i]' if cuisine_regex else ""
    area = f'({lat_min},{lng_min},{lat_max},{lng_max})'

    # Busca node + way + relation para não perder restaurantes cadastrados como polígono
    query = f"""[out:json][timeout:25];
(
  node["amenity"="{amenity}"]{cuisine_filter}{area};
  way["amenity"="{amenity}"]{cuisine_filter}{area};
  relation["amenity"="{amenity}"]{cuisine_filter}{area};
);
out center 50;"""

    logger.info(f"Overpass query amenity={amenity!r} cuisine_filter={cuisine_filter!r} bbox={bbox}")

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

        # Nodes têm lat/lon direto; ways/relations têm centro em el["center"]
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
        })

    random.shuffle(results)
    return results[:5]


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


# ── Modelos ──────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

class Restaurant(BaseModel):
    name:          Annotated[str,        "nome do estabelecimento"]
    description:   Annotated[str,        "descrição gerada a partir das tags OSM"]
    cuisine:       Annotated[str,        "tipo de cozinha extraído da tag cuisine do OSM"]
    priceRange:    Annotated[str,        "faixa de preço ($, $$, $$$, $$$$)"]
    neighborhood:  Annotated[str,        "bairro extraído das tags de endereço do OSM"]
    lat:           Annotated[float,      "latitude do estabelecimento"]
    lng:           Annotated[float,      "longitude do estabelecimento"]
    address:       Annotated[str | None, "endereço formatado (rua + número)"] = None
    phone:         Annotated[str | None, "telefone de contato"]               = None
    website:       Annotated[str | None, "site do estabelecimento"]           = None
    opening_hours: Annotated[str | None, "horário de funcionamento (padrão OSM)"] = None

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

    history = [
        {"role": "system", "content": INTENT_PROMPT},
    ]
    for m in request.messages:
        history.append({"role": m.role, "content": m.content})

    # 1. LLM extrai intenção
    try:
        intent = groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=history,
            max_tokens=200,
            temperature=0.2,
            response_model=ModelOutput,
        )
    except Exception as e:
        logger.error(f"Groq erro: {e}")
        raise HTTPException(status_code=503, detail="Serviço de IA temporariamente indisponível. Tente novamente.")

    logger.info(f"Intent parsed: {intent}")

    cuisine_input  = (intent.cuisine or "").lower().strip()
    hood           = (intent.neighborhood or "").lower().strip()
    friendly       = intent.friendly_message

    logger.info(f"Intent extraído: cuisine={cuisine_input!r} neighborhood={hood!r}")

    # 2. Usa amenity e cuisine_osm_regex gerados diretamente pelo LLM
    amenity        = intent.amenity
    cuisine_regex  = intent.cuisine_osm_regex
    logger.info(f"amenity={amenity!r} cuisine_regex={cuisine_regex!r}")

    # 3. Define bounding box
    bbox = BRASILIA_BBOX
    for key, coords in NEIGHBORHOODS.items():
        if key in hood:
            bbox = coords
            break

    # 4. Busca dados reais
    restaurants = await query_overpass(cuisine_regex, bbox, amenity)

    # 5. Fallback sem filtro de culinária (informa o usuário)
    fallback_used = False
    if not restaurants and cuisine_regex:
        logger.info("Nenhum resultado com filtro, tentando sem filtro...")
        restaurants = await query_overpass(None, bbox, amenity)
        fallback_used = True

    if not restaurants:
        raise HTTPException(status_code=404, detail="Nenhum restaurante encontrado na região. Tente outro bairro ou tipo de culinária.")

    if fallback_used and cuisine_input:
        friendly = f"Não encontrei {cuisine_input} especificamente na região, mas aqui estão outros lugares que podem te agradar:"

    return ChatResponse(message=friendly, restaurants=restaurants)


@app.get("/health")
async def health():
    return {"status": "ok"}
