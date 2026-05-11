from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import json, os, re, httpx, random, logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chatbot Restaurantes Brasília")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_groq_api_key = os.getenv("GROQ_API_KEY")
if not _groq_api_key:
    raise RuntimeError("GROQ_API_KEY não encontrada. Crie um arquivo .env com GROQ_API_KEY=sua_chave.")
groq_client = Groq(api_key=_groq_api_key, timeout=15.0)

# ── Mapeamento de intenções para tags OSM ────────────────────────────────────
# OSM usa tags variadas; esse dicionário expande a busca com regex
CUISINE_MAP = {
    "japonês":       "japanese|sushi|temaki",
    "japanese":      "japanese|sushi|temaki",
    "sushi":         "japanese|sushi|temaki",
    "pizza":         "pizza|italian",
    "italiano":      "italian|pizza",
    "churrasco":     "barbecue|churrascaria|brazilian",
    "barbecue":      "barbecue|churrascaria|brazilian",
    "brasileiro":    "brazilian|regional|barbecue",
    "vegetariano":   "vegetarian|vegan",
    "vegano":        "vegan|vegetarian",
    "hamburguer":    "burger|american",
    "burger":        "burger|american",
    "chines":        "chinese|asian",
    "chinês":        "chinese|asian",
    "chinese":       "chinese|asian",
    "árabe":         "arab|lebanese|middle_eastern",
    "arab":          "arab|lebanese|middle_eastern",
    "frutos do mar": "seafood|fish",
    "seafood":       "seafood|fish",
    "mexicano":      "mexican",
    "thai":          "thai|asian",
    "indiano":       "indian",
    "indian":        "indian",
    "coreano":       "korean",
    "korean":        "korean",
    "peruano":       "peruvian",
    "frango":        "chicken|portuguese",
    "lanche":        "sandwich|burger|fastfood",
    "sanduíche":     "sandwich|burger",
    "fast food":     "burger|fastfood|sandwich",
}

# Alguns tipos de estabelecimento usam amenity=cafe no OSM, não amenity=restaurant
AMENITY_MAP = {
    "café":          "cafe",
    "cafe":          "cafe",
    "cafeteria":     "cafe",
    "café da manhã": "cafe",
    "coffee":        "cafe",
    "padaria":       "cafe",
    "confeitaria":   "cafe",
}

# ── Prompt para extrair intenção ─────────────────────────────────────────────
INTENT_PROMPT = """Você extrai intenções de busca de restaurantes e cafés em Brasília.
Responda SOMENTE com JSON válido, sem markdown:
{
  "cuisine": "tipo de cozinha ou estabelecimento em português ou inglês (ex: japonês, pizza, churrasco, italiano, vegetariano, burger, chinês, árabe, seafood, mexicano, café, cafeteria, padaria, indiano, coreano, lanche, frango) ou null se não especificado",
  "neighborhood": "bairro mencionado em lowercase (ex: asa sul, asa norte, lago sul, lago norte, sudoeste, noroeste, águas claras, taguatinga) ou null",
  "friendly_message": "mensagem de 1 frase dizendo que está buscando os lugares"
}"""

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
    name: str
    description: str
    cuisine: str
    priceRange: str
    neighborhood: str
    lat: float
    lng: float
    address: str | None = None
    phone: str | None = None
    website: str | None = None
    opening_hours: str | None = None

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

    # 1. LLM extrai intenção
    try:
        intent_resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": INTENT_PROMPT},
                {"role": "user", "content": last_user_msg},
            ],
            max_tokens=200,
            temperature=0.2,
        )
        raw = intent_resp.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq erro: {e}")
        raise HTTPException(status_code=503, detail="Serviço de IA temporariamente indisponível. Tente novamente.")

    logger.info(f"Intent raw: {raw}")

    try:
        intent = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        try:
            intent = json.loads(match.group()) if match else {}
        except (json.JSONDecodeError, AttributeError):
            logger.warning(f"Não foi possível parsear intent: {raw!r}")
            intent = {}

    cuisine_input = (intent.get("cuisine") or "").lower().strip()
    hood          = (intent.get("neighborhood") or "").lower().strip()
    friendly      = intent.get("friendly_message", "Buscando restaurantes para você...")

    logger.info(f"Intent extraído: cuisine={cuisine_input!r} neighborhood={hood!r}")

    # 2. Determina amenity (cafe vs restaurant) e traduz culinária para regex OSM
    amenity = AMENITY_MAP.get(cuisine_input, "restaurant")
    cuisine_regex = CUISINE_MAP.get(cuisine_input) or (cuisine_input if cuisine_input else None)
    # Para amenity=cafe não filtramos por cuisine tag (cafés raramente usam essa tag no OSM)
    if amenity == "cafe":
        cuisine_regex = None
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
