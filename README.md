# 🍽️ Chatbot Restaurantes Brasília

> ⚠️ **AVISO: MODELO INICIAL / PROVA DE CONCEITO**
>
> Este projeto é um **primeiro teste exploratório** e **não deve ser usado em produção**.
> O objetivo é validar a ideia e a integração entre as tecnologias.
> Funcionalidades, estrutura de código e decisões de arquitetura estão sujeitas a grandes mudanças nas próximas versões.

**O que faz hoje:** chatbot que recebe uma mensagem do usuário (ex: "quero sushi na Asa Norte"), entende a intenção via LLM, busca restaurantes reais no OpenStreetMap e exibe os resultados em um mapa interativo.

**Visão futura:** evoluir para um assistente que conecta usuários a experiências gastronômicas de forma mais completa — aprendendo preferências ao longo do tempo e integrando dados reais dos restaurantes (cardápio, avaliações, horários).

## Stack

- **Backend**: Python + FastAPI + Groq SDK + httpx
- **Frontend**: HTML/JS + Leaflet.js
- **LLM**: Llama 3.3 70B via Groq (gratuito)
- **Mapa**: OpenStreetMap via Leaflet.js (gratuito, sem chave de API)
- **Dados de restaurantes**: Overpass API via OpenStreetMap (gratuito, sem cadastro)

## Como funciona a arquitetura

```
usuário digita mensagem
        ↓
  Groq (Llama 3.3) extrai intenção
  { cuisine: "japanese", neighborhood: "asa norte" }
        ↓
  Overpass API busca restaurantes reais no OpenStreetMap
        ↓
  Retorna dados reais com coordenadas reais para o mapa
```

## Estrutura

```
restaurante-bot/
├── backend/
│   ├── main.py           ← API FastAPI
│   ├── requirements.txt
│   └── .env              ← criar manualmente (não subir no git)
└── frontend/
    └── index.html        ← Interface com mapa
```

## Como rodar

### 1. Obter chave do Groq (gratuito)

Acesse https://console.groq.com/keys e crie uma chave.

### 2. Criar o arquivo `.env` na pasta `backend/`

```env
GROQ_API_KEY=gsk_SUA_CHAVE_AQUI
```

### 3. Instalar dependências

```bash
cd backend
pip install -r requirements.txt
```

### 4. Rodar o servidor

```bash
uvicorn main:app --reload --port 8000
```

A API estará disponível em: `http://localhost:8000`

Documentação automática: `http://localhost:8000/docs`

### 5. Abrir o frontend

Abra o arquivo `frontend/index.html` diretamente no navegador. Não precisa de servidor.

### 6. Testar a API

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Quero um restaurante japonês na Asa Norte"}
    ]
  }'
```

## Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| POST | `/chat` | Envia mensagem e recebe recomendações |
| GET | `/health` | Verifica se o servidor está no ar |

## Deploy

Para produção, recomendamos:
- **Backend**: Railway, Render, ou EC2
- **Frontend**: Vercel, Netlify, ou servir com FastAPI StaticFiles
- **Variável de ambiente**: Configure `GROQ_API_KEY` no painel do seu provedor cloud
