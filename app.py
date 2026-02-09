import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

# Importamos a função do arquivo cipriano.py
from cipriano import executar_agente

app = FastAPI(title="GSurf Bastion IA")
templates = Jinja2Templates(directory="templates")

class RequestData(BaseModel):
    pergunta: str
    imagem: Optional[str] = None
    session_id: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/conhecimento")
async def obter_grafo():
    """
    Retorna a estrutura do grafo. 
    Dica Sênior: Você pode expandir isso para verificar status de APIs em tempo real.
    """
    return {
        "nodes": [
            {"data": {"id": "gsurf", "label": "GSurf Core", "status": "online"}},
            {"data": {"id": "tef", "label": "TEF IP", "status": "online"}},
            {"data": {"id": "msitef", "label": "M-SiTEF", "status": "online"}},
            {"data": {"id": "adq", "label": "Adquirentes", "status": "online"}},
            {"data": {"id": "erro", "label": "Diagnóstico", "status": "online"}}
        ],
        "edges": [
            {"data": {"source": "gsurf", "target": "tef"}},
            {"data": {"source": "tef", "target": "msitef"}},
            {"data": {"source": "msitef", "target": "adq"}},
            {"data": {"source": "gsurf", "target": "erro"}}
        ]
    }

@app.post("/chat")
async def chat_endpoint(payload: RequestData):
    try:
        resposta_texto = executar_agente(
            mensagem_usuario=payload.pergunta, 
            imagem_b64=payload.imagem,
            session_id=payload.session_id
        )
        return {"resposta": resposta_texto}
    except Exception as e:
        return {"resposta": f"⚠️ **Erro:** {str(e)}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)