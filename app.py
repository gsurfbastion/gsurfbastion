import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
from cipriano import executar_agente
import uvicorn

app = FastAPI()

# Configura onde estão os arquivos HTML
templates = Jinja2Templates(directory="templates")

class Pergunta(BaseModel):
    pergunta: str
    imagem: Optional[str] = None  # Agora aceita imagem (opcional)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Exibe o site"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(payload: Pergunta):
    """
    Recebe JSON do site, chama o Cipriano
    e devolve um JSON organizado.
    """
    try:
        # Passamos a pergunta E a imagem (se houver) para o agente
        resposta_texto = executar_agente(payload.pergunta, payload.imagem)
        return {"resposta": resposta_texto}
    except Exception as e:
        return {"resposta": f"Erro estratégico: {str(e)}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)