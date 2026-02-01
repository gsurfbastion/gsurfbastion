import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from cipriano import executar_agente
import uvicorn

app = FastAPI()

# Configura onde estão os arquivos HTML
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Exibe o site (Tema Dark)"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(pergunta: str = Form(...)):
    """
    Recebe o formulário do site, chama o Cipriano
    e devolve um JSON organizado.
    """
    try:
        # Chama a função que ajustamos no cipriano.py
        resposta_texto = executar_agente(pergunta)
        
        # Retorna um dicionário que o JS entende como objeto
        return {"resposta": resposta_texto}
    except Exception as e:
        return {"resposta": f"Erro estratégico: {str(e)}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)