import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# ======================================================
# TOOL — Web Search (OSINT)
# ======================================================
@tool
def search_web(query: str) -> str:
    """Busca informações públicas e atuais na internet (OSINT)."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return "TAVILY_API_KEY não configurada no Render."
    
    # O Tavily utiliza a chave do ambiente automaticamente
    search = TavilySearchResults(max_results=3) 
    return search.invoke(query)

tools = [search_web]

# ======================================================
# AGENT LOGIC — CIPRIANO
# ======================================================
system_message = """
Você é Cipriano, um agente estratégico de Inteligência e Segurança,
com a presença, autoridade e frieza calculada de Don Corleone.
- Seja direto e use tom formal/imponente.
- Atue apenas em contextos legais e éticos.
"""

def executar_agente(mensagem_usuario: str):
    """Função de interface com o app.py"""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        return "Erro: GOOGLE_API_KEY não encontrada no ambiente."
    
    # Modelo atualizado para versão estável
    model = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", 
        temperature=0,
        api_key=api_key
    )
    
    # Criação do agente com o parâmetro correto 'prompt'
    agent = create_react_agent(
        model=model, 
        tools=tools, 
        prompt=system_message
    )
    
    # Preparação da entrada
    inputs = {"messages": [("user", mensagem_usuario)]}
    config = {"configurable": {"thread_id": "thread-1"}}
    
    try:
        # Execução do grafo
        resultado = agent.invoke(inputs, config)
        
        # AJUSTE DE EXTRAÇÃO:
        # Pegamos a última mensagem da lista 'messages'
        ultima_mensagem = resultado["messages"][-1]
        
        # Retornamos apenas o conteúdo textual (string)
        return ultima_mensagem.content
        
    except Exception as e:
        return f"Cipriano informa: Erro na operação técnica. Detalhes: {str(e)}"