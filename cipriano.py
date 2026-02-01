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
    
    search = TavilySearchResults(max_results=3) 
    return search.invoke(query)

tools = [search_web]

# ======================================================
# AGENT LOGIC - CIPRIANO
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
    
    # AJUSTE DO MODELO
    model = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", 
        temperature=0,
        api_key=api_key
    )
    
    agent = create_react_agent(
        model=model, 
        tools=tools, 
        prompt=system_message
    )
    
    inputs = {"messages": [("user", mensagem_usuario)]}
    config = {"configurable": {"thread_id": "thread-1"}}
    
    try:
        resultado = agent.invoke(inputs, config)
        
        # Acessamos a última mensagem
        ultima_mensagem = resultado["messages"][-1]
        
        # Lógica de extração robusta
        if hasattr(ultima_mensagem, 'content'):
            conteudo = ultima_mensagem.content
            
            # CASO 1: O conteúdo é uma string simples
            if isinstance(conteudo, str):
                return conteudo
                
            # CASO 2: O conteúdo é uma lista de blocos (Correção do JSON bruto)
            elif isinstance(conteudo, list):
                texto_final = "".join([bloco.get("text", "") for bloco in conteudo if bloco.get("type") == "text"])
                return texto_final
                
        # Fallback: converte o objeto inteiro para string se não achar .content
        return str(ultima_mensagem)

    except Exception as e:
        # Este é o bloco que estava faltando!
        return f"Cipriano informa: Falha na requisição. Verifique as credenciais. (Erro: {str(e)})"