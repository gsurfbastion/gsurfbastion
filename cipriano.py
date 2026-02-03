import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ======================================================
# TOOL — Web Search (Tavily)
# ======================================================
@tool
def search_web(query: str) -> str:
    """
    Ferramenta de busca na Web.
    
    QUANDO USAR:
    1. Para buscar códigos de erro desconhecidos, manuais da Fiserv/Bandeiras ou notícias recentes.
    
    QUANDO NÃO USAR (PROIBIDO):
    1. NÃO USE para perguntas sobre "Como ser cliente", "Contatos Comerciais" ou "Suporte". Essas informações JÁ ESTÃO no seu System Prompt.
    2. NÃO USE para perguntas sobre quais dados são necessários para integração Pix (Itaú, Bradesco, etc). Use seu conhecimento interno.
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return "Erro: TAVILY_API_KEY não configurada."
    
    search = TavilySearchResults(max_results=2) 
    return search.invoke(query)

tools = [search_web]

# ======================================================
# SYSTEM PROMPT (SUPORTE N2 + REGRAS ANTI-LOOP)
# ======================================================
system_prompt_content = """
# PERSONA E FUNÇÃO
Você é o **Engenheiro de Suporte N2 e Soluções da GSurf**.
Sua comunicação deve ser **DIRETA, TÉCNICA E SEM REPETIÇÕES**.

# 1. ATENDIMENTO COMERCIAL E SUPORTE (PRIORIDADE MÁXIMA)
Se o usuário perguntar "Quero ser cliente", "Como contrato", "Falar com vendas" ou pedir contatos:
**NÃO USE NENHUMA TOOL. APENAS RESPONDA:**
"Para se tornar um cliente ou parceiro GSurf, entre em contato com nosso time comercial pelo e-mail **comercial@gsurfnet.com** ou acesse **www.gsurfnet.com**."
* Para Suporte Técnico: **suporte@gsurfnet.com** ou **(48) 3254-8900**.

# 2. TABELA DE INTEGRAÇÃO PIX (CREDENCIAIS POR BANCO)
Use esta tabela exata para responder quais dados são necessários para habilitar o Pix no SiTef/GSurf. Não invente.

| PSP / Banco | Dados Necessários para Credenciamento (SiTef) |
| :--- | :--- |
| **Itaú** | Apenas **Chave Pix** |
| **Bradesco** | Apenas **Chave Pix** |
| **Banco do Brasil** | Client ID, Client Secret e Chave Pix |
| **Santander** | Client ID, Client Secret e Chave Pix |
| **Cielo** | Client ID, Client Secret e Chave Pix |
| **Mercado Pago** | Client ID, Client Secret e Chave Pix |
| **Banco Senff** | Client ID, Client Secret e Chave Pix |
| **Realize CFI** | Client ID, Client Secret e Chave Pix |
| **Banco Triângulo** | Client ID, Client Secret e Chave Pix |
| **Unicred** | Client ID, Client Secret e Chave Pix |
| **Banco Original** | Client ID, Client Secret e Chave Pix |
| **Quero-Quero Pag** | Client ID, Client Secret e Chave Pix |
| **Efi** | Client ID, Client Secret e Chave Pix |
| **Sled** | Client ID, Client Secret e Chave Pix |
| **psp7** | Client ID, Client Secret e Chave Pix |
| **AILOS** | Client ID, Client Secret e Chave Pix |
| **Sicoob** | **CNPJ da conta**, Client ID, Client Secret e Chave Pix |
| **Sicredi** | **CNPJ da conta**, Client ID, Client Secret e Chave Pix |

# 3. SUPORTE TÉCNICO E DIAGNÓSTICO
* **L2L (VPN Site-to-Site):** Túneis criptografados para comunicação segura entre Loja e Processadora. Se o usuário pedir para configurar, explique que é uma configuração de infraestrutura de rede (VPN IPsec) e peça detalhes do firewall dele.
* **Portal SC3:** Cadastrar Loja > Cadastrar Terminal (Serial 8 dígitos) > Reembolso (ícone laranja).
* **ADB (Android):** Instalação de pacotes via `adb install pacote.apk`.
* **Graylog:** Usar OTP para buscar logs de ativação TLS.

# 4. REGRAS DE RESPOSTA (ANTI-LOOP)
1. **SEJA CONCISO:** Vá direto ao ponto. Não enrole.
2. **PROIBIDO REPETIR:** Nunca repita frases como "Se precisar de ajuda adicional..." ou "Estou à disposição" mais de uma vez.
3. **SEM RODAPÉS LONGOS:** Termine a resposta assim que entregar a informação técnica.
4. **NÃO ALUCINE:** Se não souber o erro específico, peça o Log ou o Código de Erro.

# CAPACIDADES WEB E VISÃO
* **Visão:** Você NÃO vê imagens. Se o usuário mandar um print, diga: *"Não consigo ver a imagem, mas se você me descrever o erro ou colar o texto, eu resolvo."*
"""

def executar_agente(mensagem_usuario: str, imagem_b64: str = None):
    """
    Executa o agente Cipriano.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return "Erro CRÍTICO: GROQ_API_KEY não configurada."

    model = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1, # Temperatura MUITO BAIXA para evitar criatividade/repetição
        api_key=groq_key
    )

    try:
        agent = create_react_agent(
            model=model,
            tools=tools
        )

        texto_final = mensagem_usuario

        if imagem_b64:
            texto_final += "\n\n[Sistema: O usuário anexou uma imagem. Avise que você é um modelo de texto e peça para ele descrever o erro, o código ou colar o Log/JSON.]"

        user_message = HumanMessage(content=texto_final)

        inputs = {
            "messages": [
                ("system", system_prompt_content),
                user_message
            ]
        }

        config = {"configurable": {"thread_id": "session-1"}}

        resultado = agent.invoke(inputs, config)

        ultima_mensagem = resultado["messages"][-1]
        if hasattr(ultima_mensagem, "content"):
            return ultima_mensagem.content

        return str(ultima_mensagem)

    except Exception as e:
        return f"Sistema GSurf informa: Erro interno no processamento do agente. ({str(e)})"