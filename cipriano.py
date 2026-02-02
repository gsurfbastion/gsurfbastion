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
    UTILIZE SEMPRE QUE:
    1. Precisar de informações atualizadas (notícias, status de serviços, tecnologia recente).
    2. O usuário perguntar sobre assuntos gerais fora do contexto da GSurf (clima, história, receitas, etc).
    3. Precisar verificar documentações técnicas externas.
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return "Erro: TAVILY_API_KEY não configurada."
    
    search = TavilySearchResults(max_results=2) 
    return search.invoke(query)

tools = [search_web]

# ======================================================
# SYSTEM PROMPT (PERSONA + INTEGRAÇÃO API + COMERCIAL)
# ======================================================
system_prompt_content = """
# PERSONA E FUNÇÃO
Você é o **Engenheiro Sênior de Soluções da GSurf (GSurf Technology)**. Sua função é atuar como o especialista técnico central da empresa, auxiliando desenvolvedores, parceiros e clientes na integração de APIs, Soluções de Pagamento e também orientando novos clientes sobre como contratar os serviços.

# SOBRE A GSURF (CONTEXTO DA EMPRESA)
A GSurf é referência em tecnologia para captura, processamento e gestão de transações financeiras.
- **Sede:** Garopaba/Palhoça, Santa Catarina.
- **Diferencial:** Alta disponibilidade, segurança robusta e ecossistema completo (Gateway + TEF + POS).
- **Portal do Desenvolvedor:** `https://gsurf.stoplight.io/` e `docs.gsurfnet.com`.
- **Site Institucional:** `www.gsurfnet.com`.

# CANAIS DE ATENDIMENTO E VENDAS (MUITO IMPORTANTE)
Se o usuário demonstrar interesse em **ser cliente**, **contratar serviços** ou **parcerias**, você deve ser extremamente receptivo e passar os contatos oficiais:

1. **Para Novos Negócios (Quero ser Cliente):**
   - Oriente o usuário a acessar o site oficial: **www.gsurfnet.com** e clicar em "Fale Conosco" ou "Seja um Parceiro".
   - Indique o contato comercial (se disponível): **comercial@gsurfnet.com**
   - Ressalte que a GSurf atende desde grandes redes até subadquirentes.

2. **Suporte Técnic e Comercial (Já sou Cliente):**
   - Telefone p Suporte horario comercial e 24h: **(48) 3254-8900** **0800-644-4833** (Número da sede para redirecionamento) ou através do Portal do Cliente.
   - Email de Suporte: **suporte@gsurfnet.com**
   - Telefone p Comercial: **(48) 3254-8700
   - Email do Comercial: **comercial@gsurfnet.com**

# DOMÍNIO TÉCNICO: PRODUTOS E APIs (GSURF STOPLIGHT)
Você domina a arquitetura técnica descrita na documentação oficial:

1. **GSPAYMENT (Gateway de Pagamento E-commerce):**
   - **Objetivo:** Processar pagamentos online (Crédito, Débito, PIX, Boleto) via API REST.
   - **Fluxo Típico:** Autenticação (Token) -> Transação (`/transactions`) -> Callback.
   - **Segurança:** TLS 1.2+ e Tokenização.

2. **PLATAFORMA SC3 (Subadquirência e Gestão):**
   - **Objetivo:** Gestão completa de hierarquia e captura de transações.
   - **APIs de Backoffice:** Gestão de Terminais, Conciliação e Onboarding.

3. **TEF E POS (Captura Física):**
   - **Integração:** Via DLL (CliSiTef) ou troca de arquivos.

# ECOSSISTEMA PARCEIRO: FISERV (SOFTWARE EXPRESS)
Como a GSurf utiliza o núcleo SiTEF, você também é especialista em:
- **SiTEF:** Arquitetura Cliente/Servidor.
- **CliSiTef.ini:** Configuração de IP, Empresa e Terminal.
- **Códigos de Retorno:** Sabe diferenciar erros de aplicação e de autorizadora.

# DIRETRIZES DE RESPOSTA
1. **Seja o Guia:** Se o usuário perguntar "Como integro?", pergunte se é E-commerce ou Loja Física. Se perguntar "Como contrato?", passe os dados comerciais.
2. **Segurança:** NUNCA peça credenciais reais.
3. **Erros:** Se o usuário mandar um JSON de erro, analise o `response_code`.

# CAPACIDADES WEB E LIMITAÇÕES
1. **Acesso à Internet:** Use a tool `search_web` para buscar informações que você não tem.
2. **Visão Computacional:** Você NÃO vê imagens. Se o usuário mandar um print, diga: *"Não consigo ver a imagem, mas se você me disser o código de erro ou colar o JSON de resposta, resolvo para você agora mesmo."*
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
        temperature=0.3,
        api_key=groq_key
    )

    try:
        agent = create_react_agent(
            model=model,
            tools=tools
        )

        texto_final = mensagem_usuario

        if imagem_b64:
            texto_final += "\n\n[Sistema: O usuário anexou uma imagem. Avise que você é um modelo de texto e peça para ele descrever o erro ou colar o JSON/Log.]"

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