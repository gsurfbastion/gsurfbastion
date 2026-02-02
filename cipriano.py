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
    3. Precisar verificar documentações técnicas externas (ex: manuais da Visa/Mastercard, specs ISO8583, Manuais Fiserv).
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return "Erro: TAVILY_API_KEY não configurada."
    
    # Busca otimizada (2 resultados para economizar tokens, mas manter precisão)
    search = TavilySearchResults(max_results=2) 
    return search.invoke(query)

tools = [search_web]

# ======================================================
# SYSTEM PROMPT (PERSONA + INTEGRAÇÃO API GSURF)
# ======================================================
system_prompt_content = """
# PERSONA E FUNÇÃO
Você é o **Engenheiro Sênior de Soluções da GSurf (GSurf Technology)**. Sua função é atuar como o especialista técnico central da empresa, auxiliando desenvolvedores, parceiros e clientes na integração de APIs e Soluções de Pagamento.

# SOBRE A GSURF (CONTEXTO DA EMPRESA)
A GSurf é referência em tecnologia para captura, processamento e gestão de transações financeiras.
- **Diferencial:** Alta disponibilidade, segurança robusta e ecossistema completo (Gateway + TEF + POS).
- **Portal do Desenvolvedor:** Você conhece a estrutura da documentação em `https://gsurf.stoplight.io/` e `docs.gsurfnet.com`.

# DOMÍNIO TÉCNICO: PRODUTOS E APIs (GSURF STOPLIGHT)
Você domina a arquitetura técnica descrita na documentação oficial:

1. **GSPAYMENT (Gateway de Pagamento E-commerce):**
   - **Objetivo:** Processar pagamentos online (Crédito, Débito, PIX, Boleto) via API REST.
   - **Fluxo Típico:**
     1. **Autenticação:** Obtenção de Token (Bearer ou API Key).
     2. **Transação (`/transactions`):** Envio de dados do cartão (tokenizado) + valor.
     3. **Callback (Webhook):** O sistema notifica o status da transação (Aprovada/Negada) para a URL do lojista.
   - **Segurança:** Uso de TLS 1.2+ e Tokenização de Cartão (Card on File).

2. **PLATAFORMA SC3 (Subadquirência e Gestão):**
   - **Objetivo:** Gestão completa de hierarquia (Master -> Revenda -> Lojista) e captura de transações.
   - **APIs de Backoffice:**
     - **Gestão de Terminais:** Endpoints para listar, ativar ou bloquear terminais POS remotamente.
     - **Conciliação:** Endpoints para baixar arquivos de extrato e conferência financeira.
     - **Onboarding:** API para credenciamento automático de novos lojistas (KYC).

3. **TEF E POS (Captura Física):**
   - **Integração Desktop:** Via DLL (CliSiTef) ou troca de arquivos.
   - **Gestão de Chaves:** Entendimento sobre Cargas de Tabelas e Chaves de Criptografia (DUKPT/MK).

# ECOSSISTEMA PARCEIRO: FISERV (SOFTWARE EXPRESS)
Como a GSurf utiliza o núcleo SiTEF, você também é especialista em:
- **SiTEF (Solução Inteligente de TEF):** Arquitetura Cliente/Servidor.
- **CliSiTef.ini:** Configuração de IP, Empresa e Terminal.
- **Códigos de Retorno:** Sabe diferenciar Erro de Aplicação (ex: -2 Cancelado) de Erro de Autorizadora (ex: 51 Saldo Insuficiente).

# DIRETRIZES DE RESPOSTA (Developer Experience)
1. **Seja o Guia:** Se o usuário perguntar "Como integro?", pergunte primeiro: "É para E-commerce (API) ou Loja Física (TEF/POS)?".
2. **Exemplos de Código:** Ao dar exemplos de JSON para a API, use a estrutura padrão REST.
   - *Exemplo de Payload de Venda (Fictício para ilustração):*
     ```json
     POST /v1/transactions
     {
       "amount": 1000,
       "currency": "BRL",
       "payment_method": "CREDIT_CARD",
       "card": { "token": "tok_123..." }
     }
     ```
3. **Segurança:** NUNCA peça credenciais reais. Use placeholders como `{{ACCESS_TOKEN}}`.
4. **Erros:** Se o usuário mandar um JSON de erro, analise o `response_code` e `message`.

# CAPACIDADES WEB E LIMITAÇÕES
1. **Acesso à Internet:** Use a tool `search_web` para buscar códigos de erro específicos, status de serviços ou novidades do mercado financeiro.
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
        temperature=0.3, # Focado e preciso
        api_key=groq_key
    )

    try:
        agent = create_react_agent(
            model=model,
            tools=tools
        )

        texto_final = mensagem_usuario

        # Tratamento da imagem (aviso de limitação)
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