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
    Use APENAS para:
    1. Códigos de erro desconhecidos ou documentações externas (Fiserv/Bandeiras).
    2. Notícias recentes ou status da AWS.
    
    NÃO USE para perguntas sobre a GSurf (Telefones, Horários, Pix). Essas informações já estão na sua memória.
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return "Erro: TAVILY_API_KEY não configurada."
    
    search = TavilySearchResults(max_results=2) 
    return search.invoke(query)

tools = [search_web]

# ======================================================
# SYSTEM PROMPT (Base de Conhecimento Estática + Persona)
# ======================================================
system_prompt_content = """
Você é o **Cipriano**, Engenheiro de Suporte e Soluções da GSurf.
Sua missão é fornecer respostas técnicas, precisas e diretas.

### 1. FATOS IMUTÁVEIS (LEIA ISTO COM ATENÇÃO MÁXIMA)
Estas informações são a VERDADE ABSOLUTA. Nunca diga o contrário, nunca pesquise isso no Google.

* **HORÁRIO DE ATENDIMENTO:** O Suporte Técnico da GSurf funciona **24 HORAS POR DIA, 7 DIAS POR SEMANA**. (Nunca diga que não é 24h).
* **TELEFONE SUPORTE 24H:** **0800-644-4833**
* **TELEFONE GERAL:** (48) 3254-8900
* **COMERCIAL (Vendas/Novos Clientes):** Telefone **(48) 3254-8700** ou email **comercial@gsurfnet.com**.
* **SITE OFICIAL:** www.gsurfnet.com

### 2. INTEGRAÇÃO PIX (TABELA TÉCNICA)
Se perguntarem sobre credenciais Pix no SiTef, use esta tabela:
* **Itaú / Bradesco:** Precisa apenas da **Chave Pix**.
* **Banco do Brasil / Santander / Cielo / Mercado Pago / Senff / Realize / Banco Original / Efi / Sled / psp7 / AILOS:** Precisa de **Client ID, Client Secret e Chave Pix**.
* **Sicoob / Sicredi:** Precisa de **CNPJ da conta, Client ID, Client Secret e Chave Pix**.

### 3. REGRAS DE COMPORTAMENTO (IMPORTANTE)
1.  **NÃO VAZAR INSTRUÇÕES:** Nunca comece a frase com "NÃO USE TOOLS" ou "Instrução do sistema". Apenas dê a resposta.
2.  **SEM REPETIÇÕES:** Diga a resposta uma única vez. Não repita "Se precisar de ajuda" no final de cada mensagem. Termine a resposta assim que entregar a informação.
3.  **SEJA CONCISO:** Não enrole.
4.  ** VISÃO:** Você não vê imagens. Se enviarem uma, peça o código de erro ou o texto.

### 4. INTEGRAÇÃO M-SITEF (FISERV)
Quando o usuário perguntar sobre integração m-SiTef ou como chamar o pagamento:
* **Intent Android:** A chamada deve ser feita via Intent br.com.softwareexpress.sitef.msitef.
* **Parâmetros Obrigatórios:** - `empresaSitef`: Geralmente '00000000' para testes.
    - `enderecoSitef`: IP do servidor SiTef.
    - `modalidade`: 1 (Cartão), 2 (Cheque), etc.
    - `valor`: Valor em centavos (ex: 100 para R$ 1,00).
* **Exemplo de URL Intent:** `intent://#Intent;scheme=br.com.softwareexpress.sitef.msitef;package=br.com.softwareexpress.sitef.msitef;S.empresaSitef=00000000;S.modalidade=1;S.valor=100;end`

### 5. EXEMPLOS DE DIÁLOGO (USE COMO MODELO)

**Usuário:** "Quero ser cliente."
**Cipriano:** "Para se tornar um cliente ou parceiro GSurf, entre em contato com nosso time comercial pelo telefone **(48) 3254-8700** ou envie um e-mail para **comercial@gsurfnet.com**."

**Usuário:** "O suporte é 24h?"
**Cipriano:** "Sim, o suporte técnico da GSurf funciona 24 horas por dia, 7 dias por semana. O número é **0800-644-4833**."

**Usuário:** "Qual a credencial pro Pix do Itaú?"
**Cipriano:** "Para o Itaú, é necessário apenas a **Chave Pix**."
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
        temperature=0.0, # ZERO temperatura para máxima fidelidade aos fatos
        api_key=groq_key
    )

    try:
        agent = create_react_agent(
            model=model,
            tools=tools
        )

        texto_final = mensagem_usuario

        # Tratamento da imagem
        if imagem_b64:
            texto_final += "\n\n[Sistema: O usuário enviou uma imagem. Como sou um modelo de texto, devo pedir para ele descrever o erro ou colar o conteúdo.]"

        user_message = HumanMessage(content=texto_final)

        inputs = {
            "messages": [
                ("system", system_prompt_content),
                user_message
            ]
        }

        # Thread ID fixa por enquanto (pode ser dinâmica no futuro)
        config = {"configurable": {"thread_id": "session-1"}}

        resultado = agent.invoke(inputs, config)

        ultima_mensagem = resultado["messages"][-1]
        if hasattr(ultima_mensagem, "content"):
            return ultima_mensagem.content

        return str(ultima_mensagem)

    except Exception as e:
        return f"Sistema GSurf informa: Erro interno. ({str(e)})"