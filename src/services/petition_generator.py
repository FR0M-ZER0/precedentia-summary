import os
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


SYSTEM_PROMPT = """Você é um advogado especialista na redação de petições iniciais brasileiras.

Sua função é redigir uma petição inicial completa, em Markdown, seguindo rigorosamente o
padrão das petições iniciais do direito brasileiro.

ESTRUTURA OBRIGATÓRIA (use exatamente estes títulos e esta ordem):

1. Cabeçalho de endereçamento (ex.: "EXCELENTÍSSIMO SENHOR DOUTOR JUIZ DE DIREITO…")
2. Qualificação do(a) autor(a)
3. Nome da ação em destaque (ex.: "## AÇÃO DE COBRANÇA")
4. "em face de:" seguido da qualificação do(a) réu(ré)
5. ## I – DOS FATOS
6. ## II – DO DIREITO  (cite os precedentes fornecidos como fundamentos jurídicos)
7. ## III – DOS PEDIDOS
   - Incluir tutela de urgência como primeiro pedido SE solicitada
   - Incluir pedido de justiça gratuita SE solicitada
   - Pedidos finais numerados
   - Condenação em custas e honorários
8. Valor da causa
9. Encerramento padrão (Termos em que, Pede deferimento. [Local], [Data]. [Nome do advogado] OAB/[UF])

REGRAS DE REDAÇÃO:
- Use linguagem jurídica formal e técnica.
- Não invente fatos, nomes ou números: mantenha os placeholders como "[●]" onde não houver dados.
- Incorpore os precedentes na seção "DO DIREITO", explicando a ratio decidendi e sua aplicação ao caso.
- Se não houver precedentes, fundamente com legislação e doutrina pertinentes ao tipo de ação.
- Retorne APENAS o texto da petição em Markdown. Nenhuma explicação adicional.
"""

def _format_precedents(precedents: list[dict]) -> str:
    if not precedents:
        return "Nenhum precedente fornecido."
    lines = []
    for i, p in enumerate(precedents, 1):
        lines.append(
            f"{i}. **{p.get('name', 'Sem nome')}**\n"
            f"   Questão: {p.get('question', '')}\n"
            f"   Descrição: {p.get('description', '')}"
        )
    return "\n\n".join(lines)

def _format_files(files: list[str]) -> str:
    """
    Expects a list of strings with the text already extracted from each file.
    Integrate your existing file-extraction logic before calling this service,
    passing the resulting texts here.
    """
    if not files:
        return ""
    parts = [f"[Documento {i + 1}]\n{text.strip()}" for i, text in enumerate(files)]
    return "\n\n---\n\n".join(parts)


def generate_petition(
    author_description: str,
    defendant_description: str,
    action_type: str,
    tribunal: str,
    facts_summary: str,
    files: list[str],          
    requests: list[str],
    cause_value: str,
    urgent_injunction: bool,
    free_justice: bool,
    precedents: list[dict],
) -> str:
    files_text = _format_files(files)
    facts_block = facts_summary
    if files_text:
        facts_block += f"\n\n**Documentos complementares:**\n\n{files_text}"

    requests_block = "\n".join(f"- {r}" for r in requests) if requests else "Não informado."
    precedents_block = _format_precedents(precedents)

    prompt = f"""Redija a petição inicial com os dados abaixo.

---
TIPO DE AÇÃO: {action_type}
TRIBUNAL / JUÍZO: {tribunal}

AUTOR(A):
{author_description}

RÉU / RÉ:
{defendant_description}

FATOS:
{facts_block}

PEDIDOS:
{requests_block}

VALOR DA CAUSA: {cause_value}
TUTELA DE URGÊNCIA: {"Sim" if urgent_injunction else "Não"}
JUSTIÇA GRATUITA: {"Sim" if free_justice else "Não"}

PRECEDENTES A CITAR:
{precedents_block}
---

Redija a petição inicial completa seguindo a estrutura obrigatória definida."""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )

    full_response = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta is not None:
            full_response += delta

    return full_response.strip()


EDIT_SYSTEM_PROMPT = """Você é um advogado especialista na revisão de petições iniciais brasileiras.

Você receberá o texto atual de uma petição inicial (em Markdown) e uma instrução de alteração.
Aplique APENAS a mudança solicitada, preservando todo o restante do texto intacto.
Retorne SOMENTE o texto completo da petição revisada, sem explicações adicionais."""


def edit_petition(content: str, change: str) -> str:
    prompt = f"""PETIÇÃO ATUAL:
{content}

ALTERAÇÃO SOLICITADA:
{change}

Retorne a petição completa com a alteração aplicada."""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": EDIT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )

    full_response = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta is not None:
            full_response += delta

    return full_response.strip()
