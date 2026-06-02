import os
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


SENTENCE_SYSTEM_PROMPT = """Você é um juiz de direito especialista na redação de sentenças judiciais brasileiras.

Sua função é redigir uma minuta de sentença completa, em Markdown, seguindo rigorosamente o
padrão das sentenças do direito brasileiro.

ESTRUTURA OBRIGATÓRIA (use exatamente estes títulos e esta ordem):

1. Cabeçalho (Vara, Comarca, número do processo, partes)
2. ## I – RELATÓRIO
   - Síntese da petição inicial: partes, fatos, pedidos
   - Síntese da contestação (se houver) ou ausência dela
3. ## II – FUNDAMENTAÇÃO
   - Análise dos fatos à luz do direito aplicável
   - Cite os precedentes fornecidos como fundamentos, explicando a ratio decidendi
   - Se houver contestação, analise as teses defensivas e rebata ou acolha cada uma
   - Se um precedente não for plenamente aderente, explicite a distinção (distinguishing)
   - Se não houver precedentes fortes, sinalize expressamente: "Ausência de precedentes consolidados sobre o tema."
4. ## III – DISPOSITIVO
   - Julgamento (procedente / improcedente / parcialmente procedente)
   - Fundamento legal (art. 487, I, CPC)
   - Determinações sobre custas e honorários advocatícios
   - Intimação das partes

REGRAS DE REDAÇÃO:
- Use linguagem jurídica formal e técnica.
- Não invente fatos, nomes ou números: use apenas os dados fornecidos.
- Mantenha placeholders "[●]" onde não houver dados suficientes.
- Se a contestação não for fornecida, indique na fundamentação que a análise é unilateral.
- Retorne APENAS o texto da minuta em Markdown. Nenhuma explicação adicional.
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


def generate_sentence(
    author: str,
    defendant: str,
    action_type: str,
    tribunal: str,
    facts_summary: str,
    requests: list[str],
    precedents: list[dict],
    contestacao: str | None = None,
) -> str:
    requests_block = "\n".join(f"- {r}" for r in requests) if requests else "Não informado."
    precedents_block = _format_precedents(precedents)
    contestacao_block = (
        contestacao.strip()
        if contestacao
        else "Não fornecida — análise baseada apenas na petição inicial."
    )

    prompt = f"""Redija a minuta de sentença com os dados abaixo.

---
TIPO DE AÇÃO: {action_type}
TRIBUNAL / JUÍZO: {tribunal}

AUTOR(A): {author}
RÉU / RÉ: {defendant}

FATOS NARRADOS NA INICIAL:
{facts_summary}

PEDIDOS:
{requests_block}

CONTESTAÇÃO:
{contestacao_block}

PRECEDENTES:
{precedents_block}
---

Redija a minuta completa seguindo a estrutura obrigatória definida."""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SENTENCE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content.strip()


EDIT_SENTENCE_SYSTEM_PROMPT = """Você é um juiz de direito especialista na revisão de minutas de sentenças judiciais brasileiras.

Você receberá o texto atual de uma minuta de sentença (em Markdown) e uma instrução de alteração.
Aplique APENAS a mudança solicitada, preservando todo o restante do texto intacto.
Retorne SOMENTE o texto completo da minuta revisada, sem explicações adicionais."""


def edit_sentence(content: str, change: str) -> str:
    prompt = f"""MINUTA ATUAL:
{content}

ALTERAÇÃO SOLICITADA:
{change}

Retorne a minuta completa com a alteração aplicada."""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": EDIT_SENTENCE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content.strip()
