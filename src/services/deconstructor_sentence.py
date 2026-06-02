import json
import os
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EXAMPLE = """Você é um extrator de informações jurídicas.

Sua única função é retornar um JSON com exatamente estas chaves:
tipo, tribunal, autor, reu, fatos, pedidos, contestacao.

Nunca invente outras chaves.
Nunca adicione campos extras.
Retorne SOMENTE o JSON puro, sem markdown e sem explicações.

REGRAS:
- "autor": nome da parte autora ou null se não identificado
- "reu": nome da parte ré ou null se não identificado
- "tribunal": apenas o acrônimo (STF, STJ, TJSP etc.) ou null
- "fatos": todos os fatos em texto corrido
- "pedidos": lista de pedidos
- "contestacao":
    - se existir contestação no texto, extraia
    - se NÃO existir, gere uma contestação jurídica simples e coerente baseada nos fatos e pedidos

EXEMPLO:

Petição:
"...João da Silva ajuizou ação de alimentos contra Maria Souza...
requer o pagamento de R$ 2.000,00 mensais...
fatos: ré abandonou o lar...
endereçada ao Supremo Tribunal Federal..."

Saída:
{
  "tipo": "Ação de Alimentos",
  "tribunal": "STF",
  "autor": "João da Silva",
  "reu": "Maria Souza",
  "fatos": "Ré abandonou o lar familiar...",
  "pedidos": [
    "Pagamento de alimentos mensais no valor de R$ 2.000,00"
  ],
  "contestacao": "A parte ré alega impossibilidade financeira parcial para arcar com o valor requerido, requerendo a fixação proporcional dos alimentos conforme sua capacidade econômica."
}
"""

SCHEMA = """Retorne APENAS este JSON:

{
  "tipo": "string",
  "tribunal": "string ou null",
  "autor": "string ou null",
  "reu": "string ou null",
  "fatos": "string",
  "pedidos": ["string"],
  "contestacao": "string"
}

Não adicione nenhum campo extra.
"""

VALID_KEYS = {
    "tipo",
    "tribunal",
    "autor",
    "reu",
    "fatos",
    "pedidos",
    "contestacao",
}


def _parse_response(raw: str) -> dict:
    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]

        if raw.startswith("json"):
            raw = raw[4:]

        raw = raw.strip()

    data = json.loads(raw)

    return {k: v for k, v in data.items() if k in VALID_KEYS}


def deconstruct_petition_from_lawsuit(texto_peticao: str) -> dict:
    prompt = f"""{SCHEMA}

PETIÇÃO:
{texto_peticao}
"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": EXAMPLE},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )

    full_response = ""

    for chunk in response:
        delta = chunk.choices[0].delta.content

        if delta is not None:
            full_response += delta

    return _parse_response(full_response)
