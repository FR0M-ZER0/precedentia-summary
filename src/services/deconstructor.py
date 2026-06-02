# import json
# import os
# import ollama

# MODEL = os.getenv("OLLAMA_MODEL")
# TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", 0))

# EXAMPLE = """Você é um extrator de informações jurídicas. 
# Sua única função é retornar um JSON com exatamente estas chaves: tipo, tribunal, fatos, pedidos.
# Nunca invente outras chaves. Nunca adicione campos extras.
# Retorne SOMENTE o JSON puro, sem markdown, sem explicações.

# EXEMPLO:
# Petição: "...ação de alimentos... requer o pagamento de R$ 2.000,00 mensais... fatos: réu abandonou lar... endereçada ao Supremo Tribunal Federal"
# Saída:
# {
#   "tipo": "Ação de Alimentos",
#   "tribunal": STF,
#   "fatos": "Réu abandonou o lar familiar...",
#   "pedidos": [{"descricao": "Pagamento de alimentos mensais", "valor": 2000}]
# }

# Coloque todos os fatos em uma única frase, sem omitir, resumir ou inventar nada.
# """

# SCHEMA = """Retorne APENAS este JSON, sem nenhum campo adicional:
# {
#   "tipo": "string — tipo da ação judicial",
#   "tribunal": "string — apenas o acrônimo (ex: STJ, TJSP) ou null se não mencionado",
#   "fatos": ["string - todos os fatos em forma de texto corrido"],
#   "pedidos": ["string"]
# }"""

# VALID_KEYS = {"tipo", "tribunal", "fatos", "pedidos"}


# def _parse_response(raw: str) -> dict:
#     raw = raw.strip()

#     if "</think>" in raw:
#         raw = raw.split("</think>", 1)[-1].strip()

#     if raw.startswith("```"):
#         raw = raw.split("```", 2)[1]
#         if raw.startswith("json"):
#             raw = raw[4:]
#         raw = raw.strip()

#     data = json.loads(raw)

#     return {k: v for k, v in data.items() if k in VALID_KEYS}


# def deconstruct_petition(texto_peticao: str) -> dict:
#     prompt = f"""{EXAMPLE}

# {SCHEMA}

# PETIÇÃO INICIAL:
# {texto_peticao}"""

#     stream = ollama.chat(
#         model=MODEL,
#         messages=[{"role": "user", "content": prompt}],
#         options={"temperature": TEMPERATURE},
#         stream=True,
#     )

#     full_response = ""
#     for chunk in stream:
#         full_response += chunk["message"]["content"]

#     return _parse_response(full_response)


import json
import os
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EXAMPLE = """Você é um extrator de informações jurídicas.
Sua única função é retornar um JSON com exatamente estas chaves: tipo, tribunal, fatos, pedidos.
Nunca invente outras chaves. Nunca adicione campos extras.
Retorne SOMENTE o JSON puro, sem markdown, sem explicações.

EXEMPLO:
Petição: "...ação de alimentos... requer o pagamento de R$ 2.000,00 mensais... fatos: réu abandonou lar... endereçada ao Supremo Tribunal Federal"
Saída:
{
  "tipo": "Ação de Alimentos",
  "tribunal": "STF",
  "fatos": "Réu abandonou o lar familiar...",
  "pedidos": ["Pagamento de alimentos mensais no valor de R$ 2.000,00", "Outro pedido se houver..."]
}

Coloque todos os fatos em uma única frase, sem omitir, resumir ou inventar nada.
"""

SCHEMA = """Retorne APENAS este JSON, sem nenhum campo adicional:
{
  "tipo": "string — tipo da ação judicial",
  "tribunal": "string — apenas o acrônimo (ex: STJ, TJSP) ou null se não mencionado",
  "fatos": "string — todos os fatos em texto corrido",
  "pedidos": ["string"]
}"""

VALID_KEYS = {"tipo", "tribunal", "fatos", "pedidos"}


def _parse_response(raw: str) -> dict:
    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)

    return {k: v for k, v in data.items() if k in VALID_KEYS}


def deconstruct_petition(texto_peticao: str) -> dict:
    prompt = f"""{SCHEMA}

PETIÇÃO INICIAL:
{texto_peticao}"""

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
