import json
import os
import ollama

MODEL = os.getenv("OLLAMA_MODEL")
TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", 0))

def desconstruir_peticao(texto_peticao: str) -> dict:
    prompt = f"""Você é um especialista em direito brasileiro. Analise a petição inicial abaixo e extraia as informações em formato JSON.
Retorne APENAS o JSON, sem explicações, sem markdown, sem blocos de código.
O JSON deve seguir exatamente esta estrutura:
{{
  "tipo": "string",
  "tribunal": {{ "nome": "string", "comarca": "string", "uf": "string" }},
  "partes": {{
    "autor": {{ "nome": "string", "cpf_cnpj": "string ou null" }},
    "reu":   {{ "nome": "string", "cpf_cnpj": "string ou null" }}
  }},
  "fatos": ["string"],
  "fundamentos_juridicos": ["string"],
  "pedidos": [{{ "descricao": "string", "valor": "number ou null" }}],
  "valor_causa": "number ou null",
  "data_ajuizamento": "string ou null — formato YYYY-MM-DD"
}}
Se alguma informação não estiver presente, use null para campos simples ou [] para listas.
PETIÇÃO INICIAL:
{texto_peticao}"""

    stream = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": TEMPERATURE},
        stream=True,
    )

    full_response = ""
    for chunk in stream:
        full_response += chunk["message"]["content"]

    # Strip <think>...</think> block if present (reasoning models)
    raw = full_response.strip()
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[-1].strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)