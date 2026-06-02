import json
import os
import ollama

MODEL = os.getenv("OLLAMA_MODEL")
TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", 0))


def analyze_precedent_applicability(petition_text: str, precedent_text: str) -> dict:
    prompt = f"""Você é um especialista em direito brasileiro. Analise a petição inicial e o precedente jurídico fornecidos abaixo e avalie a aplicabilidade do precedente ao caso concreto.
Retorne APENAS o JSON, sem explicações, sem markdown, sem blocos de código.
O JSON deve seguir exatamente esta estrutura:
{{
  "aplicavel": true ou false,
  "grau_de_relevancia": "alto | medio | baixo",
  "resumo": "string — síntese objetiva da relação entre o precedente e a petição",
  "pontos_de_convergencia": ["string"],
  "pontos_de_divergencia": ["string"],
  "fundamentos_aproveitaveis": ["string — trechos ou teses do precedente diretamente utilizáveis na defesa"],
  "recomendacao": "string — orientação prática sobre como usar ou afastar o precedente"
}}
Se alguma informação não puder ser determinada, use null para campos simples ou [] para listas.

PETIÇÃO INICIAL:
{petition_text}

PRECEDENTE:
{precedent_text}"""

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
