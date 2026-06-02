import json
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("APPLICABILITY_MODEL", "gpt-4o")

SYSTEM_PROMPT = """Você é um analista jurídico especializado em direito brasileiro.
Avalie se o fundamento jurídico determinante (ratio decidendi) de cada precedente 
se aplica aos fatos da petição. Retorne SOMENTE JSON puro, sem markdown."""

def check_applicability(facts: str, petition_type: str, precedents: list[dict]) -> list[dict]:
    precedents_text = "\n\n".join([
        f"[{i}] Nome: {p.get('name', '')}\n"
        f"Espécie: {p.get('species', '')}\n"
        f"Ementa: {p.get('description', '')[:600]}\n"
        f"Resumo: {p.get('summary', '')[:400]}"
        for i, p in enumerate(precedents)
    ])

    prompt = f"""Analise se cada precedente é aplicável à petição.

TIPO DA AÇÃO: {petition_type}
FATOS: {facts[:1000]}

PRECEDENTES:
{precedents_text}

Para cada precedente, atribua um score de aplicabilidade de 0.0 a 1.0:
- 0.0 a 0.39: fundamento determinante não se encaixa nos fatos
- 0.40 a 0.79: fundamento pode se aplicar, mas há ressalvas ou lacunas fáticas  
- 0.80 a 1.0: fundamento determinante se encaixa diretamente nos fatos

Retorne SOMENTE este JSON:
{{
  "avaliacoes": [
    {{
      "indice": 0,
      "applicability_score": 0.0,
      "justification": "Explicação objetiva em português do motivo da classificação e, caso seja aplicável, o porquê do enquadramento nos fatos."
    }}
  ]
}}"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    avaliacoes = {a["indice"]: a for a in json.loads(raw)["avaliacoes"]}

    for i, precedent in enumerate(precedents):
        av = avaliacoes.get(i, {})
        score = av.get("applicability_score", 0.5)

        if score >= 0.8:
            label = "applicable"
        elif score >= 0.4:
            label = "possible_applicability"
        else:
            label = "low_applicability"

        precedent["applicability"] = label
        precedent["applicability_score"] = score
        precedent["applicability_justification"] = av.get("justification")

    return precedents