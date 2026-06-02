import os
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """Você é um advogado especialista em pesquisa jurídica e argumentação processual.

Sua função é analisar um precedente judicial e uma petição inicial, e redigir um texto jurídico 
explicando de forma clara e fundamentada como o precedente pode ser utilizado como argumento 
na petição.

Diretrizes:
- Identifique os pontos de convergência entre o precedente e os fatos/pedidos da petição.
- Explique a ratio decidendi do precedente e por que ela se aplica ao caso concreto.
- Indique em quais trechos ou pedidos da petição o precedente deve ser citado.
- Use linguagem jurídica formal e precisa.
- Seja objetivo e direto. Não invente fatos, não extrapole o que está nos textos.
- Retorne apenas o texto de análise, sem títulos, sem markdown, sem introduções genéricas.
"""


def analyze_precedent(texto_precedente: str, texto_peticao: str) -> str:
    prompt = f"""PRECEDENTE JUDICIAL:
{texto_precedente}

PETIÇÃO INICIAL:
{texto_peticao}

Com base nos textos acima, redija a análise de aplicabilidade do precedente à petição."""

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