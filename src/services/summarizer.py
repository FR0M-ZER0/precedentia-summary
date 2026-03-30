import ollama
import os
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, IsNullCondition, PayloadField

MODEL = os.getenv("OLLAMA_MODEL")
TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", 0))
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

logger = logging.getLogger(__name__)


def get_qdrant_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def fetch_unsummarized(client: QdrantClient, limit: int = 10):
    results, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=Filter(
            must=[
                IsNullCondition(
                    is_null=PayloadField(key="summary")
                )
            ]
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return results


def generate_summary(payload: dict) -> str:
    prompt = f"""Você é um assistente jurídico. Resuma o seguinte precedente em 2 a 3 frases claras e objetivas, destacando o tema central e sua relevância prática.
Retorne APENAS o resumo em texto puro, sem explicações adicionais.

Nome: {payload.get('name')}
Tribunal: {payload.get('tribunal')}
Situação: {payload.get('situation')}
Descrição: {payload.get('description')}"""

    stream = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": TEMPERATURE},
        stream=True,
    )

    full_response = ""
    for chunk in stream:
        full_response += chunk["message"]["content"]

    # Strip <think>...</think> se o modelo for de raciocínio
    raw = full_response.strip()
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[-1].strip()

    return raw


def save_summary(client: QdrantClient, point_id: int, summary: str):
    client.set_payload(
        collection_name=QDRANT_COLLECTION,
        payload={"summary": summary},
        points=[point_id],
    )


def run_summarizer():
    logger.info("Job de resumo iniciado")
    client = get_qdrant_client()

    try:
        points = fetch_unsummarized(client)

        if not points:
            logger.info("Nenhum precedente pendente de resumo")
            return

        for point in points:
            try:
                summary = generate_summary(point.payload)
                save_summary(client, point.id, summary)
                logger.info(f"Resumo salvo para precedent:{point.id}")
            except Exception as e:
                logger.error(f"Erro ao resumir precedent:{point.id}: {e}")

    except Exception as e:
        logger.error(f"Erro no job de resumo: {e}")
    finally:
        client.close()