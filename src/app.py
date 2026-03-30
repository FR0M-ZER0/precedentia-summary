from dotenv import load_dotenv
load_dotenv()

import os
import json
import logging
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler

from src.services.deconstructor import deconstruct_petition
from src.services.summarizer import run_summarizer

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

SUMMARIZER_INTERVAL_MINUTES = int(os.getenv("SUMMARIZER_INTERVAL_MINUTES", 5))

scheduler = BackgroundScheduler()
scheduler.add_job(
    run_summarizer,
    "interval",
    minutes=SUMMARIZER_INTERVAL_MINUTES,
)
scheduler.start()


@app.route("/api/deconstruct", methods=["POST"])
def extrair():
    app.logger.info("Request received")
    body = request.get_json(silent=True)

    if not body or "peticao" not in body:
        return jsonify({"error": "Campo 'peticao' é obrigatório."}), 400

    texto = body["peticao"].strip()
    if not texto:
        return jsonify({"error": "O campo 'peticao' não pode estar vazio."}), 400

    try:
        resultado = deconstruct_petition(texto)
        return jsonify(resultado), 200
    except json.JSONDecodeError as e:
        return jsonify({"error": "O modelo não retornou um JSON válido.", "detail": str(e)}), 502
    except Exception as e:
        return jsonify({"error": "Erro interno.", "detail": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)