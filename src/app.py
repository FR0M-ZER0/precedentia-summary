from dotenv import load_dotenv
load_dotenv()

import os
import json
import logging
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler

from src.services.deconstructor import deconstruct_petition
from src.services.summarizer import run_summarizer
from src.services.analyzer import analyze_precedent_applicability

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


@app.route("/api/analyze-precedent", methods=["POST"])
def analyze_precedent():
    app.logger.info("Precedent analysis request received")
    body = request.get_json(silent=True)

    if not body:
        return jsonify({"error": "Request body is required."}), 400

    missing = [field for field in ("peticao", "precedente") if field not in body]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}."}), 400

    petition_text = body["peticao"].strip()
    precedent_text = body["precedente"].strip()

    if not petition_text:
        return jsonify({"error": "Field 'peticao' must not be empty."}), 400
    if not precedent_text:
        return jsonify({"error": "Field 'precedente' must not be empty."}), 400

    try:
        result = analyze_precedent_applicability(petition_text, precedent_text)
        return jsonify(result), 200
    except json.JSONDecodeError as e:
        return jsonify({"error": "Model did not return valid JSON.", "detail": str(e)}), 502
    except Exception as e:
        return jsonify({"error": "Internal server error.", "detail": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)