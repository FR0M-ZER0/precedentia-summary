from dotenv import load_dotenv
load_dotenv()

import os
import json
import logging
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler

from src.services.deconstructor import deconstruct_petition
from src.services.summarizer import run_summarizer
from src.services.precedent_analyzer import analyze_precedent
from src.services.applicability_checker import check_applicability
from src.services.petition_generator import generate_petition, edit_petition
from src.services.deconstructor_sentence import deconstruct_petition_from_lawsuit
from src.services.sentence_generator import generate_sentence, edit_sentence

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
def analisar_precedente():
    app.logger.info("Request received")
    body = request.get_json(silent=True)
 
    if not body:
        return jsonify({"error": "Corpo da requisição inválido."}), 400
 
    precedente = (body.get("precedente") or "").strip()
    peticao = (body.get("peticao") or "").strip()
 
    if not precedente:
        return jsonify({"error": "Campo 'precedente' é obrigatório."}), 400
    if not peticao:
        return jsonify({"error": "Campo 'peticao' é obrigatório."}), 400
 
    try:
        analise = analyze_precedent(precedente, peticao)
        return jsonify({"analise": analise}), 200
    except Exception as e:
        return jsonify({"error": "Erro interno.", "detail": str(e)}), 500
    

@app.route("/api/check-applicability", methods=["POST"])
def verificar_aplicabilidade():
    app.logger.info("Request received")
    body = request.get_json(silent=True)

    if not body:
        return jsonify({"error": "Corpo da requisição inválido."}), 400

    facts = (body.get("facts") or "").strip()
    petition_type = (body.get("petition_type") or "").strip()
    precedents = body.get("precedents", [])

    if not facts:
        return jsonify({"error": "Campo 'facts' é obrigatório."}), 400
    if not precedents:
        return jsonify({"error": "Campo 'precedents' é obrigatório."}), 400

    try:
        resultado = check_applicability(facts, petition_type, precedents)
        return jsonify({"precedents": resultado}), 200
    except json.JSONDecodeError as e:
        return jsonify({"error": "O modelo não retornou um JSON válido.", "detail": str(e)}), 502
    except Exception as e:
        return jsonify({"error": "Erro interno.", "detail": str(e)}), 500

@app.route("/api/petition/generate", methods=["POST"])
def gerar_peticao():
    app.logger.info("Request received – generate petition")
    body = request.get_json(silent=True)
 
    if not body:
        return jsonify({"error": "Corpo da requisição inválido."}), 400
 
    # ── Campos obrigatórios ──────────────────────────────────────────────────
    required = [
        "author_description",
        "defendant_description",
        "action_type",
        "tribunal",
        "facts_summary",
        "cause_value",
    ]
    missing = [f for f in required if not (body.get(f) or "").strip()]
    if missing:
        return jsonify({"error": f"Campos obrigatórios ausentes: {', '.join(missing)}"}), 400
 
    # ── Campos opcionais com defaults seguros ────────────────────────────────
    files: list[str] = body.get("files", [])          # lista de textos já extraídos
    requests_list: list[str] = body.get("requests", [])
    precedents: list[dict] = body.get("precedents", [])  # [{name, question, description}]
    urgent_injunction: bool = bool(body.get("urgent_injunction", False))
    free_justice: bool = bool(body.get("free_justice", False))
 
    # ── Validações de tipo ────────────────────────────────────────────────────
    if not isinstance(files, list):
        return jsonify({"error": "O campo 'files' deve ser uma lista de strings."}), 400
    if not isinstance(requests_list, list):
        return jsonify({"error": "O campo 'requests' deve ser uma lista de strings."}), 400
    if not isinstance(precedents, list):
        return jsonify({"error": "O campo 'precedents' deve ser uma lista de objetos."}), 400
 
    try:
        petition_text = generate_petition(
            author_description=body["author_description"].strip(),
            defendant_description=body["defendant_description"].strip(),
            action_type=body["action_type"].strip(),
            tribunal=body["tribunal"].strip(),
            facts_summary=body["facts_summary"].strip(),
            files=files,
            requests=requests_list,
            cause_value=body["cause_value"].strip(),
            urgent_injunction=urgent_injunction,
            free_justice=free_justice,
            precedents=precedents,
        )
        return jsonify({"content": petition_text}), 200
 
    except Exception as e:
        app.logger.error(f"Erro ao gerar petição: {e}")
        return jsonify({"error": "Erro interno.", "detail": str(e)}), 500
 
 
@app.route("/api/petition/edit", methods=["POST"])
def editar_peticao():
    app.logger.info("Request received – edit petition")
    body = request.get_json(silent=True)
 
    if not body:
        return jsonify({"error": "Corpo da requisição inválido."}), 400
 
    content = (body.get("content") or "").strip()
    change = (body.get("change") or "").strip()
 
    if not content:
        return jsonify({"error": "Campo 'content' é obrigatório."}), 400
    if not change:
        return jsonify({"error": "Campo 'change' é obrigatório."}), 400
 
    try:
        edited = edit_petition(content=content, change=change)
        return jsonify({"content": edited}), 200
 
    except Exception as e:
        app.logger.error(f"Erro ao editar petição: {e}")
        return jsonify({"error": "Erro interno.", "detail": str(e)}), 500


@app.route("/api/deconstruct-lawsuit", methods=["POST"])
def extrair_processo():
    app.logger.info("Request received – lawsuit deconstruction")

    body = request.get_json(silent=True)

    if not body or "peticao" not in body:
        return jsonify({"error": "Campo 'peticao' é obrigatório."}), 400

    texto = body["peticao"].strip()

    if not texto:
        return jsonify({"error": "O campo 'peticao' não pode estar vazio."}), 400

    try:
        resultado = deconstruct_petition_from_lawsuit(texto)

        return jsonify({
            "tipo": resultado.get("tipo"),
            "tribunal": resultado.get("tribunal"),
            "autor": resultado.get("autor"),
            "reu": resultado.get("reu"),
            "fatos": resultado.get("fatos"),
            "pedidos": resultado.get("pedidos", []),
            "contestacao": resultado.get("contestacao"),
        }), 200

    except json.JSONDecodeError as e:
        return jsonify({
            "error": "O modelo não retornou um JSON válido.",
            "detail": str(e)
        }), 502

    except Exception as e:
        app.logger.error(f"Erro ao desconstruir petição: {e}")

        return jsonify({
            "error": "Erro interno.",
            "detail": str(e)
        }), 500

@app.route("/api/sentence/generate", methods=["POST"])
def gerar_sentenca():
    app.logger.info("Request received – generate sentence")
    body = request.get_json(silent=True)

    if not body:
        return jsonify({"error": "Corpo da requisição inválido."}), 400

    required = ["author", "defendant", "action_type", "tribunal", "facts_summary"]
    missing = [f for f in required if not (body.get(f) or "").strip()]
    if missing:
        return jsonify({"error": f"Campos obrigatórios ausentes: {', '.join(missing)}"}), 400

    requests_list: list[str] = body.get("requests", [])
    precedents: list[dict] = body.get("precedents", [])
    contestacao: str | None = (body.get("contestacao") or "").strip() or None

    if not isinstance(requests_list, list):
        return jsonify({"error": "O campo 'requests' deve ser uma lista de strings."}), 400
    if not isinstance(precedents, list):
        return jsonify({"error": "O campo 'precedents' deve ser uma lista de objetos."}), 400

    try:
        sentence_text = generate_sentence(
            author=body["author"].strip(),
            defendant=body["defendant"].strip(),
            action_type=body["action_type"].strip(),
            tribunal=body["tribunal"].strip(),
            facts_summary=body["facts_summary"].strip(),
            requests=requests_list,
            precedents=precedents,
            contestacao=contestacao,
        )
        return jsonify({"content": sentence_text}), 200

    except Exception as e:
        app.logger.error(f"Erro ao gerar sentença: {e}")
        return jsonify({"error": "Erro interno.", "detail": str(e)}), 500


@app.route("/api/sentence/edit", methods=["POST"])
def editar_sentenca():
    app.logger.info("Request received – edit sentence")
    body = request.get_json(silent=True)

    if not body:
        return jsonify({"error": "Corpo da requisição inválido."}), 400

    content = (body.get("content") or "").strip()
    change = (body.get("change") or "").strip()

    if not content:
        return jsonify({"error": "Campo 'content' é obrigatório."}), 400
    if not change:
        return jsonify({"error": "Campo 'change' é obrigatório."}), 400

    try:
        edited = edit_sentence(content=content, change=change)
        return jsonify({"content": edited}), 200

    except Exception as e:
        app.logger.error(f"Erro ao editar sentença: {e}")
        return jsonify({"error": "Erro interno.", "detail": str(e)}), 500
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
