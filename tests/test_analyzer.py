import json
import pytest
from unittest.mock import patch, MagicMock
from src.app import app
from src.services.analyzer import analyze_precedent_applicability


# ── Flask client fixture ──────────────────────────────────────────────────────

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── Helper: simulates ollama stream with a single chunk ──────────────────────

def make_ollama_stream(text: str):
    chunk = MagicMock()
    chunk.__getitem__ = lambda self, key: (
        {"message": {"content": text}}[key]
    )
    return iter([chunk])


PRECEDENT_FAKE_JSON = {
    "aplicavel": True,
    "grau_de_relevancia": "alto",
    "resumo": "O precedente trata da manutenção indevida de negativação após quitação de dívida, situação idêntica à narrada na petição.",
    "pontos_de_convergencia": [
        "Manutenção do nome do consumidor nos cadastros após quitação.",
        "Descumprimento do prazo de 5 dias úteis previsto no art. 43 do CDC.",
    ],
    "pontos_de_divergencia": [
        "No precedente, o valor indenizatório foi fixado em R$ 10.000,00; na petição, o autor pleiteia R$ 15.000,00.",
    ],
    "fundamentos_aproveitaveis": [
        "Dano moral in re ipsa: prescinde de comprovação do efetivo prejuízo.",
        "Prazo de 5 dias úteis para baixa da negativação como marco para o dever de indenizar.",
    ],
    "recomendacao": "Citar o REsp 1.704.520/SP para reforçar a tese do dano moral in re ipsa e embasar o valor pleiteado, destacando que o STJ já manteve indenizações na mesma faixa.",
}

PETICAO_TEXTO = "Petição inicial sobre negativação indevida após quitação..."
PRECEDENTE_TEXTO = "REsp 1.704.520 – SP: dano moral in re ipsa por negativação indevida..."


# ── Route tests (/api/analyze-precedent) ─────────────────────────────────────

class TestRota:
    def test_no_body_returns_400(self, client):
        r = client.post("/api/analyze-precedent")
        assert r.status_code == 400

    def test_missing_both_fields_returns_400(self, client):
        r = client.post("/api/analyze-precedent", json={"outro": "campo"})
        assert r.status_code == 400
        data = r.get_json()
        assert "peticao" in data["error"]
        assert "precedente" in data["error"]

    def test_missing_peticao_returns_400(self, client):
        r = client.post("/api/analyze-precedent", json={"precedente": PRECEDENTE_TEXTO})
        assert r.status_code == 400
        data = r.get_json()
        assert "peticao" in data["error"]

    def test_missing_precedente_returns_400(self, client):
        r = client.post("/api/analyze-precedent", json={"peticao": PETICAO_TEXTO})
        assert r.status_code == 400
        data = r.get_json()
        assert "precedente" in data["error"]

    def test_empty_peticao_returns_400(self, client):
        r = client.post("/api/analyze-precedent", json={"peticao": "   ", "precedente": PRECEDENTE_TEXTO})
        assert r.status_code == 400

    def test_empty_precedente_returns_400(self, client):
        r = client.post("/api/analyze-precedent", json={"peticao": PETICAO_TEXTO, "precedente": "   "})
        assert r.status_code == 400

    @patch("src.services.analyzer.ollama.chat")
    def test_valid_request_returns_200(self, mock_chat, client):
        mock_chat.return_value = make_ollama_stream(json.dumps(PRECEDENT_FAKE_JSON))
        r = client.post("/api/analyze-precedent", json={
            "peticao": PETICAO_TEXTO,
            "precedente": PRECEDENTE_TEXTO,
        })
        assert r.status_code == 200
        data = r.get_json()
        assert data["aplicavel"] is True
        assert data["grau_de_relevancia"] == "alto"

    @patch("src.services.analyzer.ollama.chat")
    def test_invalid_json_from_model_returns_502(self, mock_chat, client):
        mock_chat.return_value = make_ollama_stream("isso não é json")
        r = client.post("/api/analyze-precedent", json={
            "peticao": PETICAO_TEXTO,
            "precedente": PRECEDENTE_TEXTO,
        })
        assert r.status_code == 502
        data = r.get_json()
        assert "error" in data


# ── Service tests (analyze_precedent_applicability) ──────────────────────────

class TestServico:
    @patch("src.services.analyzer.ollama.chat")
    def test_returns_dict(self, mock_chat):
        mock_chat.return_value = make_ollama_stream(json.dumps(PRECEDENT_FAKE_JSON))
        result = analyze_precedent_applicability(PETICAO_TEXTO, PRECEDENTE_TEXTO)
        assert isinstance(result, dict)

    @patch("src.services.analyzer.ollama.chat")
    def test_required_fields_present(self, mock_chat):
        mock_chat.return_value = make_ollama_stream(json.dumps(PRECEDENT_FAKE_JSON))
        result = analyze_precedent_applicability(PETICAO_TEXTO, PRECEDENTE_TEXTO)
        for field in [
            "aplicavel",
            "grau_de_relevancia",
            "resumo",
            "pontos_de_convergencia",
            "pontos_de_divergencia",
            "fundamentos_aproveitaveis",
            "recomendacao",
        ]:
            assert field in result, f"Field '{field}' missing from response"

    @patch("src.services.analyzer.ollama.chat")
    def test_list_fields_are_lists(self, mock_chat):
        mock_chat.return_value = make_ollama_stream(json.dumps(PRECEDENT_FAKE_JSON))
        result = analyze_precedent_applicability(PETICAO_TEXTO, PRECEDENTE_TEXTO)
        assert isinstance(result["pontos_de_convergencia"], list)
        assert isinstance(result["pontos_de_divergencia"], list)
        assert isinstance(result["fundamentos_aproveitaveis"], list)

    @patch("src.services.analyzer.ollama.chat")
    def test_aplicavel_is_bool(self, mock_chat):
        mock_chat.return_value = make_ollama_stream(json.dumps(PRECEDENT_FAKE_JSON))
        result = analyze_precedent_applicability(PETICAO_TEXTO, PRECEDENTE_TEXTO)
        assert isinstance(result["aplicavel"], bool)

    @patch("src.services.analyzer.ollama.chat")
    def test_strips_markdown_fences(self, mock_chat):
        wrapped = f"```json\n{json.dumps(PRECEDENT_FAKE_JSON)}\n```"
        mock_chat.return_value = make_ollama_stream(wrapped)
        result = analyze_precedent_applicability(PETICAO_TEXTO, PRECEDENTE_TEXTO)
        assert result["grau_de_relevancia"] == "alto"

    @patch("src.services.analyzer.ollama.chat")
    def test_strips_think_block(self, mock_chat):
        with_think = f"<think>raciocínio interno</think>\n{json.dumps(PRECEDENT_FAKE_JSON)}"
        mock_chat.return_value = make_ollama_stream(with_think)
        result = analyze_precedent_applicability(PETICAO_TEXTO, PRECEDENTE_TEXTO)
        assert result["aplicavel"] is True

    @patch("src.services.analyzer.ollama.chat")
    def test_invalid_json_raises_exception(self, mock_chat):
        mock_chat.return_value = make_ollama_stream("isso não é json")
        with pytest.raises(json.JSONDecodeError):
            analyze_precedent_applicability(PETICAO_TEXTO, PRECEDENTE_TEXTO)

    @patch("src.services.analyzer.ollama.chat")
    def test_null_fields_accepted(self, mock_chat):
        """Optional fields like resumo and recomendacao can be null."""
        partial = {**PRECEDENT_FAKE_JSON, "resumo": None, "recomendacao": None}
        mock_chat.return_value = make_ollama_stream(json.dumps(partial))
        result = analyze_precedent_applicability(PETICAO_TEXTO, PRECEDENTE_TEXTO)
        assert result["resumo"] is None
        assert result["recomendacao"] is None

    @patch("src.services.analyzer.ollama.chat")
    def test_not_applicable_precedent(self, mock_chat):
        """Validates the schema when the precedent is deemed not applicable."""
        not_applicable = {**PRECEDENT_FAKE_JSON, "aplicavel": False, "grau_de_relevancia": "baixo"}
        mock_chat.return_value = make_ollama_stream(json.dumps(not_applicable))
        result = analyze_precedent_applicability(PETICAO_TEXTO, PRECEDENTE_TEXTO)
        assert result["aplicavel"] is False
        assert result["grau_de_relevancia"] == "baixo"