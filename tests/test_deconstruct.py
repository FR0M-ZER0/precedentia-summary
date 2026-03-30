import json
import pytest
from unittest.mock import patch, MagicMock
from src.app import app
from src.services.deconstructor import deconstruct_petition


# ── Fixture do cliente Flask ──────────────────────────────────────────────────

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── Helper: monta o chunk fake que o ollama retornaria ───────────────────────

def make_ollama_stream(text: str):
    """Simula o stream do ollama com um único chunk."""
    chunk = MagicMock()
    chunk.__getitem__ = lambda self, key: (
        {"message": {"content": text}}[key]
    )
    return iter([chunk])


PETICAO_FAKE_JSON = {
    "tipo": "Ação de Cobrança",
    "tribunal": {"nome": "TJSP", "comarca": "São Paulo", "uf": "SP"},
    "partes": {
        "autor": {"nome": "João Silva", "cpf_cnpj": "123.456.789-00"},
        "reu":   {"nome": "Empresa X Ltda", "cpf_cnpj": "00.000.000/0001-00"},
    },
    "fatos": ["O réu não pagou a dívida."],
    "fundamentos_juridicos": ["Art. 586 do CPC"],
    "pedidos": [{"descricao": "Pagamento da dívida", "valor": 5000.00}],
    "valor_causa": 5000.00,
    "data_ajuizamento": "2024-01-15",
}


# ── Testes da rota (/api/deconstruct) ────────────────────────────────────────

class TestRota:
    def test_sem_body_retorna_400(self, client):
        r = client.post("/api/deconstruct")
        assert r.status_code == 400

    def test_campo_peticao_ausente_retorna_400(self, client):
        r = client.post("/api/deconstruct", json={"outro": "campo"})
        assert r.status_code == 400

    def test_peticao_vazia_retorna_400(self, client):
        r = client.post("/api/deconstruct", json={"peticao": "   "})
        assert r.status_code == 400

    @patch("src.services.deconstructor.ollama.chat")
    def test_peticao_valida_retorna_200(self, mock_chat, client):
        mock_chat.return_value = make_ollama_stream(json.dumps(PETICAO_FAKE_JSON))
        r = client.post("/api/deconstruct", json={"peticao": "Petição teste..."})
        assert r.status_code == 200
        data = r.get_json()
        assert data["tipo"] == "Ação de Cobrança"
        assert data["valor_causa"] == 5000.00

    @patch("src.services.deconstructor.ollama.chat")
    def test_modelo_retorna_json_invalido_retorna_502(self, mock_chat, client):
        mock_chat.return_value = make_ollama_stream("isso não é json")
        r = client.post("/api/deconstruct", json={"peticao": "Petição teste..."})
        assert r.status_code == 502
        data = r.get_json()
        assert "error" in data


# ── Testes do serviço (deconstruct_petition) ─────────────────────────────────

class TestServico:
    @patch("src.services.deconstructor.ollama.chat")
    def test_retorno_json_limpo(self, mock_chat):
        mock_chat.return_value = make_ollama_stream(json.dumps(PETICAO_FAKE_JSON))
        result = deconstruct_petition("Petição qualquer")
        assert isinstance(result, dict)
        assert "pedidos" in result
        assert isinstance(result["pedidos"], list)

    @patch("src.services.deconstructor.ollama.chat")
    def test_campos_obrigatorios_presentes(self, mock_chat):
        mock_chat.return_value = make_ollama_stream(json.dumps(PETICAO_FAKE_JSON))
        result = deconstruct_petition("Petição qualquer")
        for campo in ["tipo", "tribunal", "partes", "fatos", "fundamentos_juridicos", "pedidos"]:
            assert campo in result, f"Campo '{campo}' ausente no retorno"

    @patch("src.services.deconstructor.ollama.chat")
    def test_strips_markdown_fences(self, mock_chat):
        """Modelo retornou com ```json ... ``` — deve ser limpo."""
        wrapped = f"```json\n{json.dumps(PETICAO_FAKE_JSON)}\n```"
        mock_chat.return_value = make_ollama_stream(wrapped)
        result = deconstruct_petition("Petição qualquer")
        assert result["tipo"] == "Ação de Cobrança"

    @patch("src.services.deconstructor.ollama.chat")
    def test_strips_think_block(self, mock_chat):
        """Modelo retornou com bloco <think>...</think> — deve ser ignorado."""
        with_think = f"<think>raciocínio interno</think>\n{json.dumps(PETICAO_FAKE_JSON)}"
        mock_chat.return_value = make_ollama_stream(with_think)
        result = deconstruct_petition("Petição qualquer")
        assert result["tipo"] == "Ação de Cobrança"

    @patch("src.services.deconstructor.ollama.chat")
    def test_json_invalido_lanca_excecao(self, mock_chat):
        """Se o modelo retornar lixo, deve explodir com JSONDecodeError."""
        mock_chat.return_value = make_ollama_stream("isso não é json")
        with pytest.raises(json.JSONDecodeError):
            deconstruct_petition("Petição qualquer")

    @patch("src.services.deconstructor.ollama.chat")
    def test_campos_nulos_aceitos(self, mock_chat):
        """Campos opcionais como valor_causa e data_ajuizamento podem ser null."""
        parcial = {**PETICAO_FAKE_JSON, "valor_causa": None, "data_ajuizamento": None}
        mock_chat.return_value = make_ollama_stream(json.dumps(parcial))
        result = deconstruct_petition("Petição qualquer")
        assert result["valor_causa"] is None
        assert result["data_ajuizamento"] is None

    @patch("src.services.deconstructor.ollama.chat")
    def test_fatos_e_pedidos_sao_listas(self, mock_chat):
        mock_chat.return_value = make_ollama_stream(json.dumps(PETICAO_FAKE_JSON))
        result = deconstruct_petition("Petição qualquer")
        assert isinstance(result["fatos"], list)
        assert isinstance(result["pedidos"], list)
        assert isinstance(result["fundamentos_juridicos"], list)