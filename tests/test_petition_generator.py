import pytest
from unittest.mock import patch, MagicMock
from src.app import app
from src.services.petition_generator import generate_petition, edit_petition


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── Helper: simula o stream da OpenAI ────────────────────────────────────────

def make_openai_stream(text: str):
    """Simula o stream da OpenAI com um único chunk."""
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = text
    return iter([chunk])


def make_openai_stream_multi(parts: list[str]):
    """Simula o stream da OpenAI com múltiplos chunks."""
    chunks = []
    for part in parts:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = part
        chunks.append(chunk)
    return iter(chunks)


def make_openai_stream_with_none(text: str):
    """Simula stream com um chunk None seguido do conteúdo real (caso real da API)."""
    chunk_none = MagicMock()
    chunk_none.choices = [MagicMock()]
    chunk_none.choices[0].delta.content = None

    chunk_real = MagicMock()
    chunk_real.choices = [MagicMock()]
    chunk_real.choices[0].delta.content = text

    return iter([chunk_none, chunk_real])


# ── Payloads de teste ─────────────────────────────────────────────────────────

PAYLOAD_COMPLETO = {
    "author_description": "João Silva, brasileiro, advogado, CPF 123.456.789-00",
    "defendant_description": "Empresa X Ltda, CNPJ 00.000.000/0001-00",
    "action_type": "Ação de Cobrança",
    "tribunal": "TJSP",
    "facts_summary": "O réu deixou de pagar as parcelas acordadas em contrato.",
    "files": ["Contrato assinado em 01/01/2024 com valor de R$ 10.000,00."],
    "requests": ["Condenação ao pagamento de R$ 10.000,00", "Correção monetária"],
    "cause_value": "R$ 10.000,00",
    "urgent_injunction": False,
    "free_justice": True,
    "precedents": [
        {
            "name": "REsp 1.234.567/SP",
            "question": "Cobrança de dívida contratual",
            "description": "O STJ firmou que a mora contratual gera dever de indenizar.",
        }
    ],
}

PETICAO_FAKE = """# EXCELENTÍSSIMO SENHOR DOUTOR JUIZ DE DIREITO

João Silva vem propor **Ação de Cobrança** em face de Empresa X Ltda.

## I – DOS FATOS
O réu deixou de pagar as parcelas acordadas.

## II – DO DIREITO
Conforme REsp 1.234.567/SP...

## III – DOS PEDIDOS
1. Condenação ao pagamento de R$ 10.000,00.

Dá-se à causa o valor de R$ 10.000,00."""


# ══════════════════════════════════════════════════════════════════════════════
# Testes da rota POST /api/petition/generate
# ══════════════════════════════════════════════════════════════════════════════

class TestRotaGenerate:

    # ── Validações de entrada ─────────────────────────────────────────────────

    def test_sem_body_retorna_400(self, client):
        r = client.post("/api/petition/generate")
        assert r.status_code == 400

    def test_body_vazio_retorna_400(self, client):
        r = client.post("/api/petition/generate", json={})
        assert r.status_code == 400

    @pytest.mark.parametrize("campo_ausente", [
        "author_description",
        "defendant_description",
        "action_type",
        "tribunal",
        "facts_summary",
        "cause_value",
    ])
    def test_campo_obrigatorio_ausente_retorna_400(self, client, campo_ausente):
        payload = {k: v for k, v in PAYLOAD_COMPLETO.items() if k != campo_ausente}
        r = client.post("/api/petition/generate", json=payload)
        assert r.status_code == 400
        data = r.get_json()
        assert campo_ausente in data["error"]

    @pytest.mark.parametrize("campo_ausente", [
        "author_description",
        "defendant_description",
        "action_type",
        "tribunal",
        "facts_summary",
        "cause_value",
    ])
    def test_campo_obrigatorio_vazio_retorna_400(self, client, campo_ausente):
        payload = {**PAYLOAD_COMPLETO, campo_ausente: "   "}
        r = client.post("/api/petition/generate", json=payload)
        assert r.status_code == 400

    def test_files_nao_lista_retorna_400(self, client):
        payload = {**PAYLOAD_COMPLETO, "files": "texto solto"}
        r = client.post("/api/petition/generate", json=payload)
        assert r.status_code == 400
        assert "files" in r.get_json()["error"]

    def test_requests_nao_lista_retorna_400(self, client):
        payload = {**PAYLOAD_COMPLETO, "requests": "pedido solto"}
        r = client.post("/api/petition/generate", json=payload)
        assert r.status_code == 400
        assert "requests" in r.get_json()["error"]

    def test_precedents_nao_lista_retorna_400(self, client):
        payload = {**PAYLOAD_COMPLETO, "precedents": {"name": "algo"}}
        r = client.post("/api/petition/generate", json=payload)
        assert r.status_code == 400
        assert "precedents" in r.get_json()["error"]

    # ── Casos de sucesso ──────────────────────────────────────────────────────

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_payload_completo_retorna_200(self, mock_create, client):
        mock_create.return_value = make_openai_stream(PETICAO_FAKE)
        r = client.post("/api/petition/generate", json=PAYLOAD_COMPLETO)
        assert r.status_code == 200
        data = r.get_json()
        assert "content" in data
        assert len(data["content"]) > 0

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_campos_opcionais_ausentes_retorna_200(self, mock_create, client):
        """Apenas os campos obrigatórios devem ser suficientes."""
        mock_create.return_value = make_openai_stream(PETICAO_FAKE)
        payload_minimo = {
            "author_description": "João Silva",
            "defendant_description": "Empresa X",
            "action_type": "Ação de Cobrança",
            "tribunal": "TJSP",
            "facts_summary": "Fatos do caso.",
            "cause_value": "R$ 5.000,00",
        }
        r = client.post("/api/petition/generate", json=payload_minimo)
        assert r.status_code == 200

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_listas_vazias_aceitas(self, mock_create, client):
        """files, requests e precedents podem ser listas vazias."""
        mock_create.return_value = make_openai_stream(PETICAO_FAKE)
        payload = {**PAYLOAD_COMPLETO, "files": [], "requests": [], "precedents": []}
        r = client.post("/api/petition/generate", json=payload)
        assert r.status_code == 200

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_retorno_contem_texto_gerado(self, mock_create, client):
        mock_create.return_value = make_openai_stream(PETICAO_FAKE)
        r = client.post("/api/petition/generate", json=PAYLOAD_COMPLETO)
        data = r.get_json()
        assert data["content"] == PETICAO_FAKE

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_erro_interno_retorna_500(self, mock_create, client):
        mock_create.side_effect = Exception("OpenAI indisponível")
        r = client.post("/api/petition/generate", json=PAYLOAD_COMPLETO)
        assert r.status_code == 500
        assert "error" in r.get_json()


# ══════════════════════════════════════════════════════════════════════════════
# Testes da rota POST /api/petition/edit
# ══════════════════════════════════════════════════════════════════════════════

class TestRotaEdit:

    # ── Validações de entrada ─────────────────────────────────────────────────

    def test_sem_body_retorna_400(self, client):
        r = client.post("/api/petition/edit")
        assert r.status_code == 400

    def test_content_ausente_retorna_400(self, client):
        r = client.post("/api/petition/edit", json={"change": "Altere o parágrafo 1."})
        assert r.status_code == 400
        assert "content" in r.get_json()["error"]

    def test_change_ausente_retorna_400(self, client):
        r = client.post("/api/petition/edit", json={"content": PETICAO_FAKE})
        assert r.status_code == 400
        assert "change" in r.get_json()["error"]

    def test_content_vazio_retorna_400(self, client):
        r = client.post("/api/petition/edit", json={"content": "   ", "change": "Altere algo."})
        assert r.status_code == 400

    def test_change_vazio_retorna_400(self, client):
        r = client.post("/api/petition/edit", json={"content": PETICAO_FAKE, "change": "   "})
        assert r.status_code == 400

    # ── Casos de sucesso ──────────────────────────────────────────────────────

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_edicao_valida_retorna_200(self, mock_create, client):
        peticao_editada = PETICAO_FAKE.replace("Empresa X Ltda", "Empresa Y Ltda")
        mock_create.return_value = make_openai_stream(peticao_editada)
        r = client.post("/api/petition/edit", json={
            "content": PETICAO_FAKE,
            "change": "Altere o nome da ré para Empresa Y Ltda.",
        })
        assert r.status_code == 200
        data = r.get_json()
        assert "content" in data
        assert "Empresa Y Ltda" in data["content"]

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_retorno_contem_conteudo_editado(self, mock_create, client):
        mock_create.return_value = make_openai_stream("petição editada")
        r = client.post("/api/petition/edit", json={
            "content": PETICAO_FAKE,
            "change": "Qualquer alteração.",
        })
        assert r.get_json()["content"] == "petição editada"

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_erro_interno_retorna_500(self, mock_create, client):
        mock_create.side_effect = Exception("OpenAI indisponível")
        r = client.post("/api/petition/edit", json={
            "content": PETICAO_FAKE,
            "change": "Altere algo.",
        })
        assert r.status_code == 500


# ══════════════════════════════════════════════════════════════════════════════
# Testes do serviço generate_petition
# ══════════════════════════════════════════════════════════════════════════════

class TestServicoGenerate:

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_retorna_string(self, mock_create):
        mock_create.return_value = make_openai_stream(PETICAO_FAKE)
        result = generate_petition(**{k: v for k, v in PAYLOAD_COMPLETO.items()})
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_concatena_chunks_do_stream(self, mock_create):
        """O stream pode entregar o texto em múltiplos chunks."""
        mock_create.return_value = make_openai_stream_multi(["# Petição\n", "## I – DOS FATOS\n", "Fatos aqui."])
        result = generate_petition(**{k: v for k, v in PAYLOAD_COMPLETO.items()})
        assert result == "# Petição\n## I – DOS FATOS\nFatos aqui."

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_ignora_chunks_none(self, mock_create):
        """Chunks com delta.content None não devem causar erro nem ser concatenados."""
        mock_create.return_value = make_openai_stream_with_none(PETICAO_FAKE)
        result = generate_petition(**{k: v for k, v in PAYLOAD_COMPLETO.items()})
        assert result == PETICAO_FAKE

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_prompt_contem_campos_principais(self, mock_create):
        """O prompt enviado ao modelo deve conter todos os dados do payload."""
        mock_create.return_value = make_openai_stream(PETICAO_FAKE)
        generate_petition(**{k: v for k, v in PAYLOAD_COMPLETO.items()})

        _, kwargs = mock_create.call_args
        messages = kwargs["messages"]
        user_prompt = next(m["content"] for m in messages if m["role"] == "user")

        assert PAYLOAD_COMPLETO["author_description"] in user_prompt
        assert PAYLOAD_COMPLETO["defendant_description"] in user_prompt
        assert PAYLOAD_COMPLETO["action_type"] in user_prompt
        assert PAYLOAD_COMPLETO["tribunal"] in user_prompt
        assert PAYLOAD_COMPLETO["facts_summary"] in user_prompt
        assert PAYLOAD_COMPLETO["cause_value"] in user_prompt

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_prompt_contem_texto_dos_arquivos(self, mock_create):
        mock_create.return_value = make_openai_stream(PETICAO_FAKE)
        generate_petition(**{k: v for k, v in PAYLOAD_COMPLETO.items()})

        _, kwargs = mock_create.call_args
        messages = kwargs["messages"]
        user_prompt = next(m["content"] for m in messages if m["role"] == "user")

        assert PAYLOAD_COMPLETO["files"][0] in user_prompt

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_prompt_contem_precedentes(self, mock_create):
        mock_create.return_value = make_openai_stream(PETICAO_FAKE)
        generate_petition(**{k: v for k, v in PAYLOAD_COMPLETO.items()})

        _, kwargs = mock_create.call_args
        messages = kwargs["messages"]
        user_prompt = next(m["content"] for m in messages if m["role"] == "user")

        precedente = PAYLOAD_COMPLETO["precedents"][0]
        assert precedente["name"] in user_prompt
        assert precedente["question"] in user_prompt
        assert precedente["description"] in user_prompt

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_tutela_urgencia_indicada_no_prompt(self, mock_create):
        mock_create.return_value = make_openai_stream(PETICAO_FAKE)
        payload = {**PAYLOAD_COMPLETO, "urgent_injunction": True}
        generate_petition(**{k: v for k, v in payload.items()})

        _, kwargs = mock_create.call_args
        messages = kwargs["messages"]
        user_prompt = next(m["content"] for m in messages if m["role"] == "user")

        assert "Sim" in user_prompt

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_sem_tutela_urgencia_indicada_no_prompt(self, mock_create):
        mock_create.return_value = make_openai_stream(PETICAO_FAKE)
        payload = {**PAYLOAD_COMPLETO, "urgent_injunction": False}
        generate_petition(**{k: v for k, v in payload.items()})

        _, kwargs = mock_create.call_args
        messages = kwargs["messages"]
        user_prompt = next(m["content"] for m in messages if m["role"] == "user")

        assert "Não" in user_prompt

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_sem_precedentes_nao_gera_erro(self, mock_create):
        mock_create.return_value = make_openai_stream(PETICAO_FAKE)
        payload = {**PAYLOAD_COMPLETO, "precedents": []}
        result = generate_petition(**{k: v for k, v in payload.items()})
        assert isinstance(result, str)

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_sem_arquivos_nao_gera_erro(self, mock_create):
        mock_create.return_value = make_openai_stream(PETICAO_FAKE)
        payload = {**PAYLOAD_COMPLETO, "files": []}
        result = generate_petition(**{k: v for k, v in payload.items()})
        assert isinstance(result, str)

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_resultado_sem_espacos_extras(self, mock_create):
        """O retorno deve ter strip() aplicado."""
        mock_create.return_value = make_openai_stream(f"  \n{PETICAO_FAKE}\n  ")
        result = generate_petition(**{k: v for k, v in PAYLOAD_COMPLETO.items()})
        assert result == PETICAO_FAKE


# ══════════════════════════════════════════════════════════════════════════════
# Testes do serviço edit_petition
# ══════════════════════════════════════════════════════════════════════════════

class TestServicoEdit:

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_retorna_string(self, mock_create):
        mock_create.return_value = make_openai_stream(PETICAO_FAKE)
        result = edit_petition(content=PETICAO_FAKE, change="Altere o réu.")
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_prompt_contem_content_e_change(self, mock_create):
        """O prompt deve incluir tanto o conteúdo atual quanto a instrução de mudança."""
        mock_create.return_value = make_openai_stream(PETICAO_FAKE)
        change = "Substituir o nome do réu por Empresa Y."
        edit_petition(content=PETICAO_FAKE, change=change)

        _, kwargs = mock_create.call_args
        messages = kwargs["messages"]
        user_prompt = next(m["content"] for m in messages if m["role"] == "user")

        assert PETICAO_FAKE in user_prompt
        assert change in user_prompt

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_concatena_chunks_do_stream(self, mock_create):
        mock_create.return_value = make_openai_stream_multi(["parte 1 ", "parte 2"])
        result = edit_petition(content=PETICAO_FAKE, change="Altere algo.")
        assert result == "parte 1 parte 2"

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_ignora_chunks_none(self, mock_create):
        mock_create.return_value = make_openai_stream_with_none("petição editada")
        result = edit_petition(content=PETICAO_FAKE, change="Altere algo.")
        assert result == "petição editada"

    @patch("src.services.petition_generator.client.chat.completions.create")
    def test_resultado_sem_espacos_extras(self, mock_create):
        mock_create.return_value = make_openai_stream(f"\n  {PETICAO_FAKE}  \n")
        result = edit_petition(content=PETICAO_FAKE, change="Altere algo.")
        assert result == PETICAO_FAKE
