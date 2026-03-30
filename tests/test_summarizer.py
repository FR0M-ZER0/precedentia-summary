import pytest
from unittest.mock import patch, MagicMock, call
from src.services.summarizer import (
    fetch_unsummarized,
    generate_summary,
    save_summary,
    run_summarizer,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_qdrant_point(point_id: int, payload: dict):
    """Cria um ponto fake do Qdrant."""
    point = MagicMock()
    point.id = point_id
    point.payload = payload
    return point


def make_ollama_stream(text: str):
    """Simula o stream do ollama com um único chunk."""
    chunk = MagicMock()
    chunk.__getitem__ = lambda self, key: (
        {"message": {"content": text}}[key]
    )
    return iter([chunk])


PAYLOAD_FAKE = {
    "name": "Tema 1234 STF",
    "tribunal": "STF",
    "situation": "Ativo",
    "description": "Discussão sobre constitucionalidade de tributos estaduais.",
    "url": "https://stf.jus.br/tema1234",
}

SUMMARY_FAKE = "Este precedente trata da constitucionalidade de tributos estaduais perante o STF. A discussão envolve os limites do poder tributário dos estados. Encontra-se atualmente ativo."


# ── Testes de fetch_unsummarized ──────────────────────────────────────────────

class TestFetchUnsummarized:
    def test_retorna_pontos_sem_resumo(self):
        mock_client = MagicMock()
        point = make_qdrant_point(1, PAYLOAD_FAKE)
        mock_client.scroll.return_value = ([point], None)

        result = fetch_unsummarized(mock_client)

        assert len(result) == 1
        assert result[0].id == 1

    def test_retorna_lista_vazia_quando_todos_resumidos(self):
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        result = fetch_unsummarized(mock_client)

        assert result == []

    def test_chama_scroll_com_filtro_is_null(self):
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        fetch_unsummarized(mock_client)

        _, kwargs = mock_client.scroll.call_args
        scroll_filter = kwargs.get("scroll_filter")

        assert scroll_filter is not None
        must_conditions = scroll_filter.must
        assert len(must_conditions) == 1
        # IsNullCondition guarda o campo dentro de .is_null.key
        assert must_conditions[0].is_null.key == "summary"

    def test_nao_retorna_vetores(self):
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        fetch_unsummarized(mock_client)

        _, kwargs = mock_client.scroll.call_args
        assert kwargs.get("with_vectors") is False

    def test_retorna_payloads(self):
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        fetch_unsummarized(mock_client)

        _, kwargs = mock_client.scroll.call_args
        assert kwargs.get("with_payload") is True

    def test_respeita_limite_de_lote(self):
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        fetch_unsummarized(mock_client, limit=5)

        _, kwargs = mock_client.scroll.call_args
        assert kwargs.get("limit") == 5


# ── Testes de generate_summary ────────────────────────────────────────────────

class TestGenerateSummary:
    @patch("src.services.summarizer.ollama.chat")
    def test_retorna_string(self, mock_chat):
        mock_chat.return_value = make_ollama_stream(SUMMARY_FAKE)
        point = make_qdrant_point(1, PAYLOAD_FAKE)

        result = generate_summary(point.payload)

        assert isinstance(result, str)
        assert len(result) > 0

    @patch("src.services.summarizer.ollama.chat")
    def test_strips_think_block(self, mock_chat):
        """Modelos de raciocínio emitem <think>...</think> — deve ser ignorado."""
        with_think = f"<think>pensamento interno</think>\n{SUMMARY_FAKE}"
        mock_chat.return_value = make_ollama_stream(with_think)
        point = make_qdrant_point(1, PAYLOAD_FAKE)

        result = generate_summary(point.payload)

        assert "<think>" not in result
        assert result == SUMMARY_FAKE

    @patch("src.services.summarizer.ollama.chat")
    def test_prompt_contem_campos_do_payload(self, mock_chat):
        """O prompt enviado ao modelo deve incluir os campos principais do payload."""
        mock_chat.return_value = make_ollama_stream(SUMMARY_FAKE)
        point = make_qdrant_point(1, PAYLOAD_FAKE)

        generate_summary(point.payload)

        _, kwargs = mock_chat.call_args
        prompt = kwargs["messages"][0]["content"]

        assert PAYLOAD_FAKE["name"] in prompt
        assert PAYLOAD_FAKE["tribunal"] in prompt
        assert PAYLOAD_FAKE["situation"] in prompt
        assert PAYLOAD_FAKE["description"] in prompt

    @patch("src.services.summarizer.ollama.chat")
    def test_concatena_chunks_do_stream(self, mock_chat):
        """O stream pode entregar o texto em múltiplos chunks."""
        chunks = ["Este precedente ", "trata de tributos.", " Fim."]

        def make_multi_stream(texts):
            for t in texts:
                chunk = MagicMock()
                chunk.__getitem__ = lambda self, key, t=t: (
                    {"message": {"content": t}}[key]
                )
                yield chunk

        mock_chat.return_value = make_multi_stream(chunks)
        point = make_qdrant_point(1, PAYLOAD_FAKE)

        result = generate_summary(point.payload)

        assert result == "Este precedente trata de tributos. Fim."


# ── Testes de save_summary ────────────────────────────────────────────────────

class TestSaveSummary:
    def test_chama_set_payload_com_summary(self):
        mock_client = MagicMock()

        save_summary(mock_client, point_id=42, summary=SUMMARY_FAKE)

        mock_client.set_payload.assert_called_once()
        _, kwargs = mock_client.set_payload.call_args
        assert kwargs["payload"] == {"summary": SUMMARY_FAKE}
        assert 42 in kwargs["points"]

    def test_nao_sobrescreve_outros_campos(self):
        """set_payload é aditivo — apenas o campo summary deve ser passado."""
        mock_client = MagicMock()

        save_summary(mock_client, point_id=1, summary=SUMMARY_FAKE)

        _, kwargs = mock_client.set_payload.call_args
        assert list(kwargs["payload"].keys()) == ["summary"]


# ── Testes de run_summarizer ──────────────────────────────────────────────────

class TestRunSummarizer:
    @patch("src.services.summarizer.save_summary")
    @patch("src.services.summarizer.generate_summary")
    @patch("src.services.summarizer.fetch_unsummarized")
    @patch("src.services.summarizer.get_qdrant_client")
    def test_processa_pontos_pendentes(
        self, mock_get_client, mock_fetch, mock_generate, mock_save
    ):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        points = [
            make_qdrant_point(1, PAYLOAD_FAKE),
            make_qdrant_point(2, PAYLOAD_FAKE),
        ]
        mock_fetch.return_value = points
        mock_generate.return_value = SUMMARY_FAKE

        run_summarizer()

        assert mock_generate.call_count == 2
        assert mock_save.call_count == 2

    @patch("src.services.summarizer.save_summary")
    @patch("src.services.summarizer.generate_summary")
    @patch("src.services.summarizer.fetch_unsummarized")
    @patch("src.services.summarizer.get_qdrant_client")
    def test_nao_processa_quando_nao_ha_pendentes(
        self, mock_get_client, mock_fetch, mock_generate, mock_save
    ):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_fetch.return_value = []

        run_summarizer()

        mock_generate.assert_not_called()
        mock_save.assert_not_called()

    @patch("src.services.summarizer.save_summary")
    @patch("src.services.summarizer.generate_summary")
    @patch("src.services.summarizer.fetch_unsummarized")
    @patch("src.services.summarizer.get_qdrant_client")
    def test_continua_apos_erro_em_um_ponto(
        self, mock_get_client, mock_fetch, mock_generate, mock_save
    ):
        """Falha em um ponto não deve interromper o processamento dos demais."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        points = [
            make_qdrant_point(1, PAYLOAD_FAKE),
            make_qdrant_point(2, PAYLOAD_FAKE),
        ]
        mock_fetch.return_value = points
        mock_generate.side_effect = [Exception("Ollama indisponível"), SUMMARY_FAKE]

        run_summarizer()

        assert mock_generate.call_count == 2
        assert mock_save.call_count == 1

    @patch("src.services.summarizer.save_summary")
    @patch("src.services.summarizer.generate_summary")
    @patch("src.services.summarizer.fetch_unsummarized")
    @patch("src.services.summarizer.get_qdrant_client")
    def test_fecha_cliente_ao_final(
        self, mock_get_client, mock_fetch, mock_generate, mock_save
    ):
        """O cliente Qdrant deve ser fechado mesmo em caso de erro."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_fetch.side_effect = Exception("Qdrant indisponível")

        run_summarizer()

        mock_client.close.assert_called_once()

    @patch("src.services.summarizer.save_summary")
    @patch("src.services.summarizer.generate_summary")
    @patch("src.services.summarizer.fetch_unsummarized")
    @patch("src.services.summarizer.get_qdrant_client")
    def test_salva_resumo_para_o_ponto_correto(
        self, mock_get_client, mock_fetch, mock_generate, mock_save
    ):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_fetch.return_value = [make_qdrant_point(99, PAYLOAD_FAKE)]
        mock_generate.return_value = SUMMARY_FAKE

        run_summarizer()

        mock_save.assert_called_once_with(mock_client, 99, SUMMARY_FAKE)
