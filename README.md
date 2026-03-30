# PrecedentIA - Summary

Serviço do sistema **PrecedentIA**, desenvolvido em Python, responsável por dois papéis centrais na plataforma: a **desconstrução de petições iniciais** em estruturas jurídicas padronizadas, e a **sumarização automática de precedentes** armazenados no banco vetorial.

## 🚀 Tecnologias Utilizadas

- **Python** - Linguagem principal do serviço
- **Flask** - Framework web para exposição da API REST
- **Ollama** - Execução local de modelos de linguagem para desconstrução e sumarização
- **Qdrant Client** - Integração com o banco de dados vetorial para leitura e atualização de precedentes
- **APScheduler** - Agendamento do job periódico de sumarização em background
- **Python-dotenv** - Gerenciamento de variáveis de ambiente via arquivo `.env`
- **Pytest** - Framework de testes unitários

## 🐳 Rodando a Infraestrutura (Docker)

Para que o **PrecedentIA** funcione, você precisa do Ollama e do Qdrant ativos. A forma mais rápida de subir ambos é via Docker Compose:

```bash
docker compose up -d
```

## 🧩 Serviços

### 📄 Desconstrução de petições (`/api/deconstruct`)

Endpoint REST que recebe o texto bruto de uma petição inicial e retorna uma estrutura JSON padronizada com os campos jurídicos extraídos: tipo da ação, tribunal, partes, fatos, fundamentos jurídicos, pedidos, valor da causa e data de ajuizamento.

O serviço é tolerante a variações de saída do modelo — remove blocos `<think>` de modelos de raciocínio e fences de markdown antes de parsear o JSON.

### 🔁 Sumarização de precedentes (job periódico)

Job executado em background a cada intervalo configurável que busca no Qdrant os precedentes que ainda não possuem resumo (campo `summary` ausente), gera um resumo em linguagem natural via LLM e persiste o resultado diretamente no payload do ponto — sem re-vetorizar ou afetar os demais campos.

## ⚙️ Rodando o Projeto

### 1️⃣ Verifique o ambiente Python

```bash
python --version
```

### 2️⃣ Crie e ative o ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### 3️⃣ Instale os pacotes Node.js e ative os hooks de commit

```bash
npm i
npm run prepare
```

### 4️⃣ Instale as dependências Python

```bash
pip install -r requirements.txt
```

### 5️⃣ Configure as variáveis de ambiente

```bash
cp .env.example .env
```

Preencha o `.env` com os valores adequados:

| Variável | Descrição |
|---|---|
| `OLLAMA_MODEL` | Modelo a ser utilizado (ex: `qwen3:8b`) |
| `OLLAMA_TEMPERATURE` | Temperatura do modelo (recomendado: `0`) |
| `FLASK_PORT` | Porta da API (padrão: `5000`) |
| `FLASK_DEBUG` | Modo debug do Flask (`true`/`false`) |
| `QDRANT_HOST` | Host do Qdrant |
| `QDRANT_PORT` | Porta do Qdrant (padrão: `6333`) |
| `QDRANT_COLLECTION` | Nome da coleção de precedentes |
| `SUMMARIZER_INTERVAL_MINUTES` | Intervalo do job de sumarização em minutos (padrão: `5`) |

**Nota:** Após subir o Ollama pela primeira vez, lembre-se de baixar o modelo configurado no seu `.env`
```bash
docker exec -it precedentia-ollama ollama pull qwen3:8b
```

### 6️⃣ Execute a aplicação

```bash
python -m src.app
```

## 🧪 Rodando os testes

Para executar toda a suite de testes:

```bash
pytest
```

Para rodar apenas um serviço específico:

```bash
pytest tests/test_deconstruct.py
pytest tests/test_summarizer.py
```

Para rodar com cobertura de código:

```bash
pytest --cov=src
```

## Saiba mais

Para verificar as padronizações usadas neste projeto, bem como demais documentações, visite o nosso [repositório principal](https://github.com/FR0M-ZER0/PrecedentIA)
