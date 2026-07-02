# Text Sentiment & Emotion Analyzer

Pipeline de processamento de linguagem natural (NLP) em Python para análise de sentimentos e classificação de emoções em textos (como reviews de jogos da Steam). A aplicação suporta modelos multilíngues do Hugging Face, pré-processamento robusto de texto, geração de relatórios interativos em HTML e logs automatizados.

---

## 🚀 Principais Funcionalidades

* **Classificação de Emoções e Sentimentos:**
  * Suporta modelos de **Emoção** (ex: `tabularisai/multilingual-emotion-classification`), que classificam textos em até 11 classes (alegria, raiva, tristeza, desprezo, etc.).
  * Suporta modelos de **Sentimento Direto** (ex: `cardiffnlp/xlm-roberta-base-tweet-sentiment-pt`), classificando em positivo, negativo ou neutro.
* **Heurística de Correção Baseada no Voto do Usuário (Steam):**
  * Corrige automaticamente falsos negativos de sarcasmo/ironia (como classificar *"MELHOR JOGO JÁ CRIADO !!!"* como *contempt* ou *anger*). Se o voto do usuário (`voted_up`) for `True`, emoções negativas detectadas em frases curtas são ajustadas para `joy`.
  * Adiciona uma coluna de tag (`emotion_adjustment`) sinalizando se o resultado foi `adjusted` ou se manteve o `original`.
* **Pré-Processador de Texto Inteligente:**
  * Remoção automática de tags HTML e URLs.
  * Normalização de espaços em branco e quebras de linha.
  * **Filtro de Pontuação Solta:** Remove pontuações isoladas (como `" !!!"`) para garantir a exatidão no filtro de número mínimo de palavras (`--min-words`).
* **Visualização e Relatório HTML Premium:**
  * Dashboard interativo em HTML com tema escuro moderno.
  * Exibe estatísticas de tempo de execução, taxa de concordância entre predições e recomendações, tabelas detalhadas das últimas reviews analisadas (sinalizando os ajustes) e gráficos gerados automaticamente (`matplotlib` & `seaborn`).
* **Sistema de Logging Histórico:**
  * **Log Global (`outputs/executions.log`):** Mantém o histórico completo de todas as execuções da CLI.
  * **Log Local:** Grava um arquivo individual de log (`execution.log`) na pasta específica de cada execução, ao lado das saídas salvas.

---

## 📂 Estrutura do Projeto

* `src/sentiment_analyzer/`
  * [preprocessor.py](src/sentiment_analyzer/preprocessor.py): Módulo de limpeza e normalização do texto.
  * [analyzer.py](src/sentiment_analyzer/analyzer.py): Orquestrador do pipeline (análise de DataFrames, aplicação de heurísticas e cálculo de métricas).
  * [cli.py](src/sentiment_analyzer/cli.py): Interface de Linha de Comando (CLI), geração de relatórios gráficos/HTML e gravação de logs.
  * `backends/`
    * [transformers.py](src/sentiment_analyzer/backends/transformers.py): Conector para modelos do Hugging Face.
* `tests/`: Testes unitários com `pytest`.
* `outputs/`: Diretório padrão de saída que agrupa os resultados por execução (`run_YYYYMMDD_HHMMSS_<nome_do_arquivo>`).

---

## 🛠️ Instalação e Configuração

Certifique-se de ter o Python 3.10+ instalado.

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/murillonunes/text-sentiment-analyzer.git
   cd text-sentiment-analyzer
   ```

2. **Crie e ative o ambiente virtual:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Como Executar a CLI

A CLI aceita arquivos nos formatos `.csv` e `.json`.

### Exemplos de Uso

1. **Execução Básica (Modelo Padrão de Emoções):**
   ```bash
   python -m sentiment_analyzer.cli caminho_do_seu_arquivo.csv
   ```

2. **Definindo um Limite de Palavras Mínimo (Filtro):**
   Filtra e ignora avaliações que tenham menos de 5 palavras:
   ```bash
   python -m sentiment_analyzer.cli caminho_do_seu_arquivo.csv --min-words 5
   ```

3. **Utilizando outro Modelo do Hugging Face (Sentimento em Português):**
   ```bash
   python -m sentiment_analyzer.cli caminho_do_seu_arquivo.csv -m cardiffnlp/xlm-roberta-base-tweet-sentiment-pt
   ```

### Parâmetros Disponíveis

| Flag | Descrição | Padrão |
|------|-----------|--------|
| `input_path` | Caminho para o arquivo CSV/JSON de entrada. | *(Obrigatório)* |
| `-m`, `--model-name` | Nome/caminho do modelo Hugging Face a utilizar. | `tabularisai/multilingual-emotion-classification` |
| `-o`, `--output-path` | Caminho de destino para salvar o arquivo analisado. | `outputs/run_<run_id>/<nome>_analyzed.csv` |
| `-tc`, `--text-column` | Coluna com o texto a ser analisado. | `review_text` |
| `-vc`, `--voted-up-column` | Coluna com o voto Steam (`True` / `False`). | `voted_up` |
| `-b`, `--batch-size` | Tamanho do lote (*batch size*) para inferência. | `32` |
| `--min-words` | Ignora textos com contagem de palavras menor que o valor. | `0` |
| `--skip-report` | Pula a geração dos gráficos e do dashboard HTML. | `False` |

---

## 🧪 Rodando os Testes

Para garantir a estabilidade das modificações e lógica de classificação, você pode rodar os testes usando:

```bash
python -m pytest
```
