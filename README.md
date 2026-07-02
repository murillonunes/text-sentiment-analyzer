# Text Sentiment & Emotion Analyzer

A Python-based Natural Language Processing (NLP) pipeline for sentiment analysis and emotion classification on text datasets (such as Steam game reviews). The application supports multilingual Hugging Face models, robust text preprocessing, interactive HTML dashboard generation, and automated execution logging.

---

## 🚀 Key Features

* **Emotion and Sentiment Classification:**
  * Supports **Emotion** models (e.g., `tabularisai/multilingual-emotion-classification`), which classify texts into 11 granular categories (joy, anger, sadness, contempt, etc.).
  * Supports **Direct Sentiment** models (e.g., `cardiffnlp/xlm-roberta-base-tweet-sentiment-pt`), classifying text directly into positive, negative, or neutral.
* **User Recommendation-Based Heuristic Correction (Steam):**
  * Automatically corrects sarcasm/irony false negatives (e.g., classifying *"MELHOR JOGO JÁ CRIADO !!!"* / *"BEST GAME EVER CREATED !!!"* as *contempt* or *anger*). If the user recommendation (`voted_up`) is `True`, negative emotions detected in short reviews are adjusted to `joy`.
  * Adds an adjustment tag column (`emotion_adjustment`) indicating whether the prediction was `adjusted` or remained `original`.
* **Smart Text Preprocessor:**
  * Automatic removal of HTML tags and URLs.
  * Normalization of whitespaces and line breaks.
  * **Standalone Punctuation Filtering:** Removes isolated punctuation (such as `" !!!"`) to ensure exact counts for the minimum word filter (`--min-words`).
* **Premium HTML Dashboard & Visualizations:**
  * Modern, interactive dark-theme HTML dashboard report.
  * Displays execution time, agreement rates between predictions and user recommendations, a detailed sample table highlighting adjustments, and automatically generated plots (`matplotlib` & `seaborn`).
* **Automated Logging System:**
  * **Global Log (`outputs/executions.log`):** Maintains a historical ledger of all CLI runs.
  * **Local Log:** Saves a run-specific `execution.log` file in the output directory created for that specific execution, alongside the output files.

---

## 📂 Project Structure

* `src/sentiment_analyzer/`
  * [preprocessor.py](src/sentiment_analyzer/preprocessor.py): Text cleaning and normalization module.
  * [analyzer.py](src/sentiment_analyzer/analyzer.py): Pipeline orchestrator (DataFrame analysis, heuristic correction, and metrics calculation).
  * [cli.py](src/sentiment_analyzer/cli.py): Command Line Interface (CLI), visualization generation, HTML dashboard, and logging.
  * `backends/`
    * [transformers.py](src/sentiment_analyzer/backends/transformers.py): Connector for Hugging Face models.
* `tests/`: Unit tests with `pytest`.
* `outputs/`: Default output directory grouping results by execution run (`run_YYYYMMDD_HHMMSS_<filename>`).

---

## 🛠️ Installation & Setup

Make sure you have Python 3.10+ installed.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/murillonunes/text-sentiment-analyzer.git
   cd text-sentiment-analyzer
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Running the CLI

The CLI accepts input files in `.csv` and `.json` formats.

### Usage Examples

1. **Basic Execution (Default Emotion Model):**
   ```bash
   python -m sentiment_analyzer.cli tests/fixtures/sample_reviews.csv
   ```

2. **Defining a Minimum Word Count Filter:**
   Filters and ignores reviews with fewer than 5 words:
   ```bash
   python -m sentiment_analyzer.cli tests/fixtures/sample_reviews.csv --min-words 5
   ```

3. **Using an Alternative Hugging Face Model (Portuguese Sentiment):**
   ```bash
   python -m sentiment_analyzer.cli tests/fixtures/sample_reviews.csv -m cardiffnlp/xlm-roberta-base-tweet-sentiment-pt
   ```

### Available Parameters

| Flag | Description | Default |
|------|-------------|---------|
| `input_path` | Path to the input CSV or JSON file. | *(Required)* |
| `-m`, `--model-name` | Hugging Face model identifier/path to use. | `tabularisai/multilingual-emotion-classification` |
| `-o`, `--output-path` | Custom file path to save the analyzed CSV. | `outputs/run_<run_id>/<name>_analyzed.csv` |
| `-tc`, `--text-column` | Column name containing the text to analyze. | `review_text` |
| `-vc`, `--voted-up-column` | Column name containing the user recommendation (`True` / `False`). | `voted_up` |
| `-b`, `--batch-size` | Batch size for model inference. | `32` |
| `--min-words` | Ignore texts with a word count below this threshold. | `0` |
| `--skip-report` | Skip generating the charts and the HTML dashboard report. | `False` |

---

## 🧪 Running Tests

To verify code stability and classification logic, run the test suite:

```bash
python -m pytest
```
