# Text Sentiment & Emotion Analyzer

A Python-based Natural Language Processing (NLP) pipeline for sentiment analysis and emotion classification on text datasets (such as Steam game reviews). The application supports multilingual Hugging Face models, robust text preprocessing, interactive HTML dashboard generation, and automated execution logging.

---

## 🚀 Key Features

* **Emotion and Sentiment Classification:**
  * Supports **Emotion** models (e.g., `tabularisai/multilingual-emotion-classification`), which classify texts into 11 granular categories (joy, anger, sadness, contempt, etc.).
  * Supports **Direct Sentiment** models (e.g., `cardiffnlp/xlm-roberta-base-tweet-sentiment-pt`), classifying text directly into positive, negative, or neutral.
* **Auditable Multilingual Emotion Scores:**
  * Preserves the independent probability for every label returned by a multi-label model.
  * Keeps the raw top emotion and score for backward-compatible inspection.
  * Distinguishes analyzed, empty, skipped, and failed records through `analysis_status`.
* **User Recommendation Agreement Flag (Steam):**
  * Flags disagreement between `voted_up=True` and a negative top emotion without changing the model prediction or its score.
  * Normalizes boolean, numeric (`1`/`0`), and textual recommendation values.
  * Adds `adjustment_suggested` and `emotion_adjustment` audit columns.
* **Country and Year Metrics:**
  * Generates `emotion_summary.csv` with frequency, percentage, mean probability, and ranking overall, by country, and by country/year.
  * Uses an explicit multi-label threshold, which defaults to `0.5`.
* **Smart Text Preprocessor:**
  * Automatic removal of HTML tags and URLs.
  * Normalization of whitespaces and line breaks.
  * Preserves emojis, symbols, and emotional punctuation by default.
  * Optional legacy standalone-punctuation removal remains available through the Python API.
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
| `--emotion-threshold` | Independent probability threshold used for multi-label counts and rankings. | `0.5` |
| `--skip-report` | Skip generating the charts and the HTML dashboard report. | `False` |

The analyzed CSV keeps `emotion` and `emotion_score` as the raw top label and its
probability. It also contains one `emotion_<label>_score` column per model label. These
values are model probabilities between 0 and 1; they are not direct measurements of a
person's internal emotional intensity.

Rows not analyzed have an explicit `analysis_status` and missing scores, so they are excluded
from aggregate means. Chinese characters count as text units for `--min-words`, avoiding the
assumption that every supported language separates words with spaces.

---

## 🧪 Running Tests

To verify code stability and classification logic, run the test suite:

```bash
python -m pytest
```

## Experimental segmented analysis

Long reviews that exceed the model tokenizer limit can be evaluated in
overlapping token windows without changing the regular analysis output:

```bash
python -m sentiment_analyzer.segmented_cli \
  path/to/reviews.csv \
  --previous-results path/to/reviews_analyzed.csv \
  --only-over-limit \
  --stride 64 \
  --emotion-threshold 0.5 \
  --device cpu \
  --batch-size 8
```

This produces separate segment predictions and review-level comparisons using
both a token-weighted mean and the maximum score for each emotion. The segmented
results are experimental and do not overwrite the original model predictions.

## Temporal country analysis

An analyzed CSV can be consolidated into country/year tables for exclusive
top-emotion results and independent multi-label results:

```bash
python -m sentiment_analyzer.temporal_cli \
  path/to/reviews_analyzed.csv \
  --emotion-threshold 0.5 \
  --partial-year 2026
```

The command writes separate files for analysis coverage, primary emotions,
multi-label emotions, yearly top-three rankings, and year-over-year changes.
Rows with `empty_text` remain in the coverage table but are excluded from
emotion denominators.

## Inferential temporal analysis

Statistical tests and annual confidence intervals can be generated for complete
years, with a separate sensitivity period that includes a partial year:

```bash
python -m sentiment_analyzer.statistical_cli \
  path/to/reviews_analyzed.csv \
  --complete-through 2025 \
  --partial-year 2026 \
  --emotion-threshold 0.5 \
  --dominant-language BR=brazilian \
  --dominant-language CN=schinese \
  --dominant-language US=english
```

Outputs include Wilson confidence intervals, chi-square tests with Cramér's V,
per-emotion logistic yearly trends, country-by-year interaction tests, and a
dominant-language sensitivity analysis. Multiple tests use
Benjamini-Hochberg false-discovery-rate correction.
