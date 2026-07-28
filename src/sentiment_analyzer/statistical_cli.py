import argparse
import datetime
import json
from pathlib import Path

import pandas as pd

from sentiment_analyzer.statistical import build_statistical_tables


def _parse_language_mapping(values: list[str]) -> dict[str, str]:
    mappings = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                "dominant languages must use COUNTRY=language"
            )
        country, language = value.split("=", 1)
        mappings[country.strip()] = language.strip()
    return mappings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inferential temporal emotion analysis by country."
    )
    parser.add_argument("input_path")
    parser.add_argument("--complete-through", type=int, required=True)
    parser.add_argument("--partial-year", type=int)
    parser.add_argument("--emotion-threshold", type=float, default=0.5)
    parser.add_argument(
        "--dominant-language",
        action="append",
        default=[],
        metavar="COUNTRY=LANGUAGE",
        help="Repeat for each country used in language sensitivity.",
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    try:
        dominant_languages = _parse_language_mapping(
            args.dominant_language
        )
    except ValueError as error:
        parser.error(str(error))
    if not dominant_languages:
        parser.error("at least one --dominant-language is required")

    input_path = Path(args.input_path)
    if not input_path.exists():
        parser.error(f"input file not found: {input_path}")
    frame = pd.read_csv(input_path, low_memory=False)
    tables = build_statistical_tables(
        frame,
        threshold=args.emotion_threshold,
        complete_through=args.complete_through,
        partial_year=args.partial_year,
        dominant_languages=dominant_languages,
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        args.output_dir
        or f"outputs/statistical_{timestamp}_{input_path.stem[:20]}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "annual": "annual_emotion_estimates.csv",
        "binary_tests": "country_emotion_temporal_tests.csv",
        "distribution_tests": "primary_distribution_tests.csv",
        "trends": "logistic_trends.csv",
        "interactions": "country_year_interactions.csv",
    }
    for key, filename in files.items():
        tables[key].to_csv(output_dir / filename, index=False)
    summary = {
        "generated_at": datetime.datetime.now().isoformat(),
        "input_file": str(input_path),
        **tables["summary"],
        "files": files,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Statistical analysis saved to: {output_dir}")


if __name__ == "__main__":
    main()
