import argparse
import datetime
import json
from pathlib import Path

import pandas as pd

from sentiment_analyzer.temporal import build_temporal_tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate country/year emotion tables from analyzed data."
    )
    parser.add_argument("input_path", help="Analyzed CSV produced by the CLI.")
    parser.add_argument("--emotion-threshold", type=float, default=0.5)
    parser.add_argument(
        "--partial-year",
        type=int,
        help="Year whose data is incomplete, recorded in every output table.",
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        parser.error(f"input file not found: {input_path}")
    if input_path.suffix.lower() != ".csv":
        parser.error("temporal analysis currently requires an analyzed CSV")

    frame = pd.read_csv(input_path, low_memory=False)
    tables = build_temporal_tables(
        frame,
        threshold=args.emotion_threshold,
        partial_year=args.partial_year,
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        args.output_dir
        or f"outputs/temporal_{timestamp}_{input_path.stem[:25]}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "coverage": "country_year_coverage.csv",
        "primary": "country_year_emotion_primary.csv",
        "multilabel": "country_year_emotion_multilabel.csv",
        "top3": "country_year_top3.csv",
        "yoy": "country_year_emotion_yoy.csv",
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
    print(f"Temporal analysis saved to: {output_dir}")


if __name__ == "__main__":
    main()
