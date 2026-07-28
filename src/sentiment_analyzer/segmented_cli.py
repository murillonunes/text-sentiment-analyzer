import argparse
import datetime
import json
import os
from pathlib import Path

import pandas as pd

from sentiment_analyzer.analyzer import SentimentAnalyzer
from sentiment_analyzer.segmented import (
    aggregate_segment_scores,
    build_token_windows,
)


def _read_table(path: str) -> pd.DataFrame:
    if Path(path).suffix.lower() == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimental token-window analysis for long reviews."
    )
    parser.add_argument("input_path")
    parser.add_argument("--previous-results", required=True)
    parser.add_argument("--text-column", default="review_text")
    parser.add_argument("--id-column", default="recommendation_id")
    parser.add_argument(
        "--model-name",
        default="tabularisai/multilingual-emotion-classification",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--emotion-threshold", type=float, default=0.5)
    parser.add_argument("--only-over-limit", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    if not 0.0 <= args.emotion_threshold <= 1.0:
        parser.error("--emotion-threshold must be between 0 and 1")

    source = _read_table(args.input_path)
    previous = _read_table(args.previous_results)
    for column in (args.id_column, args.text_column):
        if column not in source:
            parser.error(f"input is missing required column: {column}")
    if args.id_column not in previous or "emotion" not in previous:
        parser.error("previous results must contain the ID and emotion columns")
    if not source[args.id_column].is_unique:
        parser.error("input IDs must be unique")
    if not previous[args.id_column].is_unique:
        parser.error("previous-result IDs must be unique")

    analyzer = SentimentAnalyzer(
        backend_model=args.model_name,
        device=args.device,
        min_words=0,
    )
    metadata = analyzer.backend.metadata()
    tokenizer = analyzer.backend.pipeline.tokenizer
    max_length = int(tokenizer.model_max_length)

    previous_columns = [args.id_column, "emotion", "emotion_score"]
    previous_columns = [
        column for column in previous_columns if column in previous.columns
    ]
    frame = source.merge(
        previous[previous_columns],
        on=args.id_column,
        how="left",
        validate="one_to_one",
        suffixes=("", "_original"),
    )

    review_jobs = []
    for row_index, row in frame.iterrows():
        cleaned = analyzer.preprocessor.clean(row[args.text_column])
        windows = build_token_windows(
            tokenizer,
            cleaned,
            max_length=max_length,
            stride=args.stride,
        )
        total_content_tokens = (
            windows[-1]["start_token"] + windows[-1]["content_token_count"]
            if windows
            else 0
        )
        if args.only_over_limit and total_content_tokens <= (
            max_length - tokenizer.num_special_tokens_to_add(pair=False)
        ):
            continue
        review_jobs.append((row_index, total_content_tokens, windows))

    segment_texts = [
        window["text"]
        for _, _, windows in review_jobs
        for window in windows
    ]
    predictions = analyzer.backend.predict_batch(
        segment_texts,
        batch_size=args.batch_size,
    )

    segment_rows = []
    review_rows = []
    prediction_offset = 0
    for row_index, total_tokens, windows in review_jobs:
        row = frame.loc[row_index]
        window_predictions = predictions[
            prediction_offset : prediction_offset + len(windows)
        ]
        prediction_offset += len(windows)
        if any(
            prediction.get("status") != "analyzed"
            for prediction in window_predictions
        ):
            raise RuntimeError(
                f"segment inference failed for {args.id_column}="
                f"{row[args.id_column]}"
            )

        aggregate = aggregate_segment_scores(
            window_predictions,
            [window["aggregation_weight"] for window in windows],
            threshold=args.emotion_threshold,
        )
        review_result = {
            args.id_column: row[args.id_column],
            "country_code": row.get("country_code"),
            "language": row.get("language"),
            "date_created": row.get("date_created"),
            "original_emotion": row.get("emotion"),
            "original_emotion_score": row.get("emotion_score"),
            "total_content_tokens": total_tokens,
            "segment_count": len(windows),
            **aggregate,
        }
        review_result["mean_emotion_changed"] = (
            review_result["segmented_mean_emotion"]
            != review_result["original_emotion"]
        )
        review_result["max_emotion_changed"] = (
            review_result["segmented_max_emotion"]
            != review_result["original_emotion"]
        )
        review_rows.append(review_result)

        for window, prediction in zip(windows, window_predictions):
            segment_row = {
                args.id_column: row[args.id_column],
                "segment_index": window["segment_index"],
                "start_token": window["start_token"],
                "content_token_count": window["content_token_count"],
                "aggregation_weight": window["aggregation_weight"],
                "emotion": prediction["label"],
                "emotion_score": prediction["score"],
            }
            segment_row.update(
                {
                    f"emotion_{label}_score": score
                    for label, score in prediction["scores"].items()
                }
            )
            segment_rows.append(segment_row)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        args.output_dir
        or f"outputs/segmented_{timestamp}_{Path(args.input_path).stem[:25]}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    review_df = pd.DataFrame(review_rows)
    segment_df = pd.DataFrame(segment_rows)
    review_df.to_csv(output_dir / "segmented_review_comparison.csv", index=False)
    segment_df.to_csv(output_dir / "segment_predictions.csv", index=False)

    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "input_file": args.input_path,
        "previous_results": args.previous_results,
        "model_metadata": metadata,
        "max_length": max_length,
        "stride": args.stride,
        "emotion_threshold": args.emotion_threshold,
        "only_over_limit": args.only_over_limit,
        "selected_reviews": len(review_df),
        "analyzed_segments": len(segment_df),
        "mean_emotion_changed": (
            int(review_df["mean_emotion_changed"].sum())
            if len(review_df)
            else 0
        ),
        "max_emotion_changed": (
            int(review_df["max_emotion_changed"].sum())
            if len(review_df)
            else 0
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + os.linesep,
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
