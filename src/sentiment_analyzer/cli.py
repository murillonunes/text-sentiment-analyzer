import argparse
import datetime
import json
import os
import sys
import time
import pandas as pd

from sentiment_analyzer.analyzer import SentimentAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Emotion and Sentiment Analyzer CLI for Steam reviews and general text.")
    parser.add_argument("input_path", type=str, help="Path to input CSV or JSON file.")
    parser.add_argument("-o", "--output-path", type=str, help="Path to output CSV file. Defaults to outputs/run_<timestamp>_<filename>/<filename>_analyzed.csv.")
    parser.add_argument("-m", "--model-name", type=str, default="tabularisai/multilingual-emotion-classification",
                        help="Hugging Face model name/path.")
    parser.add_argument("-tc", "--text-column", type=str, default="review_text",
                        help="Column containing text to analyze.")
    parser.add_argument("-vc", "--voted-up-column", type=str, default="voted_up",
                        help="Column containing user recommendation (voted_up).")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size for model inference.")
    parser.add_argument("-d", "--device", type=str, default=None, choices=["cpu", "cuda"],
                        help="Device to use for model inference.")
    parser.add_argument("--skip-report", action="store_true", help="Skip generating a report/plots.")
    parser.add_argument("--report-dir", type=str, default="reports", help="Directory to save report output/plots.")
    parser.add_argument("--min-words", type=int, default=0, help="Minimum number of words required in a review to be analyzed.")
    
    args = parser.parse_args()
    
    # Start timer
    start_time = time.time()
    
    # Check input file existence
    if not os.path.exists(args.input_path):
        print(f"Error: Input file '{args.input_path}' not found.", file=sys.stderr)
        sys.exit(1)
        
    # Generate run directory name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(args.input_path))[0]
    
    # Truncate filename if it exceeds 25 characters to prevent path length issues
    max_filename_len = 25
    truncated_name = base_name
    if len(base_name) > max_filename_len:
        truncated_name = base_name[:max_filename_len].rstrip("_").rstrip("-")
        
    run_dir = f"outputs/run_{timestamp}_{truncated_name}"
    os.makedirs(run_dir, exist_ok=True)
        
    # Read file
    print(f"Loading data from '{args.input_path}'...")
    _, ext = os.path.splitext(args.input_path.lower())
    if ext == ".csv":
        df = pd.read_csv(args.input_path)
    elif ext == ".json":
        df = pd.read_json(args.input_path)
    else:
        # Try CSV, then JSON
        try:
            df = pd.read_csv(args.input_path)
        except Exception:
            try:
                df = pd.read_json(args.input_path)
            except Exception as e:
                print(f"Error: Unsupported file format or unable to read file. {e}", file=sys.stderr)
                sys.exit(1)
                
    if len(df) == 0:
        print("Error: Input file is empty.", file=sys.stderr)
        sys.exit(1)
        
    if args.text_column not in df.columns:
        print(f"Error: Text column '{args.text_column}' not found in the input data. Columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
        
    # Initialize analyzer
    print(f"Initializing SentimentAnalyzer with model '{args.model_name}' and min_words={args.min_words}...")
    analyzer = SentimentAnalyzer(backend_model=args.model_name, device=args.device, min_words=args.min_words)
    
    # Run analysis
    print(f"Starting emotion analysis on {len(df)} rows (batch size: {args.batch_size})...")
    analyzed_df = analyzer.analyze_dataframe(df, text_column=args.text_column, batch_size=args.batch_size)
    
    # Define output path
    output_path = args.output_path
    if not output_path:
        output_path = os.path.join(run_dir, f"{base_name}_analyzed.csv")
        
    # Save output
    print(f"Saving analyzed data to '{output_path}'...")
    analyzed_df.to_csv(output_path, index=False)
    
    # Evaluate agreement and metrics
    metrics = None
    if args.voted_up_column in analyzed_df.columns:
        print("\nEvaluating agreement with user recommendations...")
        metrics = SentimentAnalyzer.evaluate_agreement(
            analyzed_df, 
            emotion_col="emotion", 
            voted_up_col=args.voted_up_column
        )
        if metrics:
            print(f"Total evaluated reviews: {metrics.get('total_count')}")
            print(f"Agreement Rate (Mapped Emotion vs recommendation): {metrics.get('agreement_rate'):.2%}")
            print(f"Positive Recommendation Agreement: {metrics.get('positive_recommendation_agreement'):.2%}")
            print(f"Negative Recommendation Agreement: {metrics.get('negative_recommendation_agreement'):.2%}")
            print("Emotion Distribution:")
            for emo, cnt in metrics.get("emotion_counts", {}).items():
                print(f"  - {emo}: {cnt} ({cnt / metrics.get('total_count'):.2%})")
                
            # Save report
            if not args.skip_report:
                report_dir = args.report_dir
                if report_dir == "reports":
                    report_dir = os.path.join(run_dir, "plots")
                os.makedirs(report_dir, exist_ok=True)
                # Generate report
                generate_visualizations(analyzed_df, args.voted_up_column, report_dir)
        else:
            print("Could not evaluate agreement (insufficient valid data or missing columns).")
            
    # Save summary execution JSON
    execution_time = time.time() - start_time
    summary_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "input_file": args.input_path,
        "model_name": args.model_name,
        "min_words": args.min_words,
        "device": args.device or "auto-detect",
        "batch_size": args.batch_size,
        "execution_time_seconds": round(execution_time, 2)
    }
    if metrics:
        summary_data["metrics"] = metrics
        
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    print(f"Saved execution summary to: {summary_path}")
            
    print("\nProcessing complete!")

def generate_visualizations(df: pd.DataFrame, voted_up_col: str, report_dir: str):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        sns.set_theme(style="whitegrid")
        
        # 1. Emotion Distribution Plot
        plt.figure(figsize=(10, 6))
        # Order by count
        order = df["emotion"].value_counts().index
        sns.countplot(data=df, x="emotion", hue="emotion", order=order, palette="viridis", legend=False)
        plt.title("Distribuição Geral de Emoções Detectadas")
        plt.xlabel("Emoção")
        plt.ylabel("Quantidade")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = os.path.join(report_dir, "emotion_distribution.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")
        
        # 2. Emotion vs Voted Up Plot
        if voted_up_col in df.columns:
            plt.figure(figsize=(12, 6))
            df_plot = df.copy()
            df_plot["recommendation"] = df_plot[voted_up_col].map({True: "Recomendado", False: "Não Recomendado"})
            sns.countplot(data=df_plot, x="recommendation", hue="emotion", palette="tab10")
            plt.title("Emoção Predita vs Recomendação do Usuário")
            plt.xlabel("Recomendação")
            plt.ylabel("Quantidade")
            plt.legend(title="Emoção Predita", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plot_path_vs = os.path.join(report_dir, "emotion_vs_recommendation.png")
            plt.savefig(plot_path_vs)
            plt.close()
            print(f"Saved plot: {plot_path_vs}")
    except Exception as e:
        print(f"Warning: Could not generate visualizations due to error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
