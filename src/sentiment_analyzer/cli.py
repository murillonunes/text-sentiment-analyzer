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

    # Generate HTML Dashboard Report
    if not args.skip_report:
        write_html_report(run_dir, base_name, summary_data, analyzed_df, args.text_column, args.voted_up_column)
            
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

def write_html_report(run_dir, base_name, summary_data, analyzed_df, text_col, voted_col):
    try:
        # Build samples table rows
        sample_rows_html = ""
        preview_df = analyzed_df.head(15)
        for _, row in preview_df.iterrows():
            text_val = str(row.get(text_col, ""))
            text_snippet = text_val[:80] + "..." if len(text_val) > 80 else text_val
            
            voted_val = row.get(voted_col, None)
            if voted_val is True:
                voted_str, voted_class = "Recomendado", "badge-joy"
            elif voted_val is False:
                voted_str, voted_class = "Não Recomendado", "badge-anger"
            else:
                voted_str, voted_class = "N/A", "badge-skipped"
                
            emo = str(row.get("emotion", "skipped_short"))
            emo_clean = emo.replace(" ", "_").replace("-", "_")
            
            badge_map = {
                "joy": "badge-joy", "love": "badge-love", "anger": "badge-anger",
                "sadness": "badge-sadness", "disgust": "badge-disgust", "neutral": "badge-neutral",
                "skipped_short": "badge-skipped"
            }
            emo_class = badge_map.get(emo_clean, "badge-skipped")
            
            score_val = row.get("emotion_score", 0.0)
            score_str = f"{score_val:.1%}" if emo != "skipped_short" else "0.0%"
            
            sample_rows_html += f"""
            <tr>
                <td title="{text_val}">{text_snippet}</td>
                <td><span class="badge {voted_class}">{voted_str}</span></td>
                <td><span class="badge {emo_class}">{emo}</span></td>
                <td><strong>{score_str}</strong></td>
            </tr>
            """
            
        # Build emotion counts table rows
        emotion_rows_html = ""
        metrics = summary_data.get("metrics", {})
        if metrics and "emotion_counts" in metrics:
            total_valid = sum(metrics["emotion_counts"].values())
            for emo, count in metrics["emotion_counts"].items():
                pct = count / total_valid if total_valid > 0 else 0.0
                emo_clean = emo.replace(" ", "_").replace("-", "_")
                badge_map = {
                    "joy": "badge-joy", "love": "badge-love", "anger": "badge-anger",
                    "sadness": "badge-sadness", "disgust": "badge-disgust", "neutral": "badge-neutral",
                    "skipped_short": "badge-skipped"
                }
                emo_class = badge_map.get(emo_clean, "badge-skipped")
                emotion_rows_html += f"""
                <tr>
                    <td><span class="badge {emo_class}">{emo}</span></td>
                    <td>{count}</td>
                    <td><strong>{pct:.1%}</strong></td>
                </tr>
                """
                
        # Fill stats
        agreement = metrics.get("agreement_rate", 0.0)
        agreement_str = f"{agreement:.2%}" if "agreement_rate" in metrics else "N/A"
        
        # Load HTML template content
        html_template = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard de Análise - {base_name}</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #080c14;
            --card-bg: rgba(17, 24, 39, 0.7);
            --border-color: rgba(255, 255, 255, 0.06);
            --text-color: #f3f4f6;
            --text-muted: #9ca3af;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 2.5rem;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1250px;
            margin: 0 auto;
        }}
        
        header {{
            margin-bottom: 2.5rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }}
        
        h1 {{
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(to right, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .timestamp {{
            font-size: 0.95rem;
            color: var(--text-muted);
            background: rgba(255, 255, 255, 0.03);
            padding: 0.4rem 1.2rem;
            border-radius: 99px;
            border: 1px solid var(--border-color);
        }}
        
        .grid-meta {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2.5rem;
        }}
        
        .card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }}
        
        .card-title {{
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }}
        
        .card-value {{
            font-size: 1.6rem;
            font-weight: 700;
        }}
        
        .val-blue {{
            background: linear-gradient(to right, #60a5fa, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .val-green {{
            background: linear-gradient(to right, #34d399, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .val-pink {{
            background: linear-gradient(to right, #f472b6, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .grid-plots {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 2rem;
            margin-bottom: 2.5rem;
        }}
        
        @media (max-width: 768px) {{
            .grid-plots {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .plot-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        
        .plot-title {{
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1.2rem;
            color: #f3f4f6;
            align-self: flex-start;
        }}
        
        .plot-img {{
            max-width: 100%;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.04);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }}
        
        .section-split {{
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 2rem;
            margin-bottom: 2.5rem;
        }}
        
        @media (max-width: 950px) {{
            .section-split {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .table-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 1.5rem;
            overflow-x: auto;
        }}
        
        .section-title {{
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 1rem;
            background: linear-gradient(to right, #ffffff, #9ca3af);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }}
        
        th {{
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-color);
            text-transform: uppercase;
        }}
        
        td {{
            padding: 0.9rem 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            font-size: 0.95rem;
        }}
        
        tr:last-child td {{
            border-bottom: none;
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: capitalize;
        }}
        
        .badge-joy {{ background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.2); }}
        .badge-love {{ background: rgba(244, 114, 182, 0.15); color: #f472b6; border: 1px solid rgba(244, 114, 182, 0.2); }}
        .badge-anger {{ background: rgba(248, 113, 113, 0.15); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.2); }}
        .badge-sadness {{ background: rgba(96, 165, 250, 0.15); color: #60a5fa; border: 1px solid rgba(96, 165, 250, 0.2); }}
        .badge-disgust {{ background: rgba(251, 191, 36, 0.15); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.2); }}
        .badge-neutral {{ background: rgba(156, 163, 175, 0.15); color: #d1d5db; border: 1px solid rgba(156, 163, 175, 0.2); }}
        .badge-skipped {{ background: rgba(255, 255, 255, 0.05); color: #9ca3af; border: 1px solid rgba(255, 255, 255, 0.1); }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>Dashboard de Emoções</h1>
                <p style="color: var(--text-muted); font-size: 0.95rem; margin-top: 0.25rem;">Origem: {summary_data.get('input_file')}</p>
            </div>
            <div class="timestamp">Execução: {summary_data.get('timestamp')[:19].replace('T', ' ')}</div>
        </header>
        
        <div class="grid-meta">
            <div class="card">
                <p class="card-title">Modelo Utilizado</p>
                <p class="card-value" style="font-size: 0.95rem; line-height: 1.3; font-family: monospace; color: #60a5fa; margin-top: 0.25rem;">
                    {summary_data.get('model_name')}
                </p>
            </div>
            <div class="card">
                <p class="card-title">Média de Concordância</p>
                <p class="card-value val-green">{agreement_str}</p>
            </div>
            <div class="card">
                <p class="card-title">Tempo de Processamento</p>
                <p class="card-value val-blue">{summary_data.get('execution_time_seconds')}s</p>
            </div>
            <div class="card">
                <p class="card-title">Avaliações Totais</p>
                <p class="card-value">{len(analyzed_df)}</p>
            </div>
            <div class="card">
                <p class="card-title">Filtro de Palavras</p>
                <p class="card-value val-pink">min_words={summary_data.get('min_words')}</p>
            </div>
        </div>
        
        <div class="grid-plots">
            <div class="plot-card">
                <p class="plot-title">Distribuição Geral de Emoções</p>
                <img src="plots/emotion_distribution.png" class="plot-img" alt="Distribuição de Emoções">
            </div>
            <div class="plot-card">
                <p class="plot-title">Emoção vs Recomendação do Usuário</p>
                <img src="plots/emotion_vs_recommendation.png" class="plot-img" alt="Emoção vs Votos">
            </div>
        </div>
        
        <div class="section-split">
            <div class="table-card">
                <h2 class="section-title">Contagem de Emoções</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Emoção</th>
                            <th>Qtd</th>
                            <th>%</th>
                        </tr>
                    </thead>
                    <tbody>
                        {emotion_rows_html}
                    </tbody>
                </table>
            </div>
            
            <div class="table-card">
                <h2 class="section-title">Amostra de Reviews Analisadas (Top 15)</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Texto da Review</th>
                            <th>Voto Steam</th>
                            <th>Emoção Predita</th>
                            <th>Confiança</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sample_rows_html}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>"""
        
        report_path = os.path.join(run_dir, "report.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_template)
        print(f"Generated HTML report at: {report_path}")
    except Exception as e:
        print(f"Warning: Could not generate HTML report due to error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
