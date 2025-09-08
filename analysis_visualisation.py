import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PositionResult:
    """Position evaluation result for visualization"""
    fen: str
    model_eval: float
    stockfish_eval: Optional[float]
    category: str = "unknown"
    source: str = "manual"
    game_phase: str = "unknown"
    material_balance: int = 0
    piece_count: int = 32


class EvaluationVisualizer:
    """Create visualizations for NNUE evaluation analysis"""

    def __init__(self, style: str = "whitegrid"):
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10

    def correlation_plot(self, results: List[PositionResult],
                         output_file: Optional[str] = None) -> None:
        """Create correlation plot between model and Stockfish evaluations"""
        # Filter valid results
        valid_results = [r for r in results if r.stockfish_eval is not None]

        if len(valid_results) < 2:
            print("Insufficient data for correlation plot")
            return

        model_evals = [r.model_eval for r in valid_results]
        sf_evals = [r.stockfish_eval for r in valid_results]

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Main scatter plot
        ax1.scatter(sf_evals, model_evals, alpha=0.6, s=30)

        # Calculate correlation
        correlation, p_value = stats.pearsonr(model_evals, sf_evals)

        # Add regression line
        z = np.polyfit(sf_evals, model_evals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(sf_evals), max(sf_evals), 100)
        ax1.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        # Perfect correlation line
        min_val = min(min(sf_evals), min(model_evals))
        max_val = max(max(sf_evals), max(model_evals))
        ax1.plot([min_val, max_val], [min_val, max_val], "g--", alpha=0.5, linewidth=1)

        ax1.set_xlabel("Stockfish Evaluation (cp)")
        ax1.set_ylabel("Model Evaluation (cp)")
        ax1.set_title(f"Model vs Stockfish Correlation\\nr = {correlation:.4f}, p = {p_value:.2e}")
        ax1.grid(True, alpha=0.3)

        # Error distribution
        errors = np.array(model_evals) - np.array(sf_evals)
        ax2.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.1f}')
        ax2.axvline(np.median(errors), color='green', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(errors):.1f}')
        ax2.set_xlabel("Error (Model - Stockfish) (cp)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Error Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Residuals plot
        predicted = p(sf_evals)
        residuals = np.array(model_evals) - predicted
        ax3.scatter(predicted, residuals, alpha=0.6)
        ax3.axhline(0, color='red', linestyle='--')
        ax3.set_xlabel("Predicted Model Evaluation (cp)")
        ax3.set_ylabel("Residuals (cp)")
        ax3.set_title("Residuals Plot")
        ax3.grid(True, alpha=0.3)

        # Q-Q plot for normality check
        stats.probplot(errors, dist="norm", plot=ax4)
        ax4.set_title("Q-Q Plot (Error Normality Check)")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Correlation plot saved to {output_file}")
        else:
            plt.show()

        plt.close()

    def category_analysis_plot(self, results: List[PositionResult],
                               output_file: Optional[str] = None) -> None:
        """Create plots showing performance by category"""
        # Create DataFrame
        data = []
        for result in results:
            if result.stockfish_eval is not None:
                data.append({
                    'category': result.category,
                    'game_phase': result.game_phase,
                    'model_eval': result.model_eval,
                    'stockfish_eval': result.stockfish_eval,
                    'error': result.model_eval - result.stockfish_eval,
                    'abs_error': abs(result.model_eval - result.stockfish_eval),
                    'piece_count': result.piece_count,
                    'material_balance': result.material_balance
                })

        df = pd.DataFrame(data)

        if df.empty:
            print("No valid data for category analysis")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Error by category boxplot
        if 'game_phase' in df.columns:
            sns.boxplot(data=df, x='game_phase', y='abs_error', ax=ax1)
            ax1.set_title("Absolute Error by Game Phase")
            ax1.set_ylabel("Absolute Error (cp)")
            ax1.tick_params(axis='x', rotation=45)

        # Correlation by category
        correlations = []
        categories = df['game_phase'].unique()

        for category in categories:
            cat_data = df[df['game_phase'] == category]
            if len(cat_data) > 1:
                corr, _ = stats.pearsonr(cat_data['model_eval'], cat_data['stockfish_eval'])
                correlations.append({'category': category, 'correlation': corr})

        if correlations:
            corr_df = pd.DataFrame(correlations)
            ax2.bar(corr_df['category'], corr_df['correlation'])
            ax2.set_title("Correlation by Game Phase")
            ax2.set_ylabel("Pearson Correlation")
            ax2.tick_params(axis='x', rotation=45)
            ax2.set_ylim(0, 1)

        # Error vs piece count
        ax3.scatter(df['piece_count'], df['abs_error'], alpha=0.6)
        ax3.set_xlabel("Piece Count")
        ax3.set_ylabel("Absolute Error (cp)")
        ax3.set_title("Error vs Piece Count")

        # Error vs material balance
        ax4.scatter(df['material_balance'], df['abs_error'], alpha=0.6)
        ax4.set_xlabel("Material Balance")
        ax4.set_ylabel("Absolute Error (cp)")
        ax4.set_title("Error vs Material Balance")

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Category analysis plot saved to {output_file}")
        else:
            plt.show()

        plt.close()

    def evaluation_distribution_plot(self, results: List[PositionResult],
                                     output_file: Optional[str] = None) -> None:
        """Plot evaluation distributions"""
        valid_results = [r for r in results if r.stockfish_eval is not None]

        if not valid_results:
            print("No valid data for distribution plot")
            return

        model_evals = [r.model_eval for r in valid_results]
        sf_evals = [r.stockfish_eval for r in valid_results]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Model evaluation distribution
        ax1.hist(model_evals, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(np.mean(model_evals), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(model_evals):.1f}')
        ax1.axvline(np.median(model_evals), color='green', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(model_evals):.1f}')
        ax1.set_xlabel("Model Evaluation (cp)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Model Evaluation Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Stockfish evaluation distribution
        ax2.hist(sf_evals, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax2.axvline(np.mean(sf_evals), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(sf_evals):.1f}')
        ax2.axvline(np.median(sf_evals), color='green', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(sf_evals):.1f}')
        ax2.set_xlabel("Stockfish Evaluation (cp)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Stockfish Evaluation Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Comparison (overlaid)
        ax3.hist(model_evals, bins=30, alpha=0.5, color='blue', label='Model', edgecolor='black')
        ax3.hist(sf_evals, bins=30, alpha=0.5, color='orange', label='Stockfish', edgecolor='black')
        ax3.set_xlabel("Evaluation (cp)")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Evaluation Distribution Comparison")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to {output_file}")
        else:
            plt.show()

        plt.close()

    def performance_summary_plot(self, results: List[PositionResult],
                                 output_file: Optional[str] = None) -> None:
        """Create comprehensive performance summary plot"""
        valid_results = [r for r in results if r.stockfish_eval is not None]

        if not valid_results:
            print("No valid data for performance summary")
            return

        model_evals = np.array([r.model_eval for r in valid_results])
        sf_evals = np.array([r.stockfish_eval for r in valid_results])
        errors = model_evals - sf_evals
        abs_errors = np.abs(errors)

        fig = plt.figure(figsize=(20, 12))

        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Main correlation plot
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax1.scatter(sf_evals, model_evals, alpha=0.6, s=20)

        # Regression line
        z = np.polyfit(sf_evals, model_evals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(sf_evals.min(), sf_evals.max(), 100)
        ax1.plot(x_line, p(x_line), "r-", alpha=0.8, linewidth=2)

        # Perfect correlation line
        min_val = min(sf_evals.min(), model_evals.min())
        max_val = max(sf_evals.max(), model_evals.max())
        ax1.plot([min_val, max_val], [min_val, max_val], "g--", alpha=0.5, linewidth=1, label="Perfect correlation")

        correlation, p_value = stats.pearsonr(model_evals, sf_evals)
        ax1.set_xlabel("Stockfish Evaluation (cp)")
        ax1.set_ylabel("Model Evaluation (cp)")
        ax1.set_title(f"Model vs Stockfish Correlation (r = {correlation:.4f})")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Error histogram
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.hist(errors, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', alpha=0.8)
        ax2.set_xlabel("Error (cp)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Error Distribution")
        ax2.grid(True, alpha=0.3)

        # Absolute error histogram
        ax3 = fig.add_subplot(gs[0, 3])
        ax3.hist(abs_errors, bins=20, alpha=0.7, edgecolor='black', color='orange')
        ax3.axvline(np.median(abs_errors), color='red', linestyle='--',
                    label=f'Median: {np.median(abs_errors):.1f}')
        ax3.set_xlabel("Absolute Error (cp)")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Absolute Error Distribution")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Error vs evaluation magnitude
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.scatter(np.abs(sf_evals), abs_errors, alpha=0.6, s=20)
        ax4.set_xlabel("Evaluation Magnitude (cp)")
        ax4.set_ylabel("Absolute Error (cp)")
        ax4.set_title("Error vs Evaluation Magnitude")
        ax4.grid(True, alpha=0.3)

        # Bland-Altman plot
        ax5 = fig.add_subplot(gs[1, 3])
        mean_evals = (model_evals + sf_evals) / 2
        ax5.scatter(mean_evals, errors, alpha=0.6, s=20)
        ax5.axhline(0, color='red', linestyle='--')
        ax5.axhline(np.mean(errors) + 1.96 * np.std(errors), color='orange', linestyle='--', alpha=0.7)
        ax5.axhline(np.mean(errors) - 1.96 * np.std(errors), color='orange', linestyle='--', alpha=0.7)
        ax5.set_xlabel("Mean Evaluation (cp)")
        ax5.set_ylabel("Difference (cp)")
        ax5.set_title("Bland-Altman Plot")
        ax5.grid(True, alpha=0.3)

        # Statistics summary
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')

        stats_text = f"""
        PERFORMANCE STATISTICS

        Sample Size: {len(valid_results)}
        Pearson Correlation: {correlation:.4f} (p = {p_value:.2e})

        Mean Absolute Error: {np.mean(abs_errors):.1f} cp
        Root Mean Square Error: {np.sqrt(np.mean(errors ** 2)):.1f} cp
        Median Absolute Error: {np.median(abs_errors):.1f} cp

        Error Percentiles:
        25th: {np.percentile(abs_errors, 25):.1f} cp
        75th: {np.percentile(abs_errors, 75):.1f} cp
        90th: {np.percentile(abs_errors, 90):.1f} cp
        95th: {np.percentile(abs_errors, 95):.1f} cp

        Model Statistics:
        Mean: {np.mean(model_evals):.1f} cp
        Std: {np.std(model_evals):.1f} cp
        Range: [{model_evals.min():.0f}, {model_evals.max():.0f}] cp

        Stockfish Statistics:
        Mean: {np.mean(sf_evals):.1f} cp
        Std: {np.std(sf_evals):.1f} cp
        Range: [{sf_evals.min():.0f}, {sf_evals.max():.0f}] cp
        """

        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Performance summary saved to {output_file}")
        else:
            plt.show()

        plt.close()


class ReportGenerator:
    """Generate comprehensive analysis reports"""

    def __init__(self, visualizer: EvaluationVisualizer):
        self.visualizer = visualizer

    def generate_full_report(self, results: List[PositionResult],
                             output_dir: str = "analysis_output"):
        """Generate complete analysis report with all visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"Generating comprehensive analysis report in {output_path}")

        # Generate all plots
        self.visualizer.correlation_plot(
            results, output_path / "correlation_analysis.png"
        )

        self.visualizer.category_analysis_plot(
            results, output_path / "category_analysis.png"
        )

        self.visualizer.evaluation_distribution_plot(
            results, output_path / "distribution_analysis.png"
        )

        self.visualizer.performance_summary_plot(
            results, output_path / "performance_summary.png"
        )

        # Generate summary statistics
        self._generate_statistics_report(results, output_path / "statistics_report.json")

        # Generate text summary
        self._generate_text_summary(results, output_path / "summary_report.txt")

        print(f"Full analysis report generated in {output_path}")

    def _generate_statistics_report(self, results: List[PositionResult],
                                    output_file: Path):
        """Generate detailed statistics in JSON format"""
        valid_results = [r for r in results if r.stockfish_eval is not None]

        if not valid_results:
            return

        model_evals = np.array([r.model_eval for r in valid_results])
        sf_evals = np.array([r.stockfish_eval for r in valid_results])
        errors = model_evals - sf_evals
        abs_errors = np.abs(errors)

        # Overall statistics
        correlation, p_value = stats.pearsonr(model_evals, sf_evals)
        spearman_r, spearman_p = stats.spearmanr(model_evals, sf_evals)

        # Category-wise statistics
        categories = {}
        for category in set(r.game_phase for r in valid_results):
            cat_results = [r for r in valid_results if r.game_phase == category]
            if len(cat_results) > 1:
                cat_model = [r.model_eval for r in cat_results]
                cat_sf = [r.stockfish_eval for r in cat_results]
                cat_corr, _ = stats.pearsonr(cat_model, cat_sf)
                cat_mae = np.mean(np.abs(np.array(cat_model) - np.array(cat_sf)))

                categories[category] = {
                    "count": len(cat_results),
                    "correlation": float(cat_corr),
                    "mae": float(cat_mae)
                }

        report = {
            "sample_size": len(valid_results),
            "correlations": {
                "pearson": {"r": float(correlation), "p_value": float(p_value)},
                "spearman": {"r": float(spearman_r), "p_value": float(spearman_p)}
            },
            "error_statistics": {
                "mae": float(np.mean(abs_errors)),
                "rmse": float(np.sqrt(np.mean(errors ** 2))),
                "median_ae": float(np.median(abs_errors)),
                "mean_error": float(np.mean(errors)),
                "std_error": float(np.std(errors)),
                "max_abs_error": float(np.max(abs_errors)),
                "percentiles": {
                    "p25": float(np.percentile(abs_errors, 25)),
                    "p50": float(np.percentile(abs_errors, 50)),
                    "p75": float(np.percentile(abs_errors, 75)),
                    "p90": float(np.percentile(abs_errors, 90)),
                    "p95": float(np.percentile(abs_errors, 95)),
                    "p99": float(np.percentile(abs_errors, 99))
                }
            },
            "model_statistics": {
                "mean": float(np.mean(model_evals)),
                "std": float(np.std(model_evals)),
                "min": float(np.min(model_evals)),
                "max": float(np.max(model_evals)),
                "median": float(np.median(model_evals))
            },
            "stockfish_statistics": {
                "mean": float(np.mean(sf_evals)),
                "std": float(np.std(sf_evals)),
                "min": float(np.min(sf_evals)),
                "max": float(np.max(sf_evals)),
                "median": float(np.median(sf_evals))
            },
            "category_analysis": categories
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

    def _generate_text_summary(self, results: List[PositionResult],
                               output_file: Path):
        """Generate human-readable text summary"""
        valid_results = [r for r in results if r.stockfish_eval is not None]

        if not valid_results:
            return

        model_evals = np.array([r.model_eval for r in valid_results])
        sf_evals = np.array([r.stockfish_eval for r in valid_results])
        errors = model_evals - sf_evals
        abs_errors = np.abs(errors)

        correlation, p_value = stats.pearsonr(model_evals, sf_evals)

        with open(output_file, 'w') as f:
            f.write("NNUE MODEL EVALUATION ANALYSIS REPORT\\n")
            f.write("=" * 50 + "\\n\\n")

            f.write(f"Sample Size: {len(valid_results)} positions\\n")
            f.write(
                f"Success Rate: {len(valid_results)}/{len(results)} ({len(valid_results) / len(results) * 100:.1f}%)\\n\\n")

            f.write("CORRELATION ANALYSIS\\n")
            f.write("-" * 20 + "\\n")
            f.write(f"Pearson Correlation: {correlation:.4f}\\n")
            f.write(f"P-value: {p_value:.2e}\\n")

            if correlation > 0.9:
                f.write("Assessment: Excellent correlation\\n")
            elif correlation > 0.8:
                f.write("Assessment: Good correlation\\n")
            elif correlation > 0.6:
                f.write("Assessment: Moderate correlation\\n")
            else:
                f.write("Assessment: Poor correlation\\n")

            f.write("\\nERROR ANALYSIS\\n")
            f.write("-" * 20 + "\\n")
            f.write(f"Mean Absolute Error: {np.mean(abs_errors):.1f} cp\\n")
            f.write(f"Root Mean Square Error: {np.sqrt(np.mean(errors ** 2)):.1f} cp\\n")
            f.write(f"Median Absolute Error: {np.median(abs_errors):.1f} cp\\n")
            f.write(f"Maximum Absolute Error: {np.max(abs_errors):.1f} cp\\n")

            f.write("\\nERROR DISTRIBUTION\\n")
            f.write("-" * 20 + "\\n")
            f.write(f"25th percentile: {np.percentile(abs_errors, 25):.1f} cp\\n")
            f.write(f"50th percentile: {np.percentile(abs_errors, 50):.1f} cp\\n")
            f.write(f"75th percentile: {np.percentile(abs_errors, 75):.1f} cp\\n")
            f.write(f"90th percentile: {np.percentile(abs_errors, 90):.1f} cp\\n")
            f.write(f"95th percentile: {np.percentile(abs_errors, 95):.1f} cp\\n")


def load_results_from_csv(csv_file: str) -> List[PositionResult]:
    """Load position results from CSV file"""
    results = []

    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        result = PositionResult(
            fen=row['fen'],
            model_eval=float(row['model_eval']),
            stockfish_eval=float(row['stockfish_eval']) if pd.notna(row['stockfish_eval']) else None,
            category=row.get('category', 'unknown'),
            source=row.get('source', 'manual'),
            game_phase=row.get('game_phase', 'unknown'),
            material_balance=int(row.get('material_balance', 0)),
            piece_count=int(row.get('piece_count', 32))
        )
        results.append(result)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize NNUE evaluation analysis")
    parser.add_argument("csv_file", help="CSV file with evaluation results")
    parser.add_argument("--output-dir", default="analysis_output", help="Output directory for plots")
    parser.add_argument("--style", default="whitegrid", help="Plot style")

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.csv_file}")
    results = load_results_from_csv(args.csv_file)

    # Create visualizer and generate report
    visualizer = EvaluationVisualizer(args.style)
    generator = ReportGenerator(visualizer)

    generator.generate_full_report(results, args.output_dir)