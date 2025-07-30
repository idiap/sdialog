import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns


def analyze_metrics(csv_path):

    # Load the data from the CSV file
    df = pd.read_csv(csv_path, index_col=0)

    # Calculate the average of each metric column
    averages = df.mean().to_frame(name='Average Value')
    averages.index.name = 'Metric'

    # Generate and print the LaTeX table
    latex_table = averages.to_latex(
        caption="Average Metric Results",
        label="tab:avg_results",
        column_format="lr",
        float_format="%.3f"
    )
    print("--- LaTeX Table ---")
    print(latex_table)

    # Prepare data for plotting
    # Melt the DataFrame to a long format for seaborn
    plot_data = df.melt(var_name='Metric', value_name='Score')
    plot_data[['Metric Type', 'Condition']] = plot_data['Metric'].str.split('|', expand=True)

    # Create a figure with subplots for each metric type
    metric_types = plot_data['Metric Type'].unique()
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(metric_types),
        figsize=(5 * len(metric_types), 5),
        constrained_layout=True
    )
    fig.suptitle('Metric Score Distributions', fontsize=16)

    for i, metric_type in enumerate(metric_types):
        ax = axes[i] if len(metric_types) > 1 else axes
        subset = plot_data[plot_data['Metric Type'] == metric_type]

        sns.boxplot(x='Condition', y='Score', data=subset, ax=ax)
        ax.set_title(metric_type.upper())
        ax.set_xlabel('Condition')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)

    # Save the figure
    figure_path = 'metrics_boxplots.png'
    plt.savefig(figure_path, dpi=300)
    print(f"\nFigure saved to {figure_path}")
    # plt.show() # Uncomment to display the plot interactively


if __name__ == '__main__':
    # python3 metrics_augmentation_figure.py --csv_path="./metrics_augmentation_results_stats.csv"
    parser = argparse.ArgumentParser(description='Analyze metrics from a CSV file.')
    parser.add_argument('--csv_path', type=str, help='Path to the input CSV file.')
    args = parser.parse_args()
    analyze_metrics(args.csv_path)
