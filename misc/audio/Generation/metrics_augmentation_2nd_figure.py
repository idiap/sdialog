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

    # Drop rows where 'Score' is NaN, which can happen with missing data
    plot_data.dropna(subset=['Score'], inplace=True)

    # Custom function to split metric names
    def split_metric_name(metric_name):
        if '|' in metric_name:
            parts = metric_name.split('|', 1)
            return parts[0], parts[1]
        else:
            # For metrics without a separator, use the name as type and a default for condition
            return metric_name, 'Overall'

    # Apply the custom function to create 'Metric Type' and 'Condition'
    plot_data[['Metric Type', 'Condition']] = plot_data['Metric'].apply(lambda x: pd.Series(split_metric_name(x)))

    # Create a figure with subplots for each metric type
    metric_types = plot_data['Metric Type'].unique()

    if len(metric_types) == 0:
        print("No data available to plot after handling missing values.")
        return

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(metric_types),
        figsize=(5 * len(metric_types), 6),  # Increased height for better label visibility
        constrained_layout=True
    )
    fig.suptitle('Metric Score Distributions', fontsize=16)

    # Ensure axes is always an array
    if len(metric_types) == 1:
        axes = [axes]

    for i, metric_type in enumerate(metric_types):
        ax = axes[i]
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
    # python3 metrics_augmentation_2nd_figure.py --csv_path="./metrics_augmentation_results_stats.csv"
    parser = argparse.ArgumentParser(description='Analyze metrics from a CSV file.')
    parser.add_argument(
        '--csv_path',
        type=str,
        required=True,
        help='Path to the input CSV file.',
        default="/lustre/fsn1/projects/rech/rtl/uaj63yz/JSALT2025/sdialog/misc/audio/Generation/metrics_augmentation_results_stats.csv"
    )
    args = parser.parse_args()
    analyze_metrics(args.csv_path)
    # python3 metrics_augmentation_2nd_figure.py --csv="/lustre/fsn1/projects/rech/rtl/uaj63yz/JSALT2025/sdialog/misc/audio/Generation/metrics_augmentation_results_stats.csv"
