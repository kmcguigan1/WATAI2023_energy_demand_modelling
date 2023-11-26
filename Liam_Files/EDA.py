import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import json 

"""
Perform Exploratory Data Analysis (EDA) on a DataFrame.

Parameters:
- df (pd.DataFrame): The data to analyze.
- df_name (str): Name used for saving outputs.

Outputs:
- Saves histograms, boxplots, correlation matrix, pairplots, and time series plots as PNG files.
- Returns a dictionary with file paths to the plots and computed statistical measures.

Functions:
- plot_histograms: Histograms for numerical features.
- plot_boxplots: Boxplots for numerical features.
- plot_correlation_matrix: Heatmap of correlations between features.
- plot_pairplot: Pairwise relationships for numerical features.
- plot_time_series: Time series plot if a 'date' column exists.
- compute_skewness_kurtosis: Skewness and kurtosis for each feature.
- compute_quantile_stats: 25th, 50th, 75th percentiles.

Notes:
- Creates a directory for output if it doesn't exist.
- Assumes 'date' column is in datetime format for time series plots.
- Adjusts layout for single or multiple numerical features.
"""

def perform_eda(df, df_name):
    eda_dir = f'./{df_name}'
    if not os.path.exists(eda_dir):
        os.makedirs(eda_dir)

    def plot_histograms():
        num_features = df.select_dtypes(include=[np.number]).shape[1]
        if num_features > 1:
            fig, axes = plt.subplots(num_features, figsize=(10, num_features*3))
            axes = axes.flatten()  # This will make it always an array, even if it's only one subplot
        else:
            fig, axes = plt.subplots(figsize=(10, 6))
            axes = [axes]  # Put the single axes object in a list so it can be iterated over

        for i, col in enumerate(df.select_dtypes(include=[np.number])):
            sns.histplot(df[col], bins=20, ax=axes[i])
            axes[i].set_title(col, fontsize=10)
        fig.tight_layout()
        plt.savefig(f'{eda_dir}/{df_name}_histograms.png')
        plt.close()

    def plot_boxplots():
        num_features = df.select_dtypes(include=[np.number]).shape[1]
        if num_features > 1:
            fig, axes = plt.subplots(num_features, figsize=(10, num_features*3))
            axes = axes.flatten() 
        else:
            fig, axes = plt.subplots(figsize=(10, 6))
            axes = [axes]  

        for i, col in enumerate(df.select_dtypes(include=[np.number])):
            sns.boxplot(x=df[col], ax=axes[i])
            axes[i].set_title(col, fontsize=10)
        fig.tight_layout()
        plt.savefig(f'{eda_dir}/{df_name}_boxplots.png')
        plt.close()

    def plot_correlation_matrix():
        plt.figure(figsize=(20, 20))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.title(f'Correlation Matrix for {df_name}')
        plt.savefig(f'{eda_dir}/{df_name}_correlation_matrix.png')
        plt.close()

    def plot_pairplot():
        sns.pairplot(df.select_dtypes(include=[np.number]))
        plt.savefig(f'{eda_dir}/{df_name}_pairplot.png')
        plt.close()

    def plot_time_series():
        if 'date' in df.columns:
            # Convert the 'date' column to datetime if it hasn't been converted yet
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            column_to_plot = df.columns[0]

            plt.figure(figsize=(14, 7))
            plt.plot(df.index, df[column_to_plot], marker='o', linestyle='-')
            plt.title(f'Time Series for {df_name}')
            plt.xlabel('Date')
            plt.ylabel(column_to_plot)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{eda_dir}/{df_name}_time_series.png')
            plt.close()

    def compute_skewness_kurtosis():
        return df.skew(), df.kurtosis()

    def compute_quantile_stats():
        return df.quantile([0.25, 0.5, 0.75])

    # Running EDA methods
    plot_histograms()
    plot_boxplots()
    if df.select_dtypes(include=[np.number]).shape[1] > 1:
        plot_correlation_matrix()
        plot_pairplot()
    if 'date' in df.columns:
        plot_time_series()
    skewness, kurtosis = compute_skewness_kurtosis()
    quantiles = compute_quantile_stats()

    # Return paths to the saved plots and computed statistics
    return {
        'histograms': f'{eda_dir}/{df_name}_histograms.png',
        'boxplots': f'{eda_dir}/{df_name}_boxplots.png',
        'correlation_matrix': f'{eda_dir}/{df_name}_correlation_matrix.png' if df.select_dtypes(include=[np.number]).shape[1] > 1 else 'N/A',
        'pairplot': f'{eda_dir}/{df_name}_pairplot.png' if df.select_dtypes(include=[np.number]).shape[1] > 1 else 'N/A',
        'time_series': f'{eda_dir}/{df_name}_time_series.png' if 'date' in df.columns else 'N/A',
        'skewness': skewness.to_dict(),
        'kurtosis': kurtosis.to_dict(),
        'quantiles': quantiles.to_dict()
    }

# Load data
housing_data = pd.read_csv('../housing_data.csv')
energy_data = pd.read_csv('../daily_energy_consumption.csv')

housing_data_dirname = "HousingData"
energy_data_dirname = "EnergyConsumption"

# Perform EDA on housing data
housing_eda_results = perform_eda(housing_data, housing_data_dirname)

# Perform EDA on energy consumption data
energy_eda_results = perform_eda(energy_data, energy_data_dirname)

# Save the results to a JSON file
def save_results_to_json(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

# Save housing data EDA to JSON
housing_results_json_path = f"./{housing_data_dirname}/HousingData_EDA_Results.json"
save_results_to_json(housing_eda_results, housing_results_json_path)

# Save the energy data EDA to JSON
energy_results_json_path = f"./{energy_data_dirname}/EnergyConsumption_EDA_Results.json"
save_results_to_json(energy_eda_results, energy_results_json_path)

# Print out the file paths
print(f"Housing EDA results saved to: {housing_results_json_path}")
print(f"Energy EDA results saved to: {energy_results_json_path}")