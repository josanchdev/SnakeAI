import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

PLOTS_DIR = '../plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Consistent color palette: blue for No Penalty, orange for -0.01, green for -0.05
PALETTE = {
    'None': '#1f77b4',        # blue
    '-0.01': '#ff7f0e',       # orange
    '-0.05': '#2ca02c'        # green
}

def load_data():
    """Load and combine penalty/no-penalty CSVs, add label column."""
    no_penalty_path = '../logs/training_log_no_step_penalty.csv'
    penalty_001_path = '../logs/training_log_step_penalty_-0.01.csv'
    penalty_005_path = '../logs/training_log_step_penalty_-0.05.csv'
    for path in [no_penalty_path, penalty_001_path, penalty_005_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    no_penalty = pd.read_csv(no_penalty_path)
    penalty_001 = pd.read_csv(penalty_001_path)
    penalty_005 = pd.read_csv(penalty_005_path)
    no_penalty['Penalty'] = 'None'
    penalty_001['Penalty'] = '-0.01'
    penalty_005['Penalty'] = '-0.05'
    df = pd.concat([no_penalty, penalty_001, penalty_005], ignore_index=True)
    # Add FruitEaten and FruitPerStep columns by reverse engineering
    def calc_fruit_eaten(row):
        # Set reward structure per penalty
        if row['Penalty'] == 'None':
            reward_fruit = 5
            reward_step = 0
        elif row['Penalty'] == '-0.01':
            reward_fruit = 5
            reward_step = -0.01
        elif row['Penalty'] == '-0.05':
            reward_fruit = 5
            reward_step = -0.05
        else:
            reward_fruit = 5
            reward_step = 0
        # Death penalty depends on steps
        reward_death = -15 if row['Steps'] == 100 else -10
        fruit_eaten = (row['Reward'] - row['Steps'] * reward_step - reward_death) / reward_fruit
        # Clamp to >=0 and round to nearest int (should be integer)
        return max(0, int(round(fruit_eaten)))
    df['FruitEaten'] = df.apply(calc_fruit_eaten, axis=1)
    df['FruitPerStep'] = df['FruitEaten'] / df['Steps'].replace(0, np.nan)
    return df

def plot_total_reward(df):
    plt.figure(figsize=(10,5))
    sns.lineplot(data=df, x='GlobalEpisode', y='Reward', hue='Penalty', palette=PALETTE, alpha=0.8)
    plt.xlabel('Global Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward: No Penalty vs Step Penalty (-0.01, -0.05)')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/reward_comparison.png')
    plt.close()

def plot_smoothed_reward(df, window=100):
    plt.figure(figsize=(10,5))
    for label, data in df.groupby('Penalty'):
        data_sorted = data.sort_values('GlobalEpisode')
        smoothed = data_sorted['Reward'].rolling(window, min_periods=1).mean()
        sns.lineplot(
            x=data_sorted['GlobalEpisode'],
            y=smoothed,
            label=f"Penalty: {label}",
            color=PALETTE.get(label, None)
        )
    plt.xlabel('Global Episode')
    plt.ylabel(f'Smoothed Reward (window={window})')
    plt.title('Smoothed Reward: No Penalty vs Step Penalty (-0.01, -0.05)')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/reward_smoothed.png')
    plt.close()

def plot_steps(df):
    plt.figure(figsize=(10,5))
    sns.lineplot(data=df, x='GlobalEpisode', y='Steps', hue='Penalty', palette=PALETTE, alpha=0.8)
    plt.xlabel('Global Episode')
    plt.ylabel('Steps per Episode')
    plt.title('Steps: No Penalty vs Step Penalty (-0.01, -0.05)')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/steps_comparison.png')
    plt.close()

def plot_loss(df, window=100):
    plt.figure(figsize=(10,5))
    for label, data in df.groupby('Penalty'):
        data_sorted = data.sort_values('GlobalEpisode')
        smoothed = data_sorted['AvgLoss'].rolling(window, min_periods=1).mean()
        sns.lineplot(
            x=data_sorted['GlobalEpisode'],
            y=smoothed,
            label=f"Penalty: {label}",
            color=PALETTE.get(label, None)
        )
    plt.xlabel('Global Episode')
    plt.ylabel(f'Smoothed AvgLoss (window={window})')
    plt.title('Smoothed Loss: No Penalty vs Step Penalty (-0.01, -0.05)')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/loss_smoothed.png')
    plt.close()

def summary_table(df):
    summary = df.groupby('Penalty').agg({
        'Reward': ['mean', 'median', 'max', 'min'],
        'Steps': ['mean', 'median', 'max', 'min'],
        'AvgLoss': ['mean', 'median', 'max', 'min'],
        'FruitPerStep': ['mean']
    })
    print('Statistical Summary Table:')
    print(summary)
    summary.to_csv(f'{PLOTS_DIR}/summary_table.csv')

def plot_boxplots(df):
    # Boxplot for reward distribution in last 10% of episodes
    max_ep = df['GlobalEpisode'].max()
    cutoff = max_ep * 0.9
    plt.figure(figsize=(8,5))
    sns.boxplot(
        data=df[df['GlobalEpisode'] > cutoff],
        x='Penalty',
        y='Reward',
        hue='Penalty',
        palette=PALETTE,
        dodge=False
    )
    plt.title('Reward Distribution (Last 10% Episodes)')
    plt.ylabel('Reward')
    plt.legend([],[], frameon=False)  # Hide redundant legend
    plt.savefig(f'{PLOTS_DIR}/reward_boxplot.png')
    plt.close()

def plot_violinplots(df):
    # Violin plot for steps distribution in last 10% of episodes
    max_ep = df['GlobalEpisode'].max()
    cutoff = max_ep * 0.9
    plt.figure(figsize=(8,5))
    sns.violinplot(
        data=df[df['GlobalEpisode'] > cutoff],
        x='Penalty',
        y='Steps',
        hue='Penalty',
        palette=PALETTE,
        dodge=False
    )
    plt.title('Steps Distribution (Last 10% Episodes)')
    plt.legend([],[], frameon=False)  # Hide redundant legend
    plt.savefig(f'{PLOTS_DIR}/steps_violinplot.png')
    plt.close()

def plot_reward_breakdown(df):
    # Only if reward breakdown columns exist
    reward_cols = [col for col in df.columns if col.startswith('Reward_') and col != 'Reward']
    if not reward_cols:
        print('No reward breakdown columns found.')
        return
    plt.figure(figsize=(10,6))
    for col in reward_cols:
        sns.lineplot(x=df['GlobalEpisode'], y=df[col], label=col)
    plt.xlabel('Global Episode')
    plt.ylabel('Reward Component')
    plt.title('Reward Component Breakdown')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/reward_breakdown.png')
    plt.close()

def main():
    df = load_data()
    plot_total_reward(df)
    plot_smoothed_reward(df)
    plot_steps(df)
    plot_loss(df)
    summary_table(df)
    plot_boxplots(df)
    plot_violinplots(df)
    plot_reward_breakdown(df)
    # Bar plot for FruitPerStep_mean
    summary = df.groupby('Penalty')['FruitPerStep'].mean().reset_index()
    plt.figure(figsize=(7,5))
    sns.barplot(data=summary, x='Penalty', y='FruitPerStep', hue='Penalty', palette=PALETTE, legend=False)
    plt.ylabel('FruitPerStep (mean)')
    plt.title('Mean FruitPerStep by Penalty')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/fruitperstep_bar.png')
    plt.close()
    # Optional: scatter plot Steps_mean vs FruitPerStep_mean
    steps_summary = df.groupby('Penalty').agg({'Steps':'mean','FruitPerStep':'mean'}).reset_index()
    plt.figure(figsize=(7,5))
    sns.scatterplot(data=steps_summary, x='Steps', y='FruitPerStep', hue='Penalty', palette=PALETTE, s=120)
    for i, row in steps_summary.iterrows():
        plt.text(row['Steps'], row['FruitPerStep'], row['Penalty'], fontsize=12, ha='right')
    plt.xlabel('Steps (mean)')
    plt.ylabel('FruitPerStep (mean)')
    plt.title('Steps vs FruitPerStep (mean) by Penalty')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/steps_vs_fruitperstep_scatter.png')
    plt.close()
    print(f"All plots and tables saved to {PLOTS_DIR}/")

if __name__ == '__main__':
    main()