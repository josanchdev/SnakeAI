import pandas as pd

csv_path = '../plots/summary_table.csv'
df = pd.read_csv(csv_path, header=[0,1], index_col=0)
df.columns = ['_'.join(col).strip() for col in df.columns.values]
df.reset_index(inplace=True)

# Clean up Penalty column: convert to string, strip, and replace 'nan' with 'None'
df['Penalty'] = df['Penalty'].astype(str).str.strip().replace({'nan': 'None'})

print("Unique Penalty values:", df['Penalty'].unique())

cols = [
    'Penalty',
    'Reward_mean', 'Reward_median', 'Reward_max',
    'Steps_mean', 'Steps_median',
    'FruitPerStep_mean'
]
df_short = df[cols].round(4)

markdown_table = df_short.to_markdown(index=False)
print(markdown_table)

with open('../plots/summary_table.md', 'w') as f:
    f.write(markdown_table)