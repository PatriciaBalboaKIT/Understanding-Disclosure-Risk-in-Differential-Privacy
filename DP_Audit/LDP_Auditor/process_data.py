import pandas as pd

# Load the CSV file
df = pd.read_csv("your_file.csv")

# Group by protocol and epsilon, then compute mean and std of eps_emp
summary = df.groupby(['protocol', 'epsilon'])['eps_emp'].agg(['mean', 'std']).reset_index()

# Rename columns for clarity
summary.columns = ['protocol', 'epsilon', 'avg_eps_emp', 'std_eps_emp']

# Sort for nicer output
summary = summary.sort_values(by=['protocol', 'epsilon'])

# Display the result
print(summary)

# Optionally, save to CSV
# summary.to_csv("protocol_epsilon_stats.csv", index=False)
