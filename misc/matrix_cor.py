import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Ensures no GUI is used

# Load data
data = pd.read_csv("data/nfl_team_stats.csv")

# Select the input features for correlation analysis
features = data[[
        "def_st_td", "drives", "first_downs", "first_downs_from_passing", "first_downs_from_penalty", 
        "first_downs_from_rushing", "fourth_down_att", "fourth_down_comp", "fumbles", "interceptions", 
        "pass_att", "pass_comp", "pass_yards", "pen_num", "pen_yards", "plays", 
        "redzone_att", "redzone_comp", "rush_att", "rush_yards", "sacks_num", "sacks_yards", 
        "score", "third_down_att", "third_down_comp", "yards"
]]



# Calculate the correlation matrix
correlation_matrix = features.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,  # Show correlation values
    cmap='coolwarm',  # Use a diverging color palette
    fmt=".1f",  # Format correlation coefficients to 2 decimal places
    linewidths=0.5,  # Add gridlines for clarity
    cbar_kws={'label': 'Correlation Coefficient'}
)

# Titles and labels
plt.title('Correlation Matrix of Input Features', fontsize=18, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)

# Save the heatmap
correlation_heatmap_path = r"c:\Users\acisb\Desktop\project\feature_correlation_heatmap.png"
plt.savefig(correlation_heatmap_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Feature correlation heatmap saved to: {correlation_heatmap_path}")