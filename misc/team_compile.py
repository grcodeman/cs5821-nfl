import pandas as pd

df = pd.read_csv('data/nfl_game_stats.csv')

# sort home and team data into df
away_columns = ['season', 'week', 'date', 'away'] + [col for col in df.columns if '_away' in col]
away_df = df[away_columns].copy()
away_df.columns = away_df.columns.str.replace('_away', '')
away_df.rename(columns={'away': 'team'}, inplace=True)
away_df['home_or_away'] = 'away'

home_columns = ['season', 'week', 'date', 'home'] + [col for col in df.columns if '_home' in col]
home_df = df[home_columns].copy()
home_df.columns = home_df.columns.str.replace('_home', '')
home_df.rename(columns={'home': 'team'}, inplace=True)
home_df['home_or_away'] = 'home'

# merge df
team_stats = pd.concat([away_df, home_df], ignore_index=True)

non_numeric_cols = ['season', 'week', 'date', 'team', 'home_or_away']
numeric_cols = team_stats.columns.difference(non_numeric_cols)
team_stats[numeric_cols] = team_stats[numeric_cols].apply(pd.to_numeric, errors='coerce')

# average
team_season_avg = team_stats.groupby(['season', 'team'])[numeric_cols].mean().reset_index()

# save to csv
team_season_avg.to_csv('data/nfl_team_stats.csv', index=False)
