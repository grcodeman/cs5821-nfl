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

# add opp stats from home for away teams
for col in df.columns:
    if col.endswith('_home'):
        new_col = 'opp_' + col.replace('_home', '')
        away_df[new_col] = df[col]

# add opp stats from away for home teams
for col in df.columns:
    if col.endswith('_away'):
        new_col = 'opp_' + col.replace('_away', '')
        home_df[new_col] = df[col]

# merge df
team_stats = pd.concat([away_df, home_df], ignore_index=True)

non_numeric_cols = ['season', 'week', 'date', 'team', 'home_or_away']
numeric_cols = team_stats.columns.difference(non_numeric_cols)
team_stats[numeric_cols] = team_stats[numeric_cols].apply(pd.to_numeric, errors='coerce')

# average
team_season_avg = team_stats.groupby(['season', 'team'])[numeric_cols].mean().reset_index()

# get current cols
cols = team_season_avg.columns.tolist()

# loop through cols to move opp_ prefix stats to the back
for col in cols[:]:
    if col.startswith('opp_'):
        cols.remove(col)
        cols.append(col)
team_season_avg = team_season_avg[cols]

# save to csv
team_season_avg.to_csv('data/nfl_team_stats.csv', index=False)
