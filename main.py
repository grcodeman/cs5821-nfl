import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

# load datasets
team_stats = pd.read_csv('data/nfl_team_stats.csv')
game_data = pd.read_csv('data/nfl_spreadspoke_data.csv')
team_mapping = pd.read_csv('data/nfl_teams.csv')

total_features = [
    "home_def_st_td", "home_drives", "home_first_downs", "home_first_downs_from_passing",
    "home_first_downs_from_penalty", "home_first_downs_from_rushing", "home_fourth_down_att",
    "home_fourth_down_comp", "home_fumbles", "home_interceptions", "home_pass_att", "home_pass_comp",
    "home_pass_yards", "home_pen_num", "home_pen_yards", "home_plays", "home_possession",
    "home_redzone_att", "home_redzone_comp", "home_rush_att", "home_rush_yards", "home_sacks_num",
    "home_sacks_yards", "home_score", "home_third_down_att", "home_third_down_comp", "home_yards",
    "home_opp_def_st_td", "home_opp_drives", "home_opp_first_downs", "home_opp_first_downs_from_passing",
    "home_opp_first_downs_from_penalty", "home_opp_first_downs_from_rushing", "home_opp_fourth_down_att",
    "home_opp_fourth_down_comp", "home_opp_fumbles", "home_opp_interceptions", "home_opp_pass_att",
    "home_opp_pass_comp", "home_opp_pass_yards", "home_opp_pen_num", "home_opp_pen_yards", "home_opp_plays",
    "home_opp_possession", "home_opp_redzone_att", "home_opp_redzone_comp", "home_opp_rush_att",
    "home_opp_rush_yards", "home_opp_sacks_num", "home_opp_sacks_yards", "home_opp_score",
    "home_opp_third_down_att", "home_opp_third_down_comp", "home_opp_yards", "away_def_st_td",
    "away_drives", "away_first_downs", "away_first_downs_from_passing", "away_first_downs_from_penalty",
    "away_first_downs_from_rushing", "away_fourth_down_att", "away_fourth_down_comp", "away_fumbles",
    "away_interceptions", "away_pass_att", "away_pass_comp", "away_pass_yards", "away_pen_num",
    "away_pen_yards", "away_plays", "away_possession", "away_redzone_att", "away_redzone_comp",
    "away_rush_att", "away_rush_yards", "away_sacks_num", "away_sacks_yards", "away_score",
    "away_third_down_att", "away_third_down_comp", "away_yards", "away_opp_def_st_td", "away_opp_drives",
    "away_opp_first_downs", "away_opp_first_downs_from_passing", "away_opp_first_downs_from_penalty",
    "away_opp_first_downs_from_rushing", "away_opp_fourth_down_att", "away_opp_fourth_down_comp",
    "away_opp_fumbles", "away_opp_interceptions", "away_opp_pass_att", "away_opp_pass_comp",
    "away_opp_pass_yards", "away_opp_pen_num", "away_opp_pen_yards", "away_opp_plays", "away_opp_possession",
    "away_opp_redzone_att", "away_opp_redzone_comp", "away_opp_rush_att", "away_opp_rush_yards",
    "away_opp_sacks_num", "away_opp_sacks_yards", "away_opp_score", "away_opp_third_down_att",
    "away_opp_third_down_comp", "away_opp_yards", "over_under_line", "spread_favorite",
]


# team name mapping process
team_name_map = team_mapping.set_index('team_name')['team_name_short'].to_dict()
def standardize_team_name(name):
    return team_name_map.get(name, name)

# prepare team stats for each given game
home_stats = team_stats.add_prefix('home_')
away_stats = team_stats.add_prefix('away_')

# load rf_model
with open("pickles/rf_model.pkl", "rb") as file:
    rf_model = pickle.load(file)

def predict_total_points(home_team, away_team, over_under_line, spread_favorite, season):
    # make sure team names are correct
    home_team_std = standardize_team_name(home_team)
    away_team_std = standardize_team_name(away_team)
    
    # team stats for season
    home_team_stats = team_stats[(team_stats['season'] == season) & (team_stats['team'] == home_team_std)].add_prefix('home_')
    away_team_stats = team_stats[(team_stats['season'] == season) & (team_stats['team'] == away_team_std)].add_prefix('away_')
    
    # make sure team has stats
    if home_team_stats.empty or away_team_stats.empty:
        print("Team stats for the given season are not available.")
        return None
    
    # stat and game data
    input_data = pd.concat([home_team_stats.reset_index(drop=True), away_team_stats.reset_index(drop=True)], axis=1)
    input_data['over_under_line'] = over_under_line
    input_data['spread_favorite'] = spread_favorite
    
    non_numeric_cols = ['home_team', 'away_team', 'home_season', 'away_season']
    input_data = input_data.drop(columns=non_numeric_cols, errors='ignore')
    
    # clean up any missing cols and reorder
    missing_cols = set(total_features) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    
    input_data = input_data[total_features]
    
    # run data into model
    predicted_total = rf_model.predict(input_data)[0]
    return predicted_total

def main():

    # get input from user
    team_home_input = input("\nHome Team: ")
    team_away_input = input("\nAway Team: ")
    game_ou_input = input("\nO/U: ")
    game_spread = input("\nSpread: ")
    season = 2023

    prediction = predict_total_points(team_home_input,team_away_input,game_ou_input,game_spread,season)
    print("\n"+prediction)

    return 0    

if __name__ == '__main__':
    main()