
from shiny import ui, render, App, reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pylab 
import scipy.stats as stats
from scipy.stats import norm, kstest, poisson
from shiny.types import ImgData
from IPython.display import display
from pathlib import Path


url = Path(__file__).resolve().parent / "Player_IDs.csv"
player_ids = pd.read_csv(url)    


stat_dict = {i:j for i,j in zip(['PTS', 'REB', 'AST', 'PTS + REB + AST', '3PM', 'PTS + REB', 'PTS + AST', 'REB + AST', 'BLK', 'STL', 'BLK + STL', 'TOV', 'FTM'], ['PTS', 'REB','AST','PRA','3PM','PTR','PTA','RBA', 'BLK','STL', 'BKS', 'TOV', 'FTM'])}


def projection_nba(stat, projection, player):

    id_dict = dict(zip(player_ids.Player, player_ids.Id))

    html = pyodide.http.pyfetch('https://www.basketball-reference.com/players/c/{}/gamelog/2023'.format(id_dict[player]))
    soup = BeautifulSoup(html, features="html.parser")

    ### Player Name
    title = soup.findAll('title')[0].getText().replace('Game Log | Basketball-Reference.com', "")

    ### Columns Name (PTS, AST, etc.)
    # headers = [th.getText() for th in soup.findAll('th')[:30]]
    headers = ['G', 'Date', 'Age', 'Tm', 'H/A', 'Opp', 'Diff', 'GS', 'MP', 'FG', 'FGA',
       'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB',
       'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'GmSc', '+/-']


    ### Player Stats (per game)
    rows = soup.findAll('tr')[1:]
    player_stats = [[td.getText() for td in rows[i].findAll('td')] for i in range(len(rows))]

    ## Convert to DataFrame
    stats_df = pd.DataFrame(player_stats, columns = headers)

    ## Dropping DNP Games
    stats_df = stats_df.dropna()
    stats_home = stats_df.loc[stats_df['H/A'] == '']
    stats_away = stats_df.loc[stats_df['H/A'] == '@']

    proj_dict = {
    'PTS': stats_df['PTS'].astype(int),
    'REB': stats_df['TRB'].astype(int),
    'AST': stats_df['AST'].astype(int),
    'PRA': stats_df['PTS'].astype(int) + stats_df['TRB'].astype(int) + stats_df['AST'].astype(int),
    '3PM': stats_df['3P'].astype(int),
    'PTR': stats_df['PTS'].astype(int) + stats_df['TRB'].astype(int),
    'PTA': stats_df['PTS'].astype(int) + stats_df['AST'].astype(int),
    'RBA': stats_df['TRB'].astype(int) + stats_df['AST'].astype(int),
    'BLK': stats_df['BLK'].astype(int),
    'STL': stats_df['STL'].astype(int),
    'BKS': stats_df['BLK'].astype(int) + stats_df['STL'].astype(int),
    'TOV': stats_df['TOV'].astype(int),
    'FTM': stats_df['FT'].astype(int)
    }

    proj_home = {
    'PTS': stats_home['PTS'].astype(int),
    'REB': stats_home['TRB'].astype(int),
    'AST': stats_home['AST'].astype(int),
    'PRA': stats_home['PTS'].astype(int) + stats_home['TRB'].astype(int) + stats_home['AST'].astype(int),
    '3PM': stats_home['3P'].astype(int),
    'PTR': stats_home['PTS'].astype(int) + stats_home['TRB'].astype(int),
    'PTA': stats_home['PTS'].astype(int) + stats_home['AST'].astype(int),
    'RBA': stats_home['TRB'].astype(int) + stats_home['AST'].astype(int),
    'BLK': stats_home['BLK'].astype(int),
    'STL': stats_home['STL'].astype(int),
    'BKS': stats_home['BLK'].astype(int) + stats_home['STL'].astype(int),
    'TOV': stats_home['TOV'].astype(int),
    'FTM': stats_home['FT'].astype(int)
    }

    proj_away = {
    'PTS': stats_away['PTS'].astype(int),
    'REB': stats_away['TRB'].astype(int),
    'AST': stats_away['AST'].astype(int),
    'PRA': stats_away['PTS'].astype(int) + stats_away['TRB'].astype(int) + stats_away['AST'].astype(int),
    '3PM': stats_away['3P'].astype(int),
    'PTR': stats_away['PTS'].astype(int) + stats_away['TRB'].astype(int),
    'PTA': stats_away['PTS'].astype(int) + stats_away['AST'].astype(int),
    'RBA': stats_away['TRB'].astype(int) + stats_away['AST'].astype(int),
    'BLK': stats_away['BLK'].astype(int),
    'STL': stats_away['STL'].astype(int),
    'BKS': stats_away['BLK'].astype(int) + stats_away['STL'].astype(int),
    'TOV': stats_away['TOV'].astype(int),
    'FTM': stats_away['FT'].astype(int)
    }


    target = np.mean(proj_dict[stat])

    target_10 = np.mean(proj_dict[stat].tail(10))

    target_home = np.mean(proj_home[stat])
    target_away = np.mean(proj_away[stat])


    all_over = 1-poisson.cdf(projection, target)
    all_under = poisson.cdf(projection, target)

    home_over = 1-poisson.cdf(projection, target_home)
    home_under = poisson.cdf(projection, target_home)

    away_over = 1-poisson.cdf(projection, target_away)
    away_under = poisson.cdf(projection, target_away)

    l10_over = 1-poisson.cdf(projection, target_10)
    l10_under = poisson.cdf(projection, target_10)

    return target, target_home, target_away, target_10, all_over, all_under, home_over, home_under, away_over, away_under, l10_over, l10_under, stats_df[['G', 'H/A', 'Opp', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'TRB','AST', 'STL', 'BLK', 'TOV','PTS']]


app_ui = ui.page_fluid(
    ui.tags.title("Tony's Player Projections"),
    
   # ui.page_navbar(title = "Tony's Player Projections", bg = '#0062cc', inverse = True, footer = "DDD"),

    ui.output_image("image", height = '100px', inline = True),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_selectize("players", "Choose Player:", player_ids.Player.to_list(), multiple = False),
            ui.input_radio_buttons("target", "Choose Stat:", ['PTS', 'REB', 'AST', 'PTS + REB + AST', '3PM', 'PTS + REB', 'PTS + AST', 'REB + AST', 'BLK', 'STL', 'BLK + STL', 'TOV', 'FTM']),
            ui.input_text("projection", "Enter PrizePick Projected:", "0.0"),
            ui.input_slider("gamelog_n", "Choose Past Number of Games to Display Gamelog:", min = 5, max = 20, value = 10),
            ui.input_action_button("go", "Go!", class_="btn-success"), width = 3
            ),
        ui.panel_main(
            ui.output_text_verbatim("test"),
            ui.output_table("gamelogs"), width = 8
            )
                 )
)



def server(input, output, session):

    @output
    @render.image
    def image():
        from pathlib import Path
        dir = Path(__file__).resolve().parent
        img: ImgData = {"src": str(dir / "preap2.png"), 'inline': True, 'width': "100%", 'height':"75%"}
        return img
    
    @output
    @render.text
    @reactive.event(input.go, ignore_none=True)
    def test():
        target, target_home, target_away, target_10, all_over, all_under, home_over, home_under, away_over, away_under, l10_over, l10_under, gamelog = projection_nba(stat_dict[input.target()], float(input.projection()), input.players())
        @output
        @render.table
        @reactive.event(input.go, ignore_none=True)
        def gamelogs():
            return gamelog.iloc[-input.gamelog_n():]
        return "Projection for {}: {} \nSeason Average: {} \nHOME Average: {} \nAWAY Average: {} \nLast 10 Average: {}\n\nSeason Probability \nOver: {} \nUnder: {} \n\nHOME Probability \nOver: {} \nUnder: {} \n\nAWAY Probability \nOver: {} \nUnder: {} \n\nLast 10 Probability \nOver: {} \nUnder: {} \n".format(input.target(), input.projection(), target, target_home, target_away, target_10, all_over, all_under, home_over, home_under, away_over, away_under, l10_over, l10_under)

# This is a shiny.App object. It must be named `app`.
app = App(app_ui, server)