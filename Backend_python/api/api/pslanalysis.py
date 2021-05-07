# Commented out IPython magic to ensure Python compatibility.
import numpy as np # Numerical analysis
import pandas as pd # Data Processing
import matplotlib.pyplot as plt  # Visualization
# %matplotlib inline
import seaborn as sns  # Modern Visualization
import os

def psl_analysis():
    """<b>Reading the dataset</b>"""

    match= pd.read_csv("matches.csv")

    """<b>Getting the Basic Information about Data</b>"""

    match.shape

    """<b>Observation:</b> 
        Got dimension(rows and columns) of the dataset using method shape. The match variable has 636 rows and 18 columns. 
    """

    match.columns

    match.head()

    match.info()

    """<b>Observation:</b>
        There are missing value in the column umpire3.
    """

    match.season.unique()

    """<b>Observation:</b>
        The dataset has 10 seasons of IPL starting from 2008 to 2017.
    """

    match.id.max()

    """<b>Observation:</b>
        There are 636 IPL Matches we got in our dataset.
    """

    match.iloc[match['win_by_runs'].idxmax()]

    match.iloc[match['win_by_runs'].idxmax()]['winner']

    """<b>Observation:</b>
    <ul>
        <li>idxmax will return the id of the maximum value and iloc that takes an index value and return the row.</li>
        <li>From the dataset, in the Season 2017 the team who won was Mumbai Indians and win by runs was 146(which is highest among all seasons).</li>  
       </ul> 
    """

    match.iloc[match['win_by_wickets'].idxmax()]

    match.iloc[match['win_by_wickets'].idxmax()]['winner']

    """<b>Observation:</b>
    <ul>
        <li>idxmax will return the id of the maximum value and iloc that takes an index value and return the row.</li>
        <li>From the dataset, in the Season 2017 the team who won was Kolkata Knight Riders and win by wickets was 10(which is highest among all seasons).</li>  
       </ul> 
    """

    #sns.countplot(x="season", data = match)
    #plt.show()

    match.season.value_counts()

    """<b>Observation:</b>
    <ul>
    <li>In the Season 2013, there was most number of matches(i.e.76) played.</li> 
    </ul>
    """

    data= match.winner.value_counts()
    #sns.barplot(y=data.index, x=data, orient="h")

    match.winner.value_counts()

    """<b>Observation:</b>
    <ul>
    <li>Among all the seasons from 2008 to 2017, Mumbai Indians has won maximum number of times.</li> 
    </ul>
    """

    top_players= match.player_of_match.value_counts()[:5]
    #sns.barplot(x=top_players.index, y=top_players, orient='v')
    #plt.show()

    """<b>Observation:</b>
    <ul>
    <li>Chris Gayle is the Player who has got maximum number of player of the month over the season from 2008-2017.</li>
    </ul>
    """

    Toss= match['toss_winner'] == match['winner']
    Toss.groupby(Toss).size()

    #sns.countplot(match["toss_winner"] == match["winner"])
    #plt.show()

    """<b>Observation:</b>
    <ul>
        <li>The team who won Toss also won the match</li>
    </ul>
    """

    champ = match.drop_duplicates(subset=["season"], keep= 'last')[["season","winner"]].reset_index(drop=True)
    byseasonwinnerteam=champ.sort_values(by=["season"])

    topwinnerteam=champ['winner'].value_counts()
    top_players=[top_players.to_dict()]
    byseasonwinnerteam=[byseasonwinnerteam.to_dict()]
    topwinnerteam=[topwinnerteam.to_dict()]

    return top_players, byseasonwinnerteam, topwinnerteam


top_players, byseasonwinnerteam, topwinnerteam=psl_analysis()
print(top_players)
print(byseasonwinnerteam)
print(topwinnerteam)
