# config.py
import matplotlib.font_manager as fm

# Font paths (adjust these paths to match your environment)
FONT_PATH = 'C:/Users/RaymondCarpenter/Documents/GitHub/nfl-draft-data/Roboto_Mono/RobotoMono-VariableFont_wght.ttf'
ITALIC_FONT_PATH = 'C:/Users/RaymondCarpenter/Documents/GitHub/nfl-draft-data/Roboto_Mono/RobotoMono-Italic-VariableFont_wght.ttf'

# Font properties
ROBOTO = fm.FontProperties(fname=FONT_PATH)
ITALIC_ROBOTO = fm.FontProperties(fname=ITALIC_FONT_PATH)

TEAM_COLORS = {
    "UTSA": "#0C2340",
    "Charlotte": "#154734",
    "Kennesaw State": "#FFCC00",
    "Florida International": "#081E3F",
    "Coastal Carolina": "#007C92",
    "Air Force": "#0033A0",
    "Akron": "#041E42",
    "Alabama": "#9E1B32",
    "App State": "#FFCC00",
    "Arizona": "#CC0033",
    "Arizona State": "#8C1D40",
    "Arkansas": "#9D2235",
    "Arkansas State": "#CC092F",
    "Army": "#BFAF80",
    "Auburn": "#0C2340",
    "Ball State": "#BA0C2F",
    "Baylor": "#154733",
    "Boise State": "#0033A0",
    "Boston College": "#98002E",
    "Bowling Green": "#FA4616",
    "BYU": "#002E5D",
    "Buffalo": "#005BBB",
    "California": "#003262",
    "UCF": "#B5895B",
    "Central Michigan": "#6A0032",
    "Cincinnati": "#D50032",
    "Clemson": "#F56600",
    "Colorado": "#CFB87C",
    "Colorado State": "#2A6833",
    "Duke": "#00539B",
    "East Carolina": "#592A8A",
    "Eastern Michigan": "#006633",
    "Florida Atlantic": "#002855",
    "Florida": "#0021A5",
    "Florida State": "#782F40",
    "Fresno State": "#D50032",
    "Georgia Southern": "#041E42",
    "Georgia Tech": "#B3A369",
    "Georgia": "#BA0C2F",
    "Hawai'i": "#024731",
    "Houston": "#C8102E",
    "Illinois": "#E84A27",
    "Indiana": "#990000",
    "Iowa": "#FFCD00",
    "Iowa State": "#C8102E",
    "James Madison": "#450084",
    "Jacksonville State": "#A30D26",
    "Kansas": "#0051BA",
    "Kansas State": "#512888",
    "Kent State": "#FDC82F",
    "Kentucky": "#0033A0",
    "Louisiana": "#D50032",
    "UL Monroe": "#862633",
    "Louisiana Tech": "#0046AD",
    "Liberty": "#A6192E",
    "Louisville": "#AD0000",
    "LSU": "#461D7C",
    "Marshall": "#00A862",
    "Maryland": "#E03A3E",
    "Memphis": "#003087",
    "Miami": "#005030",
    "Miami (OH)": "#BA0C2F",
    "Michigan": "#00274C",
    "Michigan State": "#18453B",
    "Middle Tennessee": "#005DAA",
    "Minnesota": "#7A0019",
    "Missouri": "#F1B82D",
    "Mississippi State": "#660000",
    "Navy": "#00205B",
    "NC State": "#CC0000",
    "Nebraska": "#D00000",
    "Nevada": "#003366",
    "New Mexico": "#D50032",
    "New Mexico State": "#862633",
    "Northern Illinois": "#C8102E",
    "North Texas": "#00853E",
    "Northwestern": "#4E2A84",
    "Notre Dame": "#C99700",
    "Ohio": "#006A4D",
    "Ohio State": "#BB0000",
    "Oklahoma": "#841617",
    "Oklahoma State": "#FF7300",
    "Ole Miss": "#CE1126",
    "Oregon": "#154733",
    "Oregon State": "#DC4405",
    "Penn State": "#041E42",
    "Pittsburgh": "#003594",
    "Purdue": "#C28E0E",
    "Rice": "#00205B",
    "Rutgers": "#CC0033",
    "Sam Houston": "#E35205",
    "San Diego State": "#A6192E",
    "San José State": "#0055A2",
    "SMU": "#C60C30",
    "South Carolina": "#73000A",
    "South Florida": "#006747",
    "Stanford": "#8C1515",
    "Texas State": "#461D7C",
    "Syracuse": "#D44500",
    "TCU": "#4D1979",
    "Temple": "#9E1B32",
    "Tennessee": "#FF8200",
    "Texas": "#BF5700",
    "Texas A&M": "#500000",
    "Texas Tech": "#C8102E",
    "Toledo": "#FFCC00",
    "Troy": "#841617",
    "Tulane": "#008E97",
    "UAB": "#006A4D",
    "UCLA": "#2774AE",
    "UConn": "#041E42",
    "Massachusetts": "#881C1C",
    "North Carolina": "#4B9CD3",
    "UNLV": "#BA0C2F",
    "USC": "#9D2235",
    "Utah": "#CC0000",
    "Utah State": "#041E42",
    "UTEP": "#041E42",
    "Vanderbilt": "#866D4B",
    "Virginia Tech": "#630031",
    "Virginia": "#232D4B",
    "Wake Forest": "#9E7E38",
    "Washington": "#4B2E83",
    "Washington State": "#981E32",
    "Western Kentucky": "#D50032",
    "Western Michigan": "#5A3825",
    "West Virginia": "#EAAA00",
    "Wyoming": "#FFC72C",
    "South Alabama": "#041E42",
    "Georgia State": "#0033A0",
    "Southern Miss": "#FFC72C",
    "Wisconsin": "#C5050C",
    "Old Dominion": "#041E42",
    "Tulsa": "#007AC1",
    "Idaho": "#B3995D",
}

# Position baseline metrics. Add metrics for additional positions as needed.
POSITION_BASELINES = {
    "QB": [
        "passing_att", "passing_pct", "passing_yds", "passing_td",
        "passing_ypa", "passing_int", "rushing_car", "rushing_yds",
        "rushing_td", "fumbles_fum"
    ],
    "RB": [
        "rushing_car", "rushing_yds", "rushing_td", "rushing_ypc",
        "receiving_rec", "receiving_yds", "receiving_td", "fumbles_fum"
    ],
    # Add entries for WR, TE, etc. as needed.
}

# Column rename mapping for display purposes
COLUMN_RENAME_MAP = {
    "passing_att": "Attempts",
    "passing_pct": "Completion %",
    "passing_yds": "Passing Yards",
    "passing_td": "Passing TDs",
    "passing_ypa": "Yards per Attempt",
    "passing_int": "Interceptions",
    "rushing_car": "Rush Attempts",
    "rushing_yds": "Rushing Yards",
    "rushing_td": "Rushing TDs",
    "rushing_ypc": "Yards per Carry",
    "receiving_yds": "Receiving Yards",
    "receiving_td": "Receiving TDs",
    "fumbles_fum": "Fumbles",
    "receiving_rec" : "Receptions"
}

# Other configuration variables (e.g., paths to datasets, logo images)
FILTERED_STATS_PATH = "filtered_player_stats_full.csv"
COMBINE_STATS_PATH = "combine_data_unique_athlete_id_step4.csv"
LOGO_PATH = "1.png"
