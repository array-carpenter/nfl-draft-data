"""
Load and merge PFF data into the player stats pipeline.
Matches PFF players to CFBD players via (player_name, team, year).
"""
import os
import glob
import pandas as pd

PFF_DATA_DIR = "pff_data"

# PFF team_name -> CFBD team name
PFF_TEAM_MAP = {
    "AIR FORCE": "Air Force",
    "AKRON": "Akron",
    "ALABAMA": "Alabama",
    "APP STATE": "App State",
    "ARIZONA": "Arizona",
    "ARIZONA ST": "Arizona State",
    "ARK STATE": "Arkansas State",
    "ARKANSAS": "Arkansas",
    "ARMY": "Army",
    "AUBURN": "Auburn",
    "BALL ST": "Ball State",
    "BAYLOR": "Baylor",
    "BOISE ST": "Boise State",
    "BOSTON COL": "Boston College",
    "BOWL GREEN": "Bowling Green",
    "BUFFALO": "Buffalo",
    "BYU": "BYU",
    "C MICHIGAN": "Central Michigan",
    "CAL": "California",
    "CHARLOTTE": "Charlotte",
    "CINCINNATI": "Cincinnati",
    "CLEMSON": "Clemson",
    "COAST CAR": "Coastal Carolina",
    "COLO STATE": "Colorado State",
    "COLORADO": "Colorado",
    "DELAWARE": "Delaware",
    "DOMINION": "Old Dominion",
    "DUKE": "Duke",
    "E CAROLINA": "East Carolina",
    "E MICHIGAN": "Eastern Michigan",
    "FAU": "Florida Atlantic",
    "FIU": "Florida International",
    "FLORIDA": "Florida",
    "FLORIDA ST": "Florida State",
    "FRESNO ST": "Fresno State",
    "GA SOUTHRN": "Georgia Southern",
    "GA STATE": "Georgia State",
    "GA TECH": "Georgia Tech",
    "GEORGIA": "Georgia",
    "HAWAII": "Hawai'i",
    "HOUSTON": "Houston",
    "IDAHO": "Idaho",
    "ILLINOIS": "Illinois",
    "INDIANA": "Indiana",
    "IOWA": "Iowa",
    "IOWA STATE": "Iowa State",
    "JAMES MAD": "James Madison",
    "JVILLE ST": "Jacksonville State",
    "KANSAS": "Kansas",
    "KANSAS ST": "Kansas State",
    "KENNESAW": "Kennesaw State",
    "KENT STATE": "Kent State",
    "KENTUCKY": "Kentucky",
    "LA LAFAYET": "Louisiana",
    "LA MONROE": "UL Monroe",
    "LA TECH": "Louisiana Tech",
    "LIBERTY": "Liberty",
    "LOUISVILLE": "Louisville",
    "LSU": "LSU",
    "MARSHALL": "Marshall",
    "MARYLAND": "Maryland",
    "MEMPHIS": "Memphis",
    "MIAMI FL": "Miami",
    "MIAMI OH": "Miami (OH)",
    "MICH STATE": "Michigan State",
    "MICHIGAN": "Michigan",
    "MIDDLE TN": "Middle Tennessee",
    "MINNESOTA": "Minnesota",
    "MISS STATE": "Mississippi State",
    "MISSOURI": "Missouri",
    "MO STATE": "Missouri State",
    "N CAROLINA": "North Carolina",
    "N ILLINOIS": "Northern Illinois",
    "N TEXAS": "North Texas",
    "NAVY": "Navy",
    "NC STATE": "NC State",
    "NEBRASKA": "Nebraska",
    "NEVADA": "Nevada",
    "NEW MEX ST": "New Mexico State",
    "NEW MEXICO": "New Mexico",
    "NOTRE DAME": "Notre Dame",
    "NWESTERN": "Northwestern",
    "OHIO": "Ohio",
    "OHIO STATE": "Ohio State",
    "OKLA STATE": "Oklahoma State",
    "OKLAHOMA": "Oklahoma",
    "OLE MISS": "Ole Miss",
    "OREGON": "Oregon",
    "OREGON ST": "Oregon State",
    "PENN STATE": "Penn State",
    "PITTSBURGH": "Pittsburgh",
    "PURDUE": "Purdue",
    "RICE": "Rice",
    "RUTGERS": "Rutgers",
    "S ALABAMA": "South Alabama",
    "S CAROLINA": "South Carolina",
    "S DIEGO ST": "San Diego State",
    "S JOSE ST": "San José State",
    "SM HOUSTON": "Sam Houston",
    "SMU": "SMU",
    "SO MISS": "Southern Miss",
    "STANFORD": "Stanford",
    "SYRACUSE": "Syracuse",
    "TCU": "TCU",
    "TEMPLE": "Temple",
    "TENNESSEE": "Tennessee",
    "TEXAS": "Texas",
    "TEXAS A&M": "Texas A&M",
    "TEXAS ST": "Texas State",
    "TEXAS TECH": "Texas Tech",
    "TOLEDO": "Toledo",
    "TROY": "Troy",
    "TULANE": "Tulane",
    "TULSA": "Tulsa",
    "UAB": "UAB",
    "UCF": "UCF",
    "UCLA": "UCLA",
    "UCONN": "UConn",
    "UMASS": "Massachusetts",
    "UNLV": "UNLV",
    "USC": "USC",
    "USF": "South Florida",
    "UTAH": "Utah",
    "UTAH ST": "Utah State",
    "UTEP": "UTEP",
    "UTSA": "UTSA",
    "VA TECH": "Virginia Tech",
    "VANDERBILT": "Vanderbilt",
    "VIRGINIA": "Virginia",
    "W GEORGIA": "West Georgia",
    "W KENTUCKY": "Western Kentucky",
    "W MICHIGAN": "Western Michigan",
    "W VIRGINIA": "West Virginia",
    "WAKE": "Wake Forest",
    "WASH STATE": "Washington State",
    "WASHINGTON": "Washington",
    "WISCONSIN": "Wisconsin",
    "WYOMING": "Wyoming",
}


def _normalize_name(name) -> str:
    """Normalize player name for matching: remove periods, standardize suffixes."""
    import re
    if not isinstance(name, str):
        return ""
    name = name.strip()
    name = name.replace(".", "")
    name = re.sub(r"\s+(Jr|Sr|III|IV|II|V)$", r" \1", name)
    return name


def load_pff_passing() -> pd.DataFrame:
    """Load all PFF passing summary CSVs and return a combined DataFrame
    with team names mapped to CFBD format and year extracted from filename."""
    files = sorted(glob.glob(os.path.join(PFF_DATA_DIR, "*_passing_summary.csv")))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        basename = os.path.basename(f)
        year = basename.split("_")[0]
        df = pd.read_csv(f)
        df["year"] = year
        df["team"] = df["team_name"].map(PFF_TEAM_MAP)
        unmapped = df[df["team"].isna()]["team_name"].unique()
        if len(unmapped) > 0:
            print(f"PFF loader: unmapped teams in {basename}: {list(unmapped)}")
            df["team"] = df["team"].fillna(df["team_name"])
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.rename(columns={"player": "player_name"}, inplace=True)
    return combined


def merge_pff_passing(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Merge PFF passing stats into the main stats DataFrame.
    Matches on player name + team + year."""
    pff = load_pff_passing()
    if pff.empty:
        return stats_df

    pff["year"] = pff["year"].astype(str)
    stats_df["year"] = stats_df["year"].astype(str)

    # PFF grade/advanced columns to bring in
    pff_stat_cols = [
        "grades_offense", "grades_pass", "grades_run",
        "accuracy_percent", "avg_depth_of_target", "avg_time_to_throw",
        "btt_rate", "twp_rate", "drop_rate",
        "pressure_to_sack_rate", "sack_percent",
    ]
    pff_subset = pff[["player_name", "team", "year"] + pff_stat_cols].copy()

    # Prefix PFF columns to avoid collisions
    rename_map = {col: f"pff_{col}" for col in pff_stat_cols}
    pff_subset.rename(columns=rename_map, inplace=True)
    prefixed_cols = [f"pff_{col}" for col in pff_stat_cols]

    # First pass: exact name match
    pff_subset["_join_name"] = pff_subset["player_name"]
    stats_df["_join_name"] = stats_df["player"]
    merged = pd.merge(
        stats_df, pff_subset,
        left_on=["_join_name", "team", "year"],
        right_on=["_join_name", "team", "year"],
        how="left"
    )

    # Second pass: normalized name match for unmatched rows
    unmatched_mask = merged["pff_grades_pass"].isna()
    if unmatched_mask.any():
        pff_subset["_norm_name"] = pff_subset["player_name"].apply(_normalize_name)
        merged.loc[unmatched_mask, "_norm_name"] = merged.loc[unmatched_mask, "player"].apply(_normalize_name)

        unmatched_rows = merged.loc[unmatched_mask, ["_norm_name", "team", "year"]].copy()
        pff_norm = pff_subset[["_norm_name", "team", "year"] + prefixed_cols].copy()
        fallback = pd.merge(unmatched_rows, pff_norm, on=["_norm_name", "team", "year"], how="left")

        for col in prefixed_cols:
            merged.loc[unmatched_mask, col] = fallback[col].values

        pff_subset.drop(columns=["_norm_name"], inplace=True)

    merged.drop(columns=["_join_name", "player_name", "_norm_name"], errors="ignore", inplace=True)

    matched = merged[merged["pff_grades_pass"].notna()].shape[0]
    total_pff = len(pff)
    print(f"PFF merge: {matched} player-seasons matched out of {total_pff} PFF records")

    return merged
