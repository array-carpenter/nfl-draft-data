import numpy as np
import pandas as pd
from scipy.stats import rankdata
from config import POSITION_BASELINES, COMBINE_STATS_PATH
from sklearn.neighbors import NearestNeighbors

class DataProcessor:
    def __init__(self, stats_df: pd.DataFrame):
        self.stats_df = stats_df.copy()
        self.processed_df = None
        self.percentile_df = None
        self.comparison_players = None
        self.radar_data = None
        self.valid_metrics = None
        self.player_position = None
        self.non_round_metrics = ["Height (in)", "Hand Size (in)", "Arm Length (in)", "40 Yard","3Cone","Shuttle"]

    def process(self, input_player: str, player_year=2025):
        combine_columns = [
            "Height (in)", "Weight (lbs)", "Hand Size (in)", "Arm Length (in)",
            "40 Yard", "Bench Press", "Vert Leap (in)",
            "Broad Jump (in)", "Shuttle", "3Cone", "POS_GP"
        ]
        try:
            combine_df = pd.read_csv(COMBINE_STATS_PATH, usecols=["athlete_id"] + combine_columns)
        except Exception as e:
            print("Error reading combine data:", e)
            combine_df = pd.DataFrame()

        if not combine_df.empty:
            df = pd.merge(self.stats_df, combine_df, on="athlete_id", how="left")
        else:
            df = self.stats_df.copy()

        df = df.drop_duplicates(subset=["player", "year", "team", "athlete_id"])

        if input_player not in df["player"].unique():
            raise ValueError(f"Input player {input_player} not found in data.")

        # Try to pick the specified year for this player
        df_player_year = df[(df["player"] == input_player) & (df["year"] == player_year)]
        if df_player_year.empty:
            # If empty, fallback to the most recent year with stats
            fallback_year = df.loc[df["player"] == input_player, "year"].max()
            print(f"No data found for {input_player} in year={player_year}. Using fallback year={fallback_year} instead.")
            df_player_year = df[(df["player"] == input_player) & (df["year"] == fallback_year)]
        if df_player_year.empty:
            raise ValueError(f"No stats found at all for player={input_player}.")

        target_id = df_player_year["athlete_id"].iloc[0]

        df = df[~((df["player"] == input_player) & (df["athlete_id"] != target_id))]

        player_position = df.loc[df["player"] == input_player, "POS_GP"].values[0]
        if player_position in ["FS", "SS"]:
            position_key = "S"
            df = df[df["POS_GP"].isin(["FS", "SS"])]
        else:
            position_key = player_position
            df = df[df["POS_GP"] == player_position]

        baseline_metrics = POSITION_BASELINES.get(position_key, [])
        valid_metrics = [m for m in baseline_metrics if m in df.columns]

        for stat in combine_columns:
            if stat in df.columns:
                player_val = df.loc[df["player"] == input_player, stat]
                if not player_val.empty and pd.notnull(player_val.iloc[0]) and stat not in valid_metrics:
                    valid_metrics.append(stat)

        if "POS_GP" in valid_metrics:
            valid_metrics.remove("POS_GP")

        for stat in combine_columns:
            if stat in df.columns:
                df[stat] = pd.to_numeric(df[stat], errors="coerce")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        agg_funcs = {col: "sum" for col in numeric_cols if col not in combine_columns}
        if "interceptions_avg" in agg_funcs:
            agg_funcs["interceptions_avg"] = "mean"
        if "passing_ypa" in agg_funcs:
            agg_funcs["passing_ypa"] = "mean"
        for stat in combine_columns:
            if stat in df.columns:
                agg_funcs[stat] = "first"

        df_sum = df.groupby("player").agg(agg_funcs).reset_index()

        if "passing_pct" in df.columns:
            passing_pct_avg = df.groupby("player")["passing_pct"].mean().reset_index()
            df_sum = df_sum.merge(passing_pct_avg, on="player", how="left", suffixes=("", "_mean"))
            df_sum["passing_pct"] = df_sum["passing_pct_mean"] * 100
            df_sum.drop(columns=["passing_pct_mean"], inplace=True)

        if "receiving_rec" in df_sum.columns and "receiving_yds" in df_sum.columns:
            df_sum["receiving_ypr"] = df_sum.apply(
                lambda row: row["receiving_yds"] / row["receiving_rec"] if row["receiving_rec"] > 0 else 0,
                axis=1
            )

        if all(col in df.columns for col in ["rushing_yds", "rushing_car", "rushing_ypc"]):
            rushing_ypc_avg = df.groupby("player").apply(
                lambda x: x["rushing_yds"].sum() / x["rushing_car"].sum() if x["rushing_car"].sum() > 0 else 0
            ).reset_index(name="rushing_ypc")
            df_sum.drop(columns=["rushing_ypc"], errors="ignore", inplace=True)
            df_sum = df_sum.merge(rushing_ypc_avg, on="player", how="left")

            df_sum[valid_metrics] = df_sum[valid_metrics].apply(pd.to_numeric, errors="coerce")
            df_sum[valid_metrics] = df_sum[valid_metrics].fillna(df_sum[valid_metrics].mean())

        self.processed_df = df_sum
        self.valid_metrics = valid_metrics

        reverse_metrics = {"passing_int", "40 Yard", "3Cone", "Shuttle", "Fumbles"}
        self.percentile_df = df_sum.copy()
        for metric in valid_metrics:
            percentile_values = rankdata(df_sum[metric], method="average") / len(df_sum) * 100
            if metric in reverse_metrics:
                self.percentile_df[metric] = 100 - percentile_values
            else:
                self.percentile_df[metric] = percentile_values

        knn = NearestNeighbors(n_neighbors=4, metric='euclidean')
        knn.fit(self.percentile_df[valid_metrics].values)

        input_index = self.percentile_df[self.percentile_df["player"] == input_player].index
        if len(input_index) == 0:
            raise ValueError("No aggregated row found for the input player.")
        input_index = input_index[0]

        distances, indices = knn.kneighbors(
            self.percentile_df.loc[input_index, valid_metrics].values.reshape(1, -1)
        )
        neighbor_indices = list(indices[0])
        if input_index in neighbor_indices:
            neighbor_indices.remove(input_index)
        top3_players = self.percentile_df.loc[neighbor_indices, "player"].tolist()

        self.comparison_players = [input_player] + top3_players
        self.radar_data = [
            self.percentile_df[self.percentile_df["player"] == p][valid_metrics].iloc[0].values
            for p in self.comparison_players
        ]
        self.player_position = position_key
