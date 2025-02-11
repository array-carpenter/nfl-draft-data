# data_processor.py
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from config import POSITION_BASELINES

class DataProcessor:
    def __init__(self, stats_df: pd.DataFrame):
        """
        Initialize with the merged stats dataframe.
        """
        self.stats_df = stats_df.copy()
        self.processed_df = None
        self.percentile_df = None
        self.comparison_players = None

    def process(self, input_player: str):
        """
        Process the data for a given input player:
          - Filter to players with the same position as input_player.
          - Aggregate stats across seasons.
          - Correct percentage and derived metrics.
          - Compute percentile ranks.
          - Calculate similarity using Euclidean distance.
        """
        # Get the player's position
        player_position = self.stats_df.loc[self.stats_df["player"] == input_player, "position"].values[0]
        
        # Get the baseline metrics for the player's position
        valid_metrics = [m for m in POSITION_BASELINES.get(player_position, []) if m in self.stats_df.columns]
        if not valid_metrics:
            raise ValueError(f"No valid metrics found for position: {player_position}")

        # Filter players by position
        df = self.stats_df[self.stats_df["position"] == player_position]

        # Sum player stats across seasons (for non-percentage metrics)
        df_sum = df.groupby("player").sum().reset_index()

        # Correct percentage and derived metrics
        if "passing_pct" in df.columns:
            passing_pct_avg = df.groupby("player")["passing_pct"].mean().reset_index()
            df_sum = df_sum.merge(passing_pct_avg, on="player", how="left")
            df_sum["passing_pct"] = df_sum["passing_pct_y"] * 100  # Convert to percentage
            df_sum.drop(columns=["passing_pct_y"], inplace=True)

        # Correct Yards per Carry (rushing_ypc = rushing_yds / rushing_car)
        if "rushing_ypc" in df.columns and "rushing_yds" in df.columns and "rushing_car" in df.columns:
            rushing_ypc_avg = df.groupby("player").apply(
                lambda x: x["rushing_yds"].sum() / x["rushing_car"].sum() if x["rushing_car"].sum() > 0 else 0
            ).reset_index(name="rushing_ypc")
            
            df_sum.drop(columns=["rushing_ypc"], errors="ignore", inplace=True)  # Remove incorrect summed column
            df_sum = df_sum.merge(rushing_ypc_avg, on="player", how="left")

        # Ensure valid metrics are numeric
        df_sum[valid_metrics] = df_sum[valid_metrics].apply(pd.to_numeric, errors="coerce").fillna(0)

        self.processed_df = df_sum
        self.valid_metrics = valid_metrics

        # Compute percentile ranks
        self.percentile_df = df_sum.copy()
        for metric in valid_metrics:
            self.percentile_df[metric] = rankdata(df_sum[metric], method="average") / len(df_sum)

        # Extract the player's data in percentile form
        player_percentiles = self.percentile_df[self.percentile_df["player"] == input_player][valid_metrics].iloc[0]

        # Compute similarity (Euclidean distance) to other players
        self.percentile_df["similarity"] = self.percentile_df[valid_metrics].apply(
            lambda row: np.linalg.norm(row - player_percentiles), axis=1
        )

        # Get the top 3 closest players (excluding the input player)
        top3 = self.percentile_df[self.percentile_df["player"] != input_player].nsmallest(3, "similarity")
        self.comparison_players = [input_player] + top3["player"].tolist()

        # Store percentile values for plotting
        self.radar_data = [
            self.percentile_df[self.percentile_df["player"] == p][valid_metrics].iloc[0].values
            for p in self.comparison_players
        ]
        # Store player's position (for use in plot title, etc.)
        self.player_position = player_position
