import numpy as np
import pandas as pd
from scipy.stats import rankdata
from config import POSITION_BASELINES, COMBINE_STATS_PATH
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

class DataProcessor:
    def __init__(self, stats_df: pd.DataFrame):
        """
        Initialize with the filtered stats dataframe.
        Note: This processor will merge the combine data dynamically.
        """
        self.stats_df = stats_df.copy()
        self.processed_df = None
        self.percentile_df = None
        self.comparison_players = None
        self.radar_data = None
        self.valid_metrics = None
        self.player_position = None
        self.non_round_metrics = ["Height (in)", "Hand Size (in)", "Arm Length (in)", "40 Yard", "3Cone", "Shuttle"]

    def _load_combine_data(self):
        """Load combine data from the specified path."""
        combine_columns = [
            "Height (in)", "Weight (lbs)", "Hand Size (in)", "Arm Length (in)",
            "Wonderlic", "40 Yard", "Bench Press", "Vert Leap (in)",
            "Broad Jump (in)", "Shuttle", "3Cone", "60Yd Shuttle", "POS_GP"
        ]
        try:
            combine_df = pd.read_csv(COMBINE_STATS_PATH, usecols=["athlete_id"] + combine_columns)
            return combine_df
        except Exception as e:
            print("Error reading combine data:", e)
            return pd.DataFrame()

    def _filter_and_aggregate_data(self, df: pd.DataFrame, input_player: str, position_key: str):
        """Filter and aggregate data based on player position."""
        baseline_metrics = POSITION_BASELINES.get(position_key, [])
        valid_metrics = [m for m in baseline_metrics if m in df.columns]

        for stat in self.non_round_metrics:
            if stat in df.columns:
                player_val = df.loc[df["player"] == input_player, stat]
                if not player_val.empty and pd.notnull(player_val.iloc[0]) and stat not in valid_metrics:
                    valid_metrics.append(stat)

        if "POS_GP" in valid_metrics:
            valid_metrics.remove("POS_GP")

        print("Valid metrics being used:", valid_metrics)
        if not valid_metrics:
            raise ValueError(f"No valid metrics found for position: {position_key}")

        # Convert all metrics to numeric
        df[valid_metrics] = df[valid_metrics].apply(pd.to_numeric, errors="coerce").fillna(0)

        # Aggregate data
        agg_funcs = {col: "sum" for col in df.select_dtypes(include="number").columns if col not in self.non_round_metrics}
        agg_funcs.update({col: "first" for col in self.non_round_metrics if col in df.columns})

        df_sum = df.groupby("player").agg(agg_funcs).reset_index()

        # Calculate additional metrics
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

        df_sum[valid_metrics] = df_sum[valid_metrics].apply(pd.to_numeric, errors="coerce").fillna(0)
        df_sum = df_sum.drop(columns=["POS_GP"], errors="ignore")

        return df_sum, valid_metrics

    def _calculate_percentiles(self, df_sum: pd.DataFrame, valid_metrics: list):
        """Calculate percentiles for each metric."""
        reverse_metrics = {"passing_int", "40 Yard", "3Cone", "Shuttle", "Fumbles"}
        percentile_df = df_sum.copy()

        for metric in valid_metrics:
            percentile_values = rankdata(df_sum[metric], method="average") / len(df_sum) * 100
            if metric in reverse_metrics:
                percentile_df[metric] = 100 - percentile_values
            else:
                percentile_df[metric] = percentile_values

        return percentile_df

    def _apply_pca(self, percentile_df: pd.DataFrame, valid_metrics: list):
        """Apply PCA to the percentile data."""
        pca = PCA(n_components=0.95)  # Retain 95% of the variance
        pca_features = pca.fit_transform(percentile_df[valid_metrics])

        # Debug: Print PCA details
        print("\nPCA Components Shape:", pca.components_.shape)
        print("Explained Variance Ratio:", pca.explained_variance_ratio_)
        print("Cumulative Explained Variance:", np.cumsum(pca.explained_variance_ratio_))

        # Print top contributing metrics for each principal component
        for i, component in enumerate(pca.components_):
            print(f"\nPrincipal Component {i + 1}:")
            for metric, weight in zip(valid_metrics, component):
                print(f"{metric}: {weight:.4f}")

        return pca_features, pca

    def _find_similar_players(self, percentile_df: pd.DataFrame, pca_features, input_player: str):
        """Find similar players using KNN on PCA-transformed features."""
        knn = NearestNeighbors(n_neighbors=4, metric='euclidean')
        knn.fit(pca_features)

        input_index = percentile_df[percentile_df["player"] == input_player].index[0]
        distances, indices = knn.kneighbors(pca_features[input_index].reshape(1, -1))

        # Debug: Print KNN results
        print("\nKNN Distances:", distances)
        print("KNN Indices:", indices)

        neighbor_indices = list(indices[0])
        if input_index in neighbor_indices:
            neighbor_indices.remove(input_index)
        top3_players = percentile_df.loc[neighbor_indices, "player"].tolist()

        return [input_player] + top3_players

    def process(self, input_player: str):
        """Process the data for the input player."""
        # Load combine data
        combine_df = self._load_combine_data()
        df = pd.merge(self.stats_df, combine_df, on="athlete_id", how="left") if not combine_df.empty else self.stats_df.copy()

        # Validate input player
        if input_player not in df["player"].unique():
            raise ValueError(f"Input player {input_player} not found in data.")

        # Determine player position
        player_position = df.loc[df["player"] == input_player, "POS_GP"].values[0]
        position_key = "S" if player_position in ["FS", "SS"] else player_position
        df = df[df["POS_GP"].isin(["FS", "SS"])] if player_position in ["FS", "SS"] else df[df["POS_GP"] == player_position]

        # Filter and aggregate data
        df_sum, valid_metrics = self._filter_and_aggregate_data(df, input_player, position_key)
        self.processed_df = df_sum
        self.valid_metrics = valid_metrics

        # Calculate percentiles
        self.percentile_df = self._calculate_percentiles(df_sum, valid_metrics)

        # Apply PCA
        pca_features, pca = self._apply_pca(self.percentile_df, valid_metrics)

        # Find similar players
        self.comparison_players = self._find_similar_players(self.percentile_df, pca_features, input_player)

        # Prepare radar data
        self.radar_data = [
            self.percentile_df[self.percentile_df["player"] == p][valid_metrics].iloc[0].values
            for p in self.comparison_players
        ]

        self.player_position = position_key