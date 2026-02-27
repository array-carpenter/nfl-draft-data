import numpy as np
import pandas as pd
from scipy.stats import rankdata
from config import POSITION_BASELINES, COMBINE_STATS_PATH, EXCLUDE_FROM_KNN
from pff_loader import merge_pff_passing

PPA_DATA_PATH = "qb_ppa_data.csv"
WR_EPA_PATH = "wr_epa_per_rec.csv"
from sklearn.neighbors import NearestNeighbors

class DataProcessor:
    def __init__(self, stats_df: pd.DataFrame):
        self.stats_df = stats_df.copy()
        self.processed_df = None
        self.percentile_df = None
        self.comparison_players = None
        self.radar_data = None
        self.knn_metrics = None
        self.display_metrics = None
        self.valid_metrics = None
        self.player_position = None
        self.non_round_metrics = ["Height (in)", "Hand Size (in)", "Arm Length (in)", "40 Yard", "3Cone", "Shuttle"]

    def process(self, input_player: str, player_year="2025"):
        combine_columns = [
            "Height (in)", "Weight (lbs)", "Hand Size (in)", "Arm Length (in)",
            "40 Yard", "Bench Press", "Vert Leap (in)",
            "Broad Jump (in)", "Shuttle", "3Cone", "POS_GP", "POS"
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

        # Merge PPA data for QBs
        try:
            ppa_df = pd.read_csv(PPA_DATA_PATH)
            ppa_df["year"] = ppa_df["year"].astype(str)
            df["year"] = df["year"].astype(str)
            df = pd.merge(df, ppa_df[["athlete_id", "year", "ppa_per_dropback"]],
                         on=["athlete_id", "year"], how="left")
        except Exception as e:
            print("Error reading PPA data:", e)

        # Merge EPA/Rec data for WRs/TEs
        try:
            wr_epa_df = pd.read_csv(WR_EPA_PATH)
            wr_epa_df["year"] = wr_epa_df["year"].astype(str)
            df["year"] = df["year"].astype(str)
            merge_cols = ["athlete_id", "year", "epa_per_rec"]
            for col in ["incompletions", "targets"]:
                if col in wr_epa_df.columns:
                    merge_cols.append(col)
            df = pd.merge(df, wr_epa_df[merge_cols],
                         on=["athlete_id", "year"], how="left")
        except Exception as e:
            print("Error reading WR EPA data:", e)

        # Merge PFF data
        try:
            df = merge_pff_passing(df)
        except Exception as e:
            print("Error merging PFF data:", e)

        df = df.drop_duplicates(subset=["player", "year", "team", "athlete_id"])
        if input_player not in df["player"].unique():
            raise ValueError(f"Input player {input_player} not found in data.")
        df_player_year = df[(df["player"] == input_player) & (df["year"] == player_year)]
        if df_player_year.empty:
            fallback_year = df.loc[df["player"] == input_player, "year"].max()
            print(f"No data found for {input_player} in year={player_year}. Using fallback year={fallback_year} instead.")
            df_player_year = df[(df["player"] == input_player) & (df["year"] == fallback_year)]
        if df_player_year.empty:
            raise ValueError(f"No stats found at all for player={input_player}.")
        target_id = df_player_year["athlete_id"].iloc[0]
        df = df[~((df["player"] == input_player) & (df["athlete_id"] != target_id))]
        player_position = df_player_year["POS_GP"].values[0]
        if pd.isna(player_position):
            player_position = df_player_year["position"].values[0]
        if pd.isna(player_position):
            player_position = df.loc[df["player"] == input_player, "position"].dropna().values
            player_position = player_position[0] if len(player_position) > 0 else None
        is_input_player = df["player"] == input_player

        # Get specific POS for DL subtype detection
        player_pos = df_player_year["POS"].values[0] if "POS" in df_player_year.columns else None
        if pd.isna(player_pos) if player_pos is not None else True:
            player_pos = None

        if player_position in ["FS", "SS"]:
            position_key = "S"
            df = df[is_input_player | df["POS_GP"].isin(["FS", "SS"]) | df["position"].isin(["FS", "SS"])]
        elif player_position == "DL":
            # Split DL into DE/LB (edge) vs DT based on specific POS
            if player_pos in ["DT", "NT", "DL"]:
                position_key = "DT"
                df = df[is_input_player
                        | ((df["POS_GP"] == "DL") & (df["POS"].isin(["DT", "NT", "DL"])))
                        | (df["POS_GP"] == "DT")
                        | df["position"].isin(["DT", "NT"])]
            else:
                # DE or unknown DL → edge group
                position_key = "DE/LB"
                df = df[is_input_player
                        | ((df["POS_GP"] == "DL") & (df["POS"] == "DE"))
                        | (df["POS_GP"] == "LB")
                        | (df["POS_GP"] == "EDGE")
                        | df["position"].isin(["DE", "LB", "OLB", "ILB", "EDGE"])]
        elif player_position in ["LB", "EDGE"]:
            position_key = "DE/LB"
            df = df[is_input_player
                    | ((df["POS_GP"] == "DL") & (df["POS"] == "DE"))
                    | (df["POS_GP"] == "LB")
                    | (df["POS_GP"] == "EDGE")
                    | df["position"].isin(["DE", "LB", "OLB", "ILB", "EDGE"])]
        elif player_position == "DT":
            position_key = "DT"
            df = df[is_input_player
                    | ((df["POS_GP"] == "DL") & (df["POS"].isin(["DT", "NT", "DL"])))
                    | (df["POS_GP"] == "DT")
                    | df["position"].isin(["DT", "NT"])]
        else:
            position_key = player_position
            df = df[is_input_player | (df["POS_GP"] == player_position) | (df["position"] == player_position)]
        baseline_metrics = POSITION_BASELINES.get(position_key, [])
        valid_metrics = [m for m in baseline_metrics if m in df.columns]
        for stat in combine_columns:
            if stat in df.columns:
                player_val = df.loc[df["player"] == input_player, stat]
                if not player_val.empty and pd.notnull(player_val.iloc[0]) and stat not in valid_metrics:
                    valid_metrics.append(stat)
        if "POS_GP" in valid_metrics:
            valid_metrics.remove("POS_GP")
        if "POS" in valid_metrics:
            valid_metrics.remove("POS")
        for stat in combine_columns:
            if stat in df.columns:
                df[stat] = pd.to_numeric(df[stat], errors="coerce")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        agg_funcs = {col: "sum" for col in numeric_cols if col not in combine_columns and col != "athlete_id"}
        if "interceptions_avg" in agg_funcs:
            agg_funcs["interceptions_avg"] = "mean"
        if "passing_ypa" in agg_funcs:
            agg_funcs["passing_ypa"] = "mean"
        if "ppa_per_dropback" in agg_funcs:
            agg_funcs["ppa_per_dropback"] = "mean"
        if "epa_per_rec" in agg_funcs:
            agg_funcs["epa_per_rec"] = "mean"
        for stat in combine_columns:
            if stat in df.columns:
                agg_funcs[stat] = "first"
        df_sum = df.groupby(["player", "athlete_id"]).agg(agg_funcs).reset_index()
        if "passing_pct" in df.columns:
            passing_pct_avg = df.groupby(["player", "athlete_id"])["passing_pct"].mean().reset_index()
            df_sum = df_sum.merge(passing_pct_avg, on=["player", "athlete_id"], how="left", suffixes=("", "_mean"))
            df_sum["passing_pct"] = df_sum["passing_pct_mean"] * 100
            df_sum.drop(columns=["passing_pct_mean"], inplace=True)
        if "kicking_pct" in df.columns:
            kicking_pct_avg = df.groupby(["player", "athlete_id"])["kicking_pct"].mean().reset_index()
            df_sum = df_sum.merge(kicking_pct_avg, on=["player", "athlete_id"], how="left", suffixes=("", "_mean"))
            df_sum["kicking_pct"] = df_sum["kicking_pct_mean"] * 100
            df_sum.drop(columns=["kicking_pct_mean"], inplace=True)
        if "punting_ypp" in df.columns:
            punting_ypp_avg = df.groupby(["player", "athlete_id"])["punting_ypp"].mean().reset_index()
            df_sum = df_sum.merge(punting_ypp_avg, on=["player", "athlete_id"], how="left", suffixes=("", "_mean"))
            df_sum["punting_ypp"] = df_sum["punting_ypp_mean"] * 100
            df_sum.drop(columns=["punting_ypp_mean"], inplace=True)
        if "kicking_long" in df.columns:
            kicking_pct_avg = df.groupby(["player", "athlete_id"])["kicking_long"].mean().reset_index()
            df_sum = df_sum.merge(kicking_pct_avg, on=["player", "athlete_id"], how="left", suffixes=("", "_mean"))
            df_sum["kicking_lomg"] = df_sum["kicking_long_mean"] * 100
            df_sum.drop(columns=["kicking_long_mean"], inplace=True)
        if "receiving_rec" in df_sum.columns and "receiving_yds" in df_sum.columns:
            df_sum["receiving_ypr"] = df_sum.apply(lambda row: row["receiving_yds"] / row["receiving_rec"] if row["receiving_rec"] > 0 else 0, axis=1)
        if "passing_completions" in df_sum.columns and "passing_att" in df_sum.columns:
            df_sum["comp_att"] = df_sum["passing_completions"].astype(int).astype(str) + "/" + df_sum["passing_att"].astype(int).astype(str)
            if "comp_att" in baseline_metrics and "comp_att" not in valid_metrics:
                valid_metrics.insert(baseline_metrics.index("comp_att"), "comp_att")
        if all(col in df.columns for col in ["rushing_yds", "rushing_car", "rushing_ypc"]):
            rushing_agg = df.groupby(["player", "athlete_id"])[["rushing_yds", "rushing_car"]].sum()
            rushing_ypc_avg = (rushing_agg["rushing_yds"] / rushing_agg["rushing_car"].replace(0, float("nan"))).fillna(0).reset_index()
            rushing_ypc_avg.columns = ["player", "athlete_id", "rushing_ypc"]
            df_sum.drop(columns=["rushing_ypc"], errors="ignore", inplace=True)
            df_sum = df_sum.merge(rushing_ypc_avg, on=["player", "athlete_id"], how="left")
            numeric_metrics = [m for m in valid_metrics if m not in ("comp_att", "rec_targets")]
            df_sum[numeric_metrics] = df_sum[numeric_metrics].apply(pd.to_numeric, errors="coerce")
        string_columns = {"comp_att", "rec_targets"}
        exclusions = EXCLUDE_FROM_KNN.get(position_key, [])
        self.display_metrics = [m for m in valid_metrics if m not in exclusions]
        self.valid_metrics = [m for m in valid_metrics if m not in exclusions]
        self.knn_metrics = [m for m in valid_metrics if m not in exclusions and m not in string_columns]
        print("Exclusions for position", player_position, ":", exclusions)
        print("KNN metrics used:", self.knn_metrics)
        self.processed_df = df_sum
        _seasons = df.groupby(["player", "athlete_id"])["year"].nunique().reset_index()
        _seasons.columns = ["player", "athlete_id", "_seasons"]
        df_sum = df_sum.merge(_seasons, on=["player", "athlete_id"], how="left")
        rate_metrics = {"passing_ypa", "passing_pct", "ppa_per_dropback", "epa_per_rec",
                        "rushing_ypc", "receiving_ypr", "interceptions_avg",
                        "kicking_pct", "punting_ypp"} | set(combine_columns)
        reverse_metrics = {"passing_int", "40 Yard", "3Cone", "Shuttle", "Fumbles"}
        self.percentile_df = df_sum.copy()
        for metric in valid_metrics:
            if metric == "comp_att":
                values = df_sum["passing_att"] / df_sum["_seasons"]
            elif metric == "rec_targets":
                values = df_sum["targets"] / df_sum["_seasons"]
            else:
                values = df_sum[metric]
                if metric not in rate_metrics:
                    values = values / df_sum["_seasons"]
            valid = values.notna()
            ranks = pd.Series(np.nan, index=values.index)
            if valid.any():
                ranks[valid] = rankdata(values[valid], method="average") / valid.sum() * 100
            if metric in reverse_metrics:
                ranks = 100 - ranks
            self.percentile_df[metric] = ranks
        non_combine_knn = [m for m in self.knn_metrics if m not in set(combine_columns)]
        if non_combine_knn:
            has_stats = self.percentile_df[non_combine_knn].notna().any(axis=1) & (df_sum[non_combine_knn].fillna(0).sum(axis=1) > 0)
            has_stats.loc[self.percentile_df["player"] == input_player] = True
            self.percentile_df = self.percentile_df[has_stats].reset_index(drop=True)
            df_sum = df_sum[has_stats.values].reset_index(drop=True)
            self.processed_df = df_sum
        input_index = self.percentile_df[self.percentile_df["player"] == input_player].index
        if len(input_index) == 0:
            raise ValueError("No aggregated row found for the input player.")
        input_index = input_index[0]
        self.knn_metrics = [m for m in self.knn_metrics if pd.notna(self.percentile_df.loc[input_index, m])]
        knn_data = self.percentile_df[self.knn_metrics].fillna(50)
        knn = NearestNeighbors(n_neighbors=4, metric='manhattan')
        knn.fit(knn_data.values)
        distances, indices = knn.kneighbors(knn_data.loc[input_index].values.reshape(1, -1))
        neighbor_indices = list(indices[0])
        if input_index in neighbor_indices:
            neighbor_indices.remove(input_index)
        top3_players = self.percentile_df.loc[neighbor_indices, "player"].tolist()
        top3_ids = self.percentile_df.loc[neighbor_indices, "athlete_id"].tolist()
        input_id = self.percentile_df.loc[input_index, "athlete_id"]
        self.comparison_players = [input_player] + top3_players
        self.comparison_athlete_ids = [input_id] + top3_ids
        self.radar_data = []
        for aid in self.comparison_athlete_ids:
            row = self.percentile_df[self.percentile_df["athlete_id"] == aid][self.valid_metrics].iloc[0].fillna(50).values
            self.radar_data.append(row)
        self.player_position = position_key
