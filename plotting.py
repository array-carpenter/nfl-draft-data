import io
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.table import Table
from PIL import Image
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
import os
import matplotlib.gridspec as gridspec
import seaborn as sns

from config import ROBOTO, TEAM_COLORS, COLUMN_RENAME_MAP, LOGO_PATH

class DraftComparisonPlotter:
    def __init__(self, processed_data, original_stats_df, input_player: str):
        """
        processed_data: instance of DataProcessor that has been run with .process()
        original_stats_df: the merged stats DataFrame (needed for additional info like team)
        input_player: name of the input player
        """
        self.proc = processed_data
        self.stats_df = original_stats_df
        self.input_player = input_player

    def _get_headshot_url(self, player: str) -> str:
        df_player = self.stats_df[self.stats_df["player"] == player]
        if df_player.empty:
            raise ValueError(f"No data found for player: {player}")
        if "athlete_id" not in df_player.columns:
            raise ValueError("The column 'athlete_id' is not present in the data. Please check your combine file.")
        athlete_id = df_player["athlete_id"].iloc[0]
        return (
            f"https://a.espncdn.com/combiner/i?img=/i/headshots/college-football/players/full/{athlete_id}.png?"
            "w=350&h=254"
        )

    def _get_latest_teams(self):
        latest_teams = self.stats_df.loc[
            self.stats_df.groupby("player")["year"].idxmax(), ["player", "team"]
        ]
        return latest_teams.set_index("player")["team"].to_dict()

    def create_plot(self, save=False):
        # Increase figure size dramatically
        fig = plt.figure(figsize=(28, 18))
        fig.patch.set_facecolor("white")

        headshot_url = self._get_headshot_url(self.input_player)
        with urllib.request.urlopen(headshot_url) as url:
            player_image = Image.open(io.BytesIO(url.read()))

        player_img_ax = fig.add_axes([0.01, 0.76, 0.15, 0.15], frameon=False)
        player_img_ax.imshow(player_image)
        player_img_ax.set_xticks([])
        player_img_ax.set_yticks([])

        title_text = f"{self.input_player} ({self.proc.player_position}) NFL Draft Comparison"
        fig.text(
            0.18, 0.82, title_text,
            fontsize=60, fontweight="bold",
            ha="left", fontproperties=ROBOTO
        )
        fig.text(
            0.18, 0.78,
            "Ray Carpenter | TheSpade.substack.com | "
            "Player Stats Data: CFBFastR | Combine Data Since 2007 (Pro Day Adjusted): NFLCombineResults.com",
            fontsize=20, ha="left", color="#474746", fontproperties=ROBOTO
        )

        divider_ax = fig.add_axes([0, 0.75, 1, 0.005])
        divider_ax.set_facecolor("black")
        divider_ax.set_xticks([])
        divider_ax.set_yticks([])

        logo_ax = fig.add_axes([0.03, 0.55, 0.15, 0.15], frameon=False)
        logo_img = mpimg.imread(LOGO_PATH)
        logo_ax.imshow(logo_img)
        logo_ax.set_xticks([])
        logo_ax.set_yticks([])

        valid_metrics = self.proc.valid_metrics
        data_for_radar = self.proc.radar_data
        comparison_players = self.proc.comparison_players

        print(f"\nPercentile Rankings for {self.input_player}:\n")
        input_player_idx = comparison_players.index(self.input_player)
        for metric, percentile in zip(valid_metrics, data_for_radar[input_player_idx]):
            display_name = COLUMN_RENAME_MAP.get(metric, metric)
            print(f"{display_name}: {percentile:.1f}")

        num_vars = len(valid_metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
        angles_closed = np.concatenate([angles, [angles[0]]])

        latest_teams_dict = self._get_latest_teams()
        player_colors = [
            TEAM_COLORS.get(latest_teams_dict.get(player, ""), "gray")
            for player in comparison_players
        ]

        radar_height = 0.15
        radar_width = 0.15
        radar_y = 0.55
        num_players = len(comparison_players)
        col_centers = np.linspace(0.3, 0.9, num_players)

        # Plot the radar charts
        for i, player_name in enumerate(comparison_players):
            ax_pos = [
                col_centers[i] - radar_width / 2,
                radar_y,
                radar_width,
                radar_height
            ]
            rax = fig.add_axes(ax_pos, polar=True)

            # Input player's radar
            pvec_input = np.concatenate([data_for_radar[0], [data_for_radar[0][0]]])
            rax.plot(
                angles_closed, pvec_input,
                color=player_colors[0], linewidth=2, label=self.input_player
            )
            rax.fill(angles_closed, pvec_input, color=player_colors[0], alpha=0.2)

            if i > 0:
                pvec = np.concatenate([data_for_radar[i], [data_for_radar[i][0]]])
                rax.plot(
                    angles_closed, pvec,
                    color=player_colors[i], linewidth=2, label=player_name
                )
                rax.fill(angles_closed, pvec, color=player_colors[i], alpha=0.2)

            rax.set_yticklabels([])
            rax.set_xticks([])

        self._add_comparison_table(fig, valid_metrics, comparison_players, latest_teams_dict)

        if save:
            folder = "2025_post_combine"
            os.makedirs(folder, exist_ok=True)
            filename = os.path.join(
                folder,
                f"{self.input_player.replace(' ', '_')}_post_combine.png"
            )
            plt.savefig(filename, bbox_inches="tight")
            plt.close(fig)
            print(f"Plot saved to {filename}")
        else:
            plt.show()

    def _add_comparison_table(self, fig, valid_metrics, comparison_players, latest_teams_dict):
        """
        Adds a table to the figure showing comparison metrics.
        """

        table_ax = fig.add_axes([0, 0.05, 1, 0.5]) 
        table_ax.set_axis_off()

        table = Table(table_ax, bbox=[0, 0, 1, 1])

        table_fontsize = 22

        comparison_data = self.proc.processed_df.set_index("player").loc[
            comparison_players, valid_metrics
        ]
        comparison_data_t = comparison_data.transpose()
        comparison_data_t.rename(index=COLUMN_RENAME_MAP, inplace=True)


        num_rows = len(valid_metrics) + 2

        cell_width = 1.0 / (len(comparison_data_t.columns) + 1)
        cell_height = 1.0 / (num_rows)  

        for col_idx, column in enumerate(comparison_data_t.columns):
            cell = table.add_cell(
                row=0, col=col_idx + 1,
                width=cell_width, height=cell_height,
                text=column,
                loc="center",
                facecolor="#cccccc"
            )
            cell.get_text().set_fontsize(table_fontsize)
            cell.visible_edges = ""

        for col_idx, player in enumerate(comparison_data_t.columns):
            team = latest_teams_dict.get(player, "N/A")
            cell = table.add_cell(
                row=1, col=col_idx + 1,
                width=cell_width, height=cell_height,
                text=team,
                loc="center",
                facecolor="#f0f0f0",
                fontproperties=ROBOTO
            )
            cell.get_text().set_fontsize(table_fontsize)
            cell.visible_edges = ""

        for row_idx, (row_name, row_vals) in enumerate(comparison_data_t.iterrows()):
            row_num = row_idx + 2  

            cell = table.add_cell(
                row=row_num,
                col=0,
                width=cell_width,
                height=cell_height,
                text=row_name,
                loc="center",
                facecolor="#cccccc",
                fontproperties=ROBOTO
            )
            cell.get_text().set_fontsize(table_fontsize)
            cell.visible_edges = "horizontal"

            for col_idx, val in enumerate(row_vals):
                if row_name in {
                    "40-Yard Dash", "3-Cone Drill", "Height (in)",
                    "Hand Size (in)", "Arm Length (in)", "Shuttle","Yards per Carry"
                }:
                    formatted_val = f"{val:.2f}"
                elif row_name == "Yards per Attempt":
                    formatted_val = f"{val:.2f}"
                elif row_name == "Yards per Attempt":
                    formatted_val = f"{val:.2f}"
                elif row_name == "Completion %":
                    formatted_val = f"{val:.1f}%"
                elif row_name == "Defensive Sacks":
                    formatted_val = f"{val:.1f}"
                else:
                    formatted_val = f"{int(val)}"

                cell = table.add_cell(
                    row=row_num,
                    col=col_idx + 1,
                    width=cell_width,
                    height=cell_height,
                    text=formatted_val,
                    loc="center",
                    fontproperties=ROBOTO
                )
                cell.get_text().set_fontsize(table_fontsize)
                cell.visible_edges = "horizontal"

        table_ax.add_table(table)

        table.scale(1.0, 1.3)
class SinglePlayerPlotter:
    def __init__(self, processed_data, original_stats_df, input_player: str):
        self.proc = processed_data
        self.input_player = input_player
        self.stats_df = original_stats_df

    def _get_player_team(self):
        player_df = self.stats_df[self.stats_df['player'] == self.input_player]
        if player_df.empty:
            raise ValueError(f"No team data found for player {self.input_player}")
        latest_year = player_df['year'].max()
        return player_df[player_df['year'] == latest_year]['team'].iloc[0]

    def create_plot(self, save=False):
        fig, axes = plt.subplots(len(self.proc.valid_metrics), 1, figsize=(12, len(self.proc.valid_metrics) * 1.2), sharex=True)
        fig.patch.set_facecolor('white')

        player_team = self._get_player_team()
        team_color = TEAM_COLORS.get(player_team, "#444444")

        percentiles_df = self.proc.percentile_df
        processed_df = self.proc.processed_df
        valid_metrics = self.proc.valid_metrics

        for i, metric in enumerate(valid_metrics[::-1]):
            ax = axes[i]
            values = percentiles_df[metric].dropna().values
            player_percentile = percentiles_df.loc[percentiles_df['player'] == self.input_player, metric].values[0]
            player_raw_value = processed_df.loc[processed_df['player'] == self.input_player, metric].values[0]

            kde = gaussian_kde(values,bw_method = 0.1)
            x_vals = np.linspace(0, 100, 200)
            kde_vals = kde(x_vals)

            ax.fill_between(x_vals, kde_vals, color='lightgrey', alpha=0.6)
            ax.fill_between(x_vals, kde_vals, where=(x_vals <= player_percentile), color=team_color, alpha=0.9)

            ax.text(-5, 0, COLUMN_RENAME_MAP.get(metric, metric), fontsize=12, fontproperties=ROBOTO, fontweight="bold", ha="right", va='center')

            ax.text(.9, 0.5, f"{player_raw_value:.2f}", fontsize=11, ha="left", 
                    va='center', fontproperties=ROBOTO, color='black', transform=ax.transAxes)
            ax.text(.9, 0.35, f"{player_percentile:.0f}%tile", fontsize=11, ha="left", 
                    va='center', fontproperties=ROBOTO, color=team_color, transform=ax.transAxes)

            ax.set_xlim(0, 120)
            ax.set_yticks([])
            ax.set_ylabel('')

            for spine in ax.spines.values():
                spine.set_visible(False)

            if i < len(valid_metrics) - 1:
                ax.set_xticks([])

       # axes[-1].set_xlabel("Percentile Rank", fontsize=14, fontproperties=ROBOTO, fontweight='bold')

        plt.suptitle(
            f"{self.input_player} NFL Draft Percentile Profile",
            fontsize=24,
            fontweight='bold',
            fontproperties=ROBOTO,
            color=team_color,
            y=0.95
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save:
            filename = f"{self.input_player.replace(' ', '_')}_percentile_ridges.png"
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            print(f"Plot saved to {filename}")
        else:
            plt.show()