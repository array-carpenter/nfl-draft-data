# plotting.py
import io
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.table import Table
from PIL import Image
import pandas as pd

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
        """
        Returns the headshot URL for the given player using their athlete_id.
        """
        df_player = self.stats_df[self.stats_df["player"] == player]
        if df_player.empty:
            raise ValueError(f"No data found for player: {player}")
        if "athlete_id" not in df_player.columns:
            raise ValueError("The column 'athlete_id' is not present in the data. Please check your combine file.")
        athlete_id = df_player["athlete_id"].iloc[0]
        return f"https://a.espncdn.com/combiner/i?img=/i/headshots/college-football/players/full/{athlete_id}.png?w=350&h=254"

    def _get_latest_teams(self):
        """
        Returns a dictionary mapping players to their most recent team.
        Assumes that the 'year' column exists in the data.
        """
        latest_teams = self.stats_df.loc[self.stats_df.groupby("player")["year"].idxmax(), ["player", "team"]]
        return latest_teams.set_index("player")["team"].to_dict()

    def create_plot(self):
        # Get the dynamic headshot URL and load the image
        headshot_url = self._get_headshot_url(self.input_player)
        with urllib.request.urlopen(headshot_url) as url:
            player_image = Image.open(io.BytesIO(url.read()))
        
        # Create the figure
        fig = plt.figure(figsize=(18, 12))
        fig.patch.set_facecolor("white")
        
        # Display headshot
        player_img_ax = fig.add_axes([0.01, 0.76, 0.15, 0.15], frameon=False)
        player_img_ax.imshow(player_image)
        player_img_ax.set_xticks([])
        player_img_ax.set_yticks([])

        # Title and footer
        title_text = f"{self.input_player} ({self.proc.player_position}) NFL Draft Comparison"
        fig.text(0.18, 0.82, title_text, fontsize=40, fontweight="bold",
                 ha="left", fontproperties=ROBOTO)
        fig.text(0.18, 0.78,
                 "Ray Carpenter | @array-carpenter | TheSpade.substack.com | Player Stats Data: CFBFastR | Combine Data: NFLCombineResults.com",
                 fontsize=12, fontweight="bold", ha="left", color="gray", fontproperties=ROBOTO)

        # Divider line
        divider_ax = fig.add_axes([0, 0.75, 1, 0.005])
        divider_ax.set_facecolor("black")
        divider_ax.set_xticks([])
        divider_ax.set_yticks([])

        # Display logo
        logo_ax = fig.add_axes([0.03, 0.55, 0.15, 0.15], frameon=False)
        logo_img = mpimg.imread(LOGO_PATH)
        logo_ax.imshow(logo_img)
        logo_ax.set_xticks([])
        logo_ax.set_yticks([])

        # Prepare radar chart settings
        valid_metrics = self.proc.valid_metrics
        data_for_radar = self.proc.radar_data
        num_vars = len(valid_metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
        angles_closed = np.concatenate([angles, [angles[0]]])
        comparison_players = self.proc.comparison_players
        latest_teams_dict = self._get_latest_teams()
        player_colors = [TEAM_COLORS.get(latest_teams_dict.get(player, ""), "gray") for player in comparison_players]
        
        radar_height = 0.15
        radar_width = 0.15
        radar_y = 0.55  
        num_players = len(comparison_players)
        col_centers = np.linspace(0.3, 0.9, num_players)
        
        # Plot radar charts for each player
        for i, player_name in enumerate(comparison_players):
            ax_pos = [col_centers[i] - radar_width / 2, radar_y, radar_width, radar_height]
            rax = fig.add_axes(ax_pos, polar=True)
            # Plot the input player's data as reference
            pvec_input = np.concatenate([data_for_radar[0], [data_for_radar[0][0]]])
            rax.plot(angles_closed, pvec_input, color=player_colors[0], linewidth=2, label=self.input_player)
            rax.fill(angles_closed, pvec_input, color=player_colors[0], alpha=0.2)
            # Plot the comparison player's data if not the input
            if i > 0:
                pvec = np.concatenate([data_for_radar[i], [data_for_radar[i][0]]])
                rax.plot(angles_closed, pvec, color=player_colors[i], linewidth=2, label=player_name)
                rax.fill(angles_closed, pvec, color=player_colors[i], alpha=0.2)
            rax.set_yticklabels([])
            rax.set_xticks([])

        # Build the comparison table below the radar charts
        self._add_comparison_table(fig, valid_metrics, comparison_players, latest_teams_dict)

        plt.show()

    def _add_comparison_table(self, fig, valid_metrics, comparison_players, latest_teams_dict):
        """
        Adds a table to the figure showing comparison metrics.
        """
        table_ax = fig.add_axes([0, 0.1, 1, 0.35])
        table_ax.set_axis_off()
        table = Table(table_ax, bbox=[0, 0, 1, 1])
        cell_width = 1.0 / (len(comparison_players) + 1)
        cell_height = 1 / (len(valid_metrics) + 3)  # +3 rows for header and team row
        
        # Prepare data: pivot the processed data (rows are metrics)
        comparison_data = self.proc.processed_df.set_index("player").loc[comparison_players, valid_metrics]
        comparison_data_t = comparison_data.transpose()
        comparison_data_t.rename(index=COLUMN_RENAME_MAP, inplace=True)
        table_fontsize = 22

        # Add column headers (player names)
        for col_idx, column in enumerate(comparison_data_t.columns):
            cell = table.add_cell(0, col_idx + 1, cell_width, cell_height,
                                  text=column, loc="center", facecolor="#cccccc")
            cell.get_text().set_fontsize(table_fontsize)
            cell.visible_edges = ''

        # Add team row underneath the headers
        for col_idx, player in enumerate(comparison_data_t.columns):
            team = latest_teams_dict.get(player, "N/A")
            cell = table.add_cell(1, col_idx + 1, cell_width, cell_height,
                                  text=team, loc="center", facecolor="#f0f0f0", fontproperties=ROBOTO)
            cell.get_text().set_fontsize(table_fontsize)
            cell.visible_edges = ''

        # Add row labels and data cells
        for row_idx, (row_name, row_vals) in enumerate(comparison_data_t.iterrows()):
            cell = table.add_cell(row_idx + 2, 0, cell_width, cell_height,
                                  text=row_name, loc="center", facecolor="#cccccc", fontproperties=ROBOTO)
            cell.get_text().set_fontsize(table_fontsize)
            cell.visible_edges = "horizontal"
            for col_idx, val in enumerate(row_vals):
                if row_name == "Yards per Attempt":
                    formatted_val = f"{val:.2f}"
                elif row_name == "Completion %":
                    formatted_val = f"{val:.1f}%"
                else:
                    formatted_val = f"{int(val)}"
                cell = table.add_cell(row_idx + 2, col_idx + 1, cell_width, cell_height,
                                      text=formatted_val, loc="center", fontproperties=ROBOTO)
                cell.get_text().set_fontsize(table_fontsize)
                cell.visible_edges = 'horizontal'
        table_ax.add_table(table)
