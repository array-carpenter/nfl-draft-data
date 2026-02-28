import io
import json
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

from config import ROBOTO, INSTRUMENT_SERIF, TEAM_COLORS, COLUMN_RENAME_MAP, LOGO_PATH

DRAFT_PICKS_PATH = "draft_picks.csv"

def get_draft_position(athlete_id, stats_df=None):
    """Look up a player's NFL draft position from draft_picks.csv."""
    try:
        picks = pd.read_csv(DRAFT_PICKS_PATH)
        match = picks[picks["college_athlete_id"] == athlete_id]
        if not match.empty:
            row = match.iloc[0]
            return f"Round {int(row['round'])} Pick {int(row['pick'])}"
    except Exception:
        pass
    if stats_df is not None:
        player_rows = stats_df[stats_df["athlete_id"] == athlete_id]
        if not player_rows.empty and str(player_rows["year"].max()) == "2025":
            return "?"
    return "Undrafted"

TEAM_LOGO_MAP_PATH = "team_logo_map.json"
LOGOS_DIR = "logos"

def get_team_logo_path(team_name):
    """Map a team name to its logo file path."""
    try:
        with open(TEAM_LOGO_MAP_PATH, "r") as f:
            logo_map = json.load(f)
        logo_file = logo_map.get(team_name)
        if logo_file:
            path = os.path.join(LOGOS_DIR, logo_file)
            if os.path.exists(path):
                return path
    except Exception:
        pass
    return None

class DraftComparisonPlotter:
    def __init__(self, processed_data, original_stats_df, input_player: str):
        self.proc = processed_data
        self.stats_df = original_stats_df
        self.input_player = input_player

    def _get_athlete_id(self, player: str):
        df_player = self.stats_df[self.stats_df["player"] == player]
        if df_player.empty:
            df_player = self.proc.processed_df[self.proc.processed_df["player"] == player]
        if df_player.empty or "athlete_id" not in df_player.columns:
            return None
        athlete_id = df_player["athlete_id"].iloc[0]
        if pd.isna(athlete_id) or athlete_id < 0:
            return None
        return int(athlete_id)

    def _get_latest_teams(self):
        latest_teams = self.stats_df.loc[
            self.stats_df.groupby("athlete_id")["year"].idxmax(), ["athlete_id", "team"]
        ]
        return latest_teams.set_index("athlete_id")["team"].to_dict()

    def create_plot(self, save=False):
        fig = plt.figure(figsize=(28, 18))
        fig.patch.set_facecolor("#DDEBEC")

        athlete_id = self._get_athlete_id(self.input_player)
        player_image = None
        if athlete_id:
            for path in ("college-football", "nfl"):
                try:
                    url = f"https://a.espncdn.com/combiner/i?img=/i/headshots/{path}/players/full/{athlete_id}.png?w=350&h=254"
                    with urllib.request.urlopen(url) as resp:
                        player_image = Image.open(io.BytesIO(resp.read()))
                    break
                except Exception:
                    continue
        if player_image:
            player_img_ax = fig.add_axes([0.01, 0.76, 0.15, 0.15], frameon=False)
            player_img_ax.imshow(player_image)
            player_img_ax.set_xticks([])
            player_img_ax.set_yticks([])

        title_text = f"{self.input_player} ({self.proc.player_position}) NFL Draft Comparison"
        fig.text(
            0.18, 0.82, title_text,
            fontsize=72, fontweight="bold",
            ha="left", fontproperties=INSTRUMENT_SERIF
        )
        fig.text(
            0.18, 0.78,
            "Ray Carpenter | TheSpade.substack.com | Stats: CFBD | Combine Data: Various Sources",
            fontsize=26, ha="left", color="#474746", fontproperties=ROBOTO
        )

        # Add team logo to top right
        latest_teams_dict = self._get_latest_teams()
        comparison_athlete_ids = self.proc.comparison_athlete_ids
        player_team = latest_teams_dict.get(comparison_athlete_ids[0], "")
        team_logo_path = get_team_logo_path(player_team)
        if team_logo_path:
            team_logo_ax = fig.add_axes([0.85, 0.76, 0.15, 0.15], frameon=False)
            team_logo_img = Image.open(team_logo_path)
            team_logo_ax.imshow(team_logo_img)
            team_logo_ax.set_xticks([])
            team_logo_ax.set_yticks([])

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
        for metric, percentile in zip(valid_metrics, data_for_radar[0]):
            display_name = COLUMN_RENAME_MAP.get(metric, metric)
            print(f"{display_name}: {percentile:.1f}")

        num_vars = len(valid_metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
        angles_closed = np.concatenate([angles, [angles[0]]])

        player_colors = [
            TEAM_COLORS.get(latest_teams_dict.get(aid, ""), "gray")
            for aid in comparison_athlete_ids
        ]

        radar_height = 0.15
        radar_width = 0.15
        radar_y = 0.55
        num_players = len(comparison_players)
        col_centers = np.linspace(0.3, 0.9, num_players)

        for i, player_name in enumerate(comparison_players):
            ax_pos = [
                col_centers[i] - radar_width / 2,
                radar_y,
                radar_width,
                radar_height
            ]
            rax = fig.add_axes(ax_pos, polar=True)

            pvec_input = np.concatenate([data_for_radar[0], [data_for_radar[0][0]]])
            rax.plot(angles_closed, pvec_input, color=player_colors[0], linewidth=2)
            rax.fill(angles_closed, pvec_input, color=player_colors[0], alpha=0.2)

            if i > 0:
                pvec = np.concatenate([data_for_radar[i], [data_for_radar[i][0]]])
                rax.plot(angles_closed, pvec, color=player_colors[i], linewidth=2)
                rax.fill(angles_closed, pvec, color=player_colors[i], alpha=0.2)

            rax.set_yticklabels([])
            rax.set_xticks([])

        self._add_comparison_table(fig, valid_metrics, comparison_players, comparison_athlete_ids, latest_teams_dict)

        if save:
            folder = "2026_pre_combine"
            os.makedirs(folder, exist_ok=True)
            filename = os.path.join(folder, f"{self.input_player.replace(' ', '_')}_pre_combine.png")
            plt.savefig(filename, bbox_inches="tight")
            plt.close(fig)
            print(f"Plot saved to {filename}")
        return fig

    def _add_comparison_table(self, fig, valid_metrics, comparison_players, comparison_athlete_ids, latest_teams_dict):
        table_ax = fig.add_axes([0, 0.05, 1, 0.5])
        table_ax.set_axis_off()
        table = Table(table_ax, bbox=[0, 0, 1, 1])
        table_fontsize = 30

        proc_df = self.proc.processed_df
        pctl_df = self.proc.percentile_df
        comp_rows = [proc_df[proc_df["athlete_id"] == aid].iloc[0] for aid in comparison_athlete_ids]
        comparison_data = pd.DataFrame(comp_rows)[valid_metrics].T
        comparison_data.columns = comparison_players
        comparison_data.rename(index=COLUMN_RENAME_MAP, inplace=True)

        pctl_rows = [pctl_df[pctl_df["athlete_id"] == aid].iloc[0] for aid in comparison_athlete_ids]
        percentile_data = pd.DataFrame(pctl_rows)[valid_metrics].T
        percentile_data.columns = comparison_players

        num_rows = len(valid_metrics) + 3
        cell_width = 1.0 / (len(comparison_players) + 1)
        cell_height = 1.0 / num_rows

        for col_idx, name in enumerate(comparison_players):
            cell = table.add_cell(row=0, col=col_idx + 1, width=cell_width, height=cell_height, text=name, loc="center", facecolor="#cccccc")
            cell.get_text().set_fontsize(table_fontsize)
            cell.visible_edges = ""

        for col_idx, aid in enumerate(comparison_athlete_ids):
            team = latest_teams_dict.get(aid, "N/A")
            cell = table.add_cell(row=1, col=col_idx + 1, width=cell_width, height=cell_height, text=team, loc="center", facecolor="#f0f0f0", fontproperties=ROBOTO)
            cell.get_text().set_fontsize(table_fontsize)
            cell.visible_edges = ""

        for row_idx, (row_name, row_vals) in enumerate(comparison_data.iterrows()):
            row_num = row_idx + 2
            cell = table.add_cell(row=row_num, col=0, width=cell_width, height=cell_height, text=row_name, loc="center", facecolor="#cccccc", fontproperties=ROBOTO)
            cell.get_text().set_fontsize(table_fontsize)
            cell.set_edgecolor("#DDEBEC")

            pctile_row = percentile_data.iloc[row_idx]
            leader_idx = pctile_row.argmax()

            for col_idx, val in enumerate(row_vals):
                if pd.isna(val):
                    formatted_val = "—"
                elif row_name in {"40-Yard Dash", "10-Yard Split", "3-Cone Drill", "Height (in)", "Hand Size (in)", "Arm Length (in)", "Shuttle", "Yards per Carry"}:
                    formatted_val = f"{val:.2f}"
                elif row_name == "Yards per Attempt":
                    formatted_val = f"{val:.2f}"
                elif row_name == "EPA/Dropback":
                    formatted_val = f"{val:.3f}"
                elif row_name == "EPA/Reception":
                    formatted_val = f"{val:.1f}"
                elif row_name == "Catch Rate":
                    formatted_val = f"{val:.1%}"
                elif row_name in {"Comp/Att", "Rec/Targets"}:
                    formatted_val = str(val)
                elif row_name == "YPC":
                    formatted_val = f"{val:.1f}"
                elif row_name == "Completion %":
                    formatted_val = f"{val:.1f}%"
                elif row_name == "Defensive Sacks":
                    formatted_val = f"{val:.1f}"
                else:
                    formatted_val = f"{int(val)}"

                if col_idx == leader_idx:
                    bg, fg = "#FFFF99", "black"
                else:
                    bg, fg = "#DDEBEC", "black"

                cell = table.add_cell(row=row_num, col=col_idx + 1, width=cell_width, height=cell_height, text=formatted_val, loc="center", fontproperties=ROBOTO, facecolor=bg)
                cell.get_text().set_fontsize(table_fontsize)
                cell.get_text().set_color(fg)
                cell.set_edgecolor("#DDEBEC")

        # Draft position row at the bottom
        draft_row_num = len(valid_metrics) + 2
        cell = table.add_cell(row=draft_row_num, col=0, width=cell_width, height=cell_height, text="Draft Position", loc="center", facecolor="#cccccc", fontproperties=ROBOTO)
        cell.get_text().set_fontsize(table_fontsize)
        cell.set_edgecolor("#DDEBEC")
        for col_idx, aid in enumerate(comparison_athlete_ids):
            draft_text = get_draft_position(aid, stats_df=self.stats_df)
            cell = table.add_cell(row=draft_row_num, col=col_idx + 1, width=cell_width, height=cell_height, text=draft_text, loc="center", fontproperties=ROBOTO, facecolor="#DDEBEC")
            cell.get_text().set_fontsize(table_fontsize)
            cell.set_edgecolor("#DDEBEC")

        table_ax.add_table(table)
        table.scale(1.0, 1.3)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = table_ax.transData.inverted()
        cells = table.get_celld()
        drawn_y = set()
        for row in range(2, num_rows):
            cell = cells.get((row, 0))
            if cell:
                bbox = cell.get_window_extent(renderer)
                y_bottom = inv.transform(bbox)[0][1]
                y_top = inv.transform(bbox)[1][1]
                for y in (y_bottom, y_top):
                    yr = round(y, 4)
                    if yr not in drawn_y:
                        table_ax.axhline(y=y, xmin=0, xmax=1, color="black", linewidth=0.5)
                        drawn_y.add(yr)


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
        fig, axes = plt.subplots(len(self.proc.valid_metrics), 1, figsize=(12, len(self.proc.valid_metrics) * 1.2 + 2), sharex=True)
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

            kde = gaussian_kde(values, bw_method=0.1)
            x_vals = np.linspace(0, 100, 200)
            kde_vals = kde(x_vals)

            ax.fill_between(x_vals, kde_vals, color='lightgrey', alpha=0.6)
            ax.fill_between(x_vals, kde_vals, where=(x_vals <= player_percentile), color=team_color, alpha=0.9)

            ax.text(-5, 0, COLUMN_RENAME_MAP.get(metric, metric), fontsize=12, fontproperties=ROBOTO, fontweight="bold", ha="right", va='center')
            if isinstance(player_raw_value, str):
                raw_text = player_raw_value
            else:
                raw_text = f"{player_raw_value:.2f}"
            ax.text(.9, 0.5, raw_text, fontsize=11, ha="left", va='center', fontproperties=ROBOTO, color='black', transform=ax.transAxes)
            ax.text(.9, 0.35, f"{player_percentile:.0f}%tile", fontsize=11, ha="left", va='center', fontproperties=ROBOTO, color=team_color, transform=ax.transAxes)

            ax.set_xlim(0, 120)
            ax.set_yticks([])
            ax.set_ylabel('')

            for spine in ax.spines.values():
                spine.set_visible(False)

            if i < len(valid_metrics) - 1:
                ax.set_xticks([])

        plt.suptitle(
            f"{self.input_player} NFL Draft Percentile Profile",
            fontsize=24,
            fontweight='bold',
            fontproperties=ROBOTO,
            color=team_color,
            y=0.97
        )

        img_ax = fig.add_axes([0.88, 0.90, 0.13, 0.13], frameon=False)
        img = Image.open("1.png")
        img_ax.imshow(img)
        img_ax.axis("off")

        fig.text(
            0.01, 0.01,
            "Ray Carpenter | TheSpade.substack.com",
            ha="left", fontsize=12,
            fontproperties=ROBOTO, color="#333333"
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        if save:
            folder = "percentile_ridges"
            os.makedirs(folder, exist_ok=True)
            filename = os.path.join(folder, f"{self.input_player.replace(' ', '_')}_percentile_ridges.png")
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            print(f"Plot saved to {filename}")
        return fig