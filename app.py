import io

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_data
from data_processor import DataProcessor
from plotting import DraftComparisonPlotter
from config import FILTERED_STATS_PATH, COMBINE_STATS_PATH

st.set_page_config(page_title="NFL Draft Comparison Cards", layout="wide")

# Bump this any time data_processor / config / plotting logic changes to bust the cache.
_CACHE_VERSION = 2


@st.cache_data
def load_combine_options():
    combine = pd.read_csv(COMBINE_STATS_PATH)
    combine = combine.sort_values("Year", ascending=False)
    options = []
    for _, row in combine.iterrows():
        label = f"{row['player']} ({row['POS_GP']}, {row['College']}, {row['Year']})"
        options.append({
            "label": label,
            "player": row["player"],
            "year": str(int(row["Year"])),
        })
    return options


@st.cache_data
def get_stats_df():
    return load_data(FILTERED_STATS_PATH, COMBINE_STATS_PATH)


@st.cache_data
def run_processor(player_name, year, _v=_CACHE_VERSION):
    stats_df = get_stats_df()
    processor = DataProcessor(stats_df)
    processor.process(player_name, player_year=year)
    return processor, stats_df


@st.cache_data
def render_card(player_name, year, _v=_CACHE_VERSION):
    """Render the comparison card to a high-res PNG buffer."""
    processor, stats_df = run_processor(player_name, year)
    plotter = DraftComparisonPlotter(processor, stats_df, player_name)
    fig = plotter.create_plot(save=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


st.title("NFL Draft Comparison Cards")
st.markdown("These are NFL Draft comparisons built using a k-nearest neighbor (KNN) algorithm that takes raw combine and production stats, calculates percentiles, then uses those percentiles to find each prospect's 3 most similar players. Volume stats are normalized per season to account for the different number of years each college player played. Designed by [Ray Carpenter](https://TheSpade.Substack.com).")

options = load_combine_options()
labels = [o["label"] for o in options]

selected = st.selectbox(
    "Search for a player",
    options=range(len(labels)),
    format_func=lambda i: labels[i],
    index=None,
    placeholder="Type a player name...",
)

if selected is not None:
    player_name = options[selected]["player"]
    year = options[selected]["year"]

    with st.spinner(f"Processing {player_name}..."):
        try:
            png_bytes = render_card(player_name, year)
        except Exception as e:
            st.error(f"Error processing {player_name}: {e}")
            st.stop()

    st.image(png_bytes, use_container_width=True)

    st.download_button(
        label="Download Comparison Card (PNG)",
        data=png_bytes,
        file_name=f"{player_name.replace(' ', '_')}_comparison.png",
        mime="image/png",
    )
