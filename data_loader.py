# data_loader.py
import pandas as pd

def load_data(filtered_stats_path: str, combine_stats_path: str) -> pd.DataFrame:
    """
    Loads and merges the filtered stats and combine stats datasets.
    """
    # Load datasets
    filtered_stats = pd.read_csv(filtered_stats_path)
    combine_stats = pd.read_csv(combine_stats_path)
    
    # Check that combine_stats contains the athlete_id (for verification purposes)
    if "athlete_id" not in combine_stats.columns:
        raise ValueError("Column 'athlete_id' not found in the combine stats file.")
    
    # Rename columns in combine_stats so that 'POS' becomes 'position'
    combine_stats.rename(columns={"Name": "player", "POS": "position"}, inplace=True)
    
    # Merge only the position column from combine_stats into filtered_stats.
    # (We do not merge athlete_id since filtered_stats already has it.)
    merged_stats = filtered_stats.merge(
        combine_stats[["player", "position"]],
        on="player",
        how="left"
    )
    
    # Optionally handle missing positions (example below)
    merged_stats.loc[merged_stats["player"] == "Ashton Jeanty", "position"] = "RB"
    
    return merged_stats
