# main.py
from data_loader import load_data
from data_processor import DataProcessor
from plotting import DraftComparisonPlotter
from config import FILTERED_STATS_PATH, COMBINE_STATS_PATH

def main():
    # Load and merge data
    stats_df = load_data(FILTERED_STATS_PATH, COMBINE_STATS_PATH)
    
    # Set the input player to "Cameron Skattebo" (athlete_id 4696981)
    input_player = "Cameron Skattebo"
    
    # Process the data for the input player
    processor = DataProcessor(stats_df)
    processor.process(input_player)
    
    # Create and display the draft comparison plot
    plotter = DraftComparisonPlotter(processor, stats_df, input_player)
    plotter.create_plot()

if __name__ == "__main__":
    main()
