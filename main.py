from data_loader import load_data
from data_processor import DataProcessor
from plotting import DraftComparisonPlotter
from config import FILTERED_STATS_PATH, COMBINE_STATS_PATH

def main():
    # Load and merge data
    stats_df = load_data(FILTERED_STATS_PATH, COMBINE_STATS_PATH)
    
    input_player = "Will Campbell"
    
    # Process the data for the input player
    processor = DataProcessor(stats_df)
    processor.process(input_player)
    
    plotter = DraftComparisonPlotter(processor, stats_df, input_player)
    # Save the image instead of displaying it
    plotter.create_plot(save=True)

if __name__ == "__main__":
    main()
