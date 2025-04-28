#!/usr/bin/env python

from data_loader import load_data
from data_processor import DataProcessor
from plotting import DraftComparisonPlotter, SinglePlayerPlotter
from config import FILTERED_STATS_PATH, COMBINE_STATS_PATH

def main():
    stats_df = load_data(FILTERED_STATS_PATH, COMBINE_STATS_PATH)
    
    input_player = "Ashton Gillotte" 

    processor = DataProcessor(stats_df)
    processor.process(input_player)

    # Create Comparison Plot
    comparison_plotter = DraftComparisonPlotter(processor, stats_df, input_player)
    comparison_plotter.create_plot(save=True)

    # Create Single Player Plot
    single_plotter = SinglePlayerPlotter(processor, stats_df, input_player)
    single_plotter.create_plot(save=True)

if __name__ == "__main__":
    main()

# TypeError: DataFrame.reset_index() got an unexpected keyword argument 'name' 
## This means that the player's athlete_id is wrong in the combine csv. Please check ESPN for the athlete_id, Happening specifically with OTs.