from data_loader import load_data
from data_processor import DataProcessor
from plotting import DraftComparisonPlotter, SinglePlayerPlotter
from config import FILTERED_STATS_PATH, COMBINE_STATS_PATH

def main(viz_type="comparison"):
    stats_df = load_data(FILTERED_STATS_PATH, COMBINE_STATS_PATH)
    
    input_player = "Moliki Matavao"
    
    processor = DataProcessor(stats_df)
    processor.process(input_player)

    if viz_type == "comparison":
        plotter = DraftComparisonPlotter(processor, stats_df, input_player)
        plotter.create_plot(save=True)

    elif viz_type == "single_player":
        plotter = SinglePlayerPlotter(processor, stats_df, input_player)
        plotter.create_plot(save=True)

    else:
        raise ValueError(f"Unsupported viz_type: {viz_type}")

if __name__ == "__main__":
    main(viz_type="comparison") 

# TypeError: DataFrame.reset_index() got an unexpected keyword argument 'name' 
## This means that the player's athlete_id is wrong in the combine csv. Please check ESPN for the athlete_id, Happening specifically with OTs. 