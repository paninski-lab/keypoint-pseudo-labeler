import os
import pandas as pd
from eks.utils import convert_slp_dlc, convert_lp_dlc, make_output_dataframe

def format_data_walk(input_dir, data_type, video_name):
    input_dfs_list = []
    keypoint_names = None  # Initialize as None to ensure it's defined correctly later

    # Helper function to traverse directories
    def traverse_directories(directory):
        nonlocal keypoint_names  # Ensure we're using the outer keypoint_names variable
        for root, _, files in os.walk(directory):
            for input_file in files:
                if input_file == video_name:
                    file_path = os.path.join(root, input_file)

                    if data_type == 'slp':
                        markers_curr = convert_slp_dlc(root, input_file)
                        keypoint_names = [c[1] for c in markers_curr.columns[::3] if
                                          not c[1].startswith('Unnamed')]
                        markers_curr_fmt = markers_curr
                    elif data_type in ['lp', 'dlc']:
                        markers_curr = pd.read_csv(
                            file_path, header=[0, 1, 2], index_col=0)
                        keypoint_names = [c[1] for c in markers_curr.columns[::3] if
                                          not c[1].startswith('Unnamed')]
                        model_name = markers_curr.columns[0][0]
                        if data_type == 'lp':
                            markers_curr_fmt = convert_lp_dlc(
                                markers_curr, keypoint_names, model_name=model_name)
                        else:
                            markers_curr_fmt = markers_curr

                    markers_curr_fmt.to_csv('fmt_input.csv', index=False)
                    input_dfs_list.append(markers_curr_fmt)

    # Traverse input_dir and its subdirectories
    traverse_directories(input_dir)

    if len(input_dfs_list) == 0:
        raise FileNotFoundError(f'No predictions.csv files found in {input_dir}')

    output_df = make_output_dataframe(input_dfs_list[0])
    # returns both the formatted marker data and the empty dataframe for EKS output
    return input_dfs_list, output_df, keypoint_names