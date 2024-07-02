import os
import yaml
from omegaconf import DictConfig

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eks.core import jax_ensemble
from matplotlib.ticker import LogLocator

from pseudo_labeler.utils import format_data_walk
from pseudo_labeler.evaluation import compute_likelihoods_and_variance, plot_heatmaps


if __name__ == "__main__":
    config_file = "./configs/pipeline_example.yaml"
    with open(config_file, "r") as file:
        cfg = yaml.safe_load(file)

    lightning_pose_config_path = cfg.get("lightning_pose_config")
    with open(lightning_pose_config_path, "r") as file:
        lightning_pose_cfg = yaml.safe_load(file)

    cfg_lp = DictConfig(lightning_pose_cfg)

    data_dir = cfg_lp.data.data_dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    input_dir = os.path.join(parent_dir, (
                f"../outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
                f"pseudo={cfg['n_pseudo_labels']}/networks/"
            )
        )
    data_type = 'lp'
    csv_name = "predictions_new.csv"

    bodypart_list = cfg_lp["data"]["keypoint_names"]

    input_dfs, output_df, _ = format_data_walk(input_dir, data_type, csv_name)
    print(f'Found {len(input_dfs)} models.')

    likelihoods_above_thresh, summed_ensemble_vars, combined_df = compute_likelihoods_and_variance(cfg_lp, input_dfs, bodypart_list)
    plot_heatmaps(likelihoods_above_thresh, summed_ensemble_vars, bodypart_list, data_dir, script_dir)
