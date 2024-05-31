"""Example pipeline script."""

import argparse

import yaml
from omegaconf import DictConfig


def pipeline(config_file: str):

    # load pipeline config file
    cfg = yaml.safe_load(open(config_file, "r"))

    # -------------------------------------------------------------------------------------
    # train k supervised models on n hand-labeled frames and compute labeled OOD metrics
    # -------------------------------------------------------------------------------------
    print(f'training {cfg["n_ensemble_base"]} baseline models')

    # -------------------------------------------------------------------------------------
    # run inference on all InD/OOD videos and compute unsupervised metrics
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # optional: run eks on all InD/OOD videos
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # select frames to add to the dataset
    # -------------------------------------------------------------------------------------
    print(
        f'selecting {cfg["n_pseudo_labels"]} pseudo-labels using {cfg["pseudo_labeler"]} '
        f'({cfg["selection_strategy"]} strategy)'
    )

    # train model(s) on expanded dataset
    print(f'training {cfg["n_ensemble_final"]} baseline models')

    # -------------------------------------------------------------------------------------
    # run inference on all InD/OOD videos and compute unsupervised metrics
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # save out all predictions/metrics in dataframe(s)
    # -------------------------------------------------------------------------------------


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        help='absolute path to .yaml configuration file',
        type=str,
    )
    args = parser.parse_args()
    pipeline(args.config)
