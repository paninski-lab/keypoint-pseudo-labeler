"""Example pipeline script."""

import argparse

import yaml
from omegaconf import DictConfig

from pseudo_labeler.train import train


def pipeline(config_file: str):

    # load pipeline config file
    cfg = yaml.safe_load(open(config_file, "r"))

    # -------------------------------------------------------------------------------------
    # train k supervised models on n hand-labeled frames and compute labeled OOD metrics
    # -------------------------------------------------------------------------------------
    print(f'training {len(cfg["init_ensemble_seeds"])} baseline models')
    for k in cfg["init_ensemble_seeds"]:
        # load lightning pose config file
        cfg_lp = use omega conf to load
        # - update data.data_dir, maybe some other paths
        # - update training.rng_seed_data_pt
        # - add iteration-specific fields to the config

        # define the output directory
        results_dir = todo
        """
        mirror-mouse/100_1000-eks-random/rng0
        mirror-mouse/100_1000-eks-random/rng1
        mirror-mouse/100_1000-eks-random/eks
        ...
        mirror-mouse/100_10000-eks-strategy2/rng0
        """

        # train model
        # if we run inference on videos inside train(), then we should pass a list of video
        # directories to loop over; these should probably be stored in pipeline config file
        train(cfg=cfg_lp, results_dir=results_dir)

    # -------------------------------------------------------------------------------------
    # run inference on all InD/OOD videos and compute unsupervised metrics
    # -------------------------------------------------------------------------------------
    # this is actually already in the train function - do we want to split it?

    # -------------------------------------------------------------------------------------
    # optional: run eks on all InD/OOD videos
    # -------------------------------------------------------------------------------------
    # keemin is the expert here
    # again, will need to be careful with directory organization

    # -------------------------------------------------------------------------------------
    # select frames to add to the dataset
    # -------------------------------------------------------------------------------------
    print(
        f'selecting {cfg["n_pseudo_labels"]} pseudo-labels using {cfg["pseudo_labeler"]} '
        f'({cfg["selection_strategy"]} strategy)'
    )
    # - we need to add frames to the existing dataset
    # - for each strategy/run/whatever, need to make a new csv file with updated frames

    # train model(s) on expanded dataset
    print(f'training {len(cfg["final_ensemble_seeds"])} baseline models')
    for k in cfg["final_ensemble_seeds"]:
        # load lightning pose config file
        cfg_lp = use omega conf to load
        # - update data.data_dir, maybe some other paths
        # - update training.rng_seed_data_pt
        # - add iteration-specific fields to the config
        # - NEW: change labeled data field (data.csv_file)

        # define the output directory
        results_dir = todo

        # train model
        # if we run inference on videos inside train(), then we should pass a list of video
        # directories to loop over; these should probably be stored in pipeline config file
        train(cfg=cfg_lp, results_dir=results_dir)

    # -------------------------------------------------------------------------------------
    # run inference on all InD/OOD videos and compute unsupervised metrics
    # -------------------------------------------------------------------------------------
    # do as above

    # -------------------------------------------------------------------------------------
    # save out all predictions/metrics in dataframe(s)
    # -------------------------------------------------------------------------------------
    # can think about this later


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
