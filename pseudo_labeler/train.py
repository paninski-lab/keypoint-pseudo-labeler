"""Functions for training."""

import gc
import os
import random
import shutil
from typing import Optional

import lightning.pytorch as pl
import numpy as np
import torch
from lightning_pose.utils import pretty_print_cfg, pretty_print_str
from lightning_pose.utils.io import (
    check_video_paths,
    return_absolute_data_paths,
    return_absolute_path,
)
from lightning_pose.utils.predictions import predict_dataset, predict_single_video
from lightning_pose.utils.scripts import (
    calculate_train_batches,
    compute_metrics,
    get_callbacks,
    get_data_module,
    get_dataset,
    get_imgaug_transform,
    get_loss_factories,
    get_model,
)
from moviepy.editor import VideoFileClip
from omegaconf import DictConfig, OmegaConf


def train(cfg: DictConfig, results_dir: str) -> None:

    # mimic hydra, change dir into results dir
    pwd = os.getcwd()
    os.makedirs(results_dir, exist_ok=True)
    os.chdir(results_dir)

    # reset all seeds
    seed = 0
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # ----------------------------------------------------------------------------------
    # set up data/model objects
    # ----------------------------------------------------------------------------------

    pretty_print_cfg(cfg)

    data_dir, video_dir = return_absolute_data_paths(data_cfg=cfg.data)

    # imgaug transform
    imgaug_transform = get_imgaug_transform(cfg=cfg)

    # dataset
    dataset = get_dataset(cfg=cfg, data_dir=data_dir, imgaug_transform=imgaug_transform)

    # datamodule; breaks up dataset into train/val/test
    data_module = get_data_module(cfg=cfg, dataset=dataset, video_dir=video_dir)

    # build loss factory which orchestrates different losses
    loss_factories = get_loss_factories(cfg=cfg, data_module=data_module)

    # model
    model = get_model(cfg=cfg, data_module=data_module, loss_factories=loss_factories)

    # ----------------------------------------------------------------------------------
    # set up and run training
    # ----------------------------------------------------------------------------------

    # logger
    logger = pl.loggers.TensorBoardLogger("tb_logs", name=cfg.model.model_name)

    # early stopping, learning rate monitoring, model checkpointing, backbone unfreezing
    callbacks = get_callbacks(cfg, early_stopping=False)

    # calculate number of batches for both labeled and unlabeled data per epoch
    limit_train_batches = calculate_train_batches(cfg, dataset)

    # set up trainer
    # TODO: !! we will likely need to update this to be based on steps rather than epochs
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.training.max_epochs,
        min_epochs=cfg.training.min_epochs,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        log_every_n_steps=cfg.training.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        limit_train_batches=limit_train_batches,
    )

    # train model!
    trainer.fit(model=model, datamodule=data_module)

    # save config file
    cfg_file_local = os.path.join(results_dir, "config.yaml")
    with open(cfg_file_local, "w") as fp:
        OmegaConf.save(config=cfg, f=fp.name)

    # ----------------------------------------------------------------------------------
    # post-training analysis: labeled frames
    # ----------------------------------------------------------------------------------
    hydra_output_directory = os.getcwd()
    print("Hydra output directory: {}".format(hydra_output_directory))
    # get best ckpt
    best_ckpt = os.path.abspath(trainer.checkpoint_callback.best_model_path)
    # check if best_ckpt is a file
    if not os.path.isfile(best_ckpt):
        raise FileNotFoundError("Cannot find checkpoint. Have you trained for too few epochs?")

    # make unaugmented data_loader if necessary
    if cfg.training.imgaug != "default":
        cfg_pred = cfg.copy()
        cfg_pred.training.imgaug = "default"
        imgaug_transform_pred = get_imgaug_transform(cfg=cfg_pred)
        dataset_pred = get_dataset(
            cfg=cfg_pred, data_dir=data_dir, imgaug_transform=imgaug_transform_pred)
        data_module_pred = get_data_module(
            cfg=cfg_pred, dataset=dataset_pred, video_dir=video_dir)
        data_module_pred.setup()
    else:
        data_module_pred = data_module

    # predict on all labeled frames (train/val/test)
    pretty_print_str("Predicting train/val/test images...")
    # compute and save frame-wise predictions
    preds_file = os.path.join(hydra_output_directory, "predictions.csv")
    predict_dataset(
        cfg=cfg,
        trainer=trainer,
        model=model,
        data_module=data_module_pred,
        ckpt_file=best_ckpt,
        preds_file=preds_file,
    )
    # compute and save various metrics
    try:
        compute_metrics(cfg=cfg, preds_file=preds_file, data_module=data_module_pred)
    except Exception as e:
        print(f"Error computing metrics\n{e}")

    # ----------------------------------------------------------------------------------
    # post-training analysis: predict on OOD labeled frames
    # ----------------------------------------------------------------------------------
    # update config file to point to OOD data
    csv_file_ood = os.path.join(cfg.data.data_dir, cfg.data.csv_file).replace(".csv", "_new.csv")
    if os.path.exists(csv_file_ood):
        cfg_ood = cfg.copy()
        cfg_ood.data.csv_file = csv_file_ood
        cfg_ood.training.imgaug = "default"
        cfg_ood.training.train_prob = 1
        cfg_ood.training.val_prob = 0
        cfg_ood.training.train_frames = 1
        # build dataset/datamodule
        imgaug_transform_ood = get_imgaug_transform(cfg=cfg_ood)
        dataset_ood = get_dataset(
            cfg=cfg_ood, data_dir=data_dir, imgaug_transform=imgaug_transform_ood
        )
        data_module_ood = get_data_module(cfg=cfg_ood, dataset=dataset_ood, video_dir=video_dir)
        data_module_ood.setup()
        pretty_print_str("Predicting OOD images...")
        # compute and save frame-wise predictions
        preds_file_ood = os.path.join(hydra_output_directory, "predictions_new.csv")
        predict_dataset(
            cfg=cfg_ood,
            trainer=trainer,
            model=model,
            data_module=data_module_ood,
            ckpt_file=best_ckpt,
            preds_file=preds_file_ood,
        )
        # compute and save various metrics
        try:
            compute_metrics(
                cfg=cfg_ood, preds_file=preds_file_ood, data_module=data_module_ood
            )
        except Exception as e:
            print(f"Error computing metrics\n{e}")

    # ----------------------------------------------------------------------------------
    # post-training analysis: unlabeled videos
    # ----------------------------------------------------------------------------------
    if cfg.eval.predict_vids_after_training:
        pretty_print_str("Predicting videos...")
        if cfg.eval.test_videos_directory is None:
            filenames = []
        else:
            filenames = check_video_paths(return_absolute_path(cfg.eval.test_videos_directory))
            pretty_print_str(
                f"Found {len(filenames)} videos to predict on "
                f"(in cfg.eval.test_videos_directory)"
            )

        for v, video_file in enumerate(filenames):
            assert os.path.isfile(video_file)
            pretty_print_str(f"Predicting video: {video_file}...")
            # get save name for prediction csv file
            video_pred_dir = os.path.join(hydra_output_directory, "video_preds")
            video_pred_name = os.path.splitext(os.path.basename(video_file))[0]
            prediction_csv_file = os.path.join(video_pred_dir, video_pred_name + ".csv")
            inference_with_metrics(
                video_file=video_file,
                ckpt_file=best_ckpt,
                cfg=cfg,
                preds_file=prediction_csv_file,
                data_module=data_module_pred,
                trainer=trainer,
            )

    # ----------------------------------------------------------------------------------
    # clean up
    # ----------------------------------------------------------------------------------
    # remove lightning logs
    shutil.rmtree(os.path.join(results_dir, "lightning_logs"), ignore_errors=True)

    # change directory back
    os.chdir(pwd)

    # clean up memory
    del imgaug_transform
    del dataset
    del data_module
    del data_module_pred
    del loss_factories
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()


def inference_with_metrics(
    video_file: str,
    cfg: DictConfig,
    preds_file: str,
    ckpt_file: Optional[str] = None,
    data_module: Optional[callable] = None,
    trainer: Optional[pl.Trainer] = None,
    metrics: bool = True,
) -> pd.DataFrame:

    # update video size in config
    video_clip = VideoFileClip(video_file)
    cfg.data.image_orig_dims.width = video_clip.w
    cfg.data.image_orig_dims.height = video_clip.h

    # compute predictions if they don't already exist
    if not os.path.exists(preds_file):
        preds_df = predict_single_video(
            video_file=video_file,
            ckpt_file=ckpt_file,
            cfg_file=cfg,
            preds_file=preds_file,
            data_module=data_module,
            trainer=trainer,
        )
    else:
        preds_df = pd.read_csv(preds_file, header=[0, 1, 2], index_col=0)

    # compute and save various metrics
    if metrics:
        try:
            compute_metrics(cfg=cfg, preds_file=preds_file, data_module=data_module)
        except Exception as e:
            print(f"Error predicting on {video_file}:\n{e}")

    video_clip.close()
    del video_clip
    gc.collect()

    return preds_df
