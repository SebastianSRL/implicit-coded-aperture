#!/usr/bin/env python
import os
import argparse

import torch.optim
import torch.utils.data
import numpy as np

from itertools import product
from contextlib import redirect_stdout
from pathlib import Path
from core import M2S, Dataset
from config.default import get_cfg_defaults


# config file path
# activate gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"Torch version: {torch.__version__}")


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default=None)
    return args.parse_args()


def main(cfg):
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    if cfg.DECODER.NAME == "dssp":
        cfg.defrost()
        cfg.merge_from_list(["ENCODER.STRIDES", 1])
        cfg.freeze()

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # path configurations
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    # # Paths and conditions
    result_path = cfg.TRAIN.RESULT_DIR
    save_path = cfg.TRAIN.SAVE_DIR
    save_path = f"{result_path}/{save_path}_{cfg.MODEL.NAME}_no_noise"

    if cfg.ENCODER.TRAINABLE:
        if cfg.INR.USE:
            save_path = save_path + f"_trainable_mask_{cfg.INR.NAME}"
        else:
            save_path = save_path + f"_trainable_mask_baseline"
    else:
        save_path = save_path + "_random_mask"

    print(f"Experiment will be saved in {save_path}")

    checkpoint_path = f"{save_path}/checkpoints"
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # load dataset
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    # Save the specified configuration
    with open(f"{save_path}/test.yml", "w") as f:
        with redirect_stdout(f):
            print(cfg.dump())

    dataset = Dataset(
        cfg.TRAIN.DATASET_DIR,
        cfg.TRAIN.BATCH_SIZE,
        cfg.ENCODER.PATCH_SIZE,
        cfg.TRAIN.WORKERS,
    )

    train_loader, val_loader = dataset.get_arad_dataset(gen_dataset_path=None)

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # load model and hyperparams
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    m2s_model = M2S(
        cfg=cfg,
        save_path=save_path,
        device=device,
        input_shape=cfg.DECODER.SHAPE,
        model=cfg.MODEL.NAME,
        lr=cfg.DECODER.LR,
    )

    # summary model
    num_parameters = sum(
        [l.nelement() for l in m2s_model.computational_decoder.parameters()]
    )
    print(f"Number of parameters: {num_parameters}")

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # load checkpoint
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    if cfg.TRAIN.WEIGHTS_DIR:
        m2s_model.load_checkpoint(
            cfg.cfg.TRAIN.WEIGHTS_DIR, epoch=cfg.TRAIN.INIT_EPOCHS
        )
        print("¡Model checkpoint loaded correctly!")

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # Train
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    m2s_model.train(
        train_loader, cfg.TRAIN.INIT_EPOCHS, cfg.TRAIN.EPOCHS, val_loader=val_loader[0]
    )


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    args = get_args()
    if args.config:
        cfg.merge_from_file(args.config)
    cfg.freeze()
    main(cfg)
