import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Tuple

import cv2
import numpy as np
import tensorflow as tf

from src.RSP.config import RSPConfig


@dataclass
class RSPDataConfig:
    img_version: Literal["fpp", "tpp", "both"] = "tpp"
    min_distance: int = 4
    max_distance: int = 48
    repeated_sampling: int = 2
    img_size: tuple = (224, 224)
    flip_p = float = 0.5
    scale: tuple = (0.5, 1.0)
    ratio: tuple = (3.0 / 4.0, 4.0 / 3.0)


def paired_random_resized_crop(
    flip_p: float = 0.5,
    size: tuple = (224, 224),
    scale: tuple = (0.5, 1.0),
    ratio: tuple = (3.0 / 4.0, 4.0 / 3.0),
):
    def transform(img1: np.ndarray, img2: np.ndarray):
        # Image dimensions
        img_h, img_w = img1.shape[:2]

        # Randomly determine the scale and aspect ratio
        area = img_h * img_w
        target_area = random.uniform(*scale) * area
        aspect_ratio = random.uniform(*ratio)

        # Compute the width and height of the cropped image
        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        # Ensure the crop size is within the image dimensions
        w = max(1, min(w, img_w))
        h = max(1, min(h, img_h))

        # Randomly select the top-left corner of the crop
        i = random.randint(0, img_h - h)
        j = random.randint(0, img_w - w)

        # Perform the crop
        img1 = img1[i : i + h, j : j + w, :]
        img2 = img2[i : i + h, j : j + w, :]

        # Resize to the desired size
        img1 = cv2.resize(img1, size)
        img2 = cv2.resize(img2, size)

        # Randomly flip the images
        if random.random() < flip_p:
            img1 = np.fliplr(img1)
            img2 = np.fliplr(img2)

        return img1, img2

    return transform


def img_normalize(img: np.ndarray):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img / 255.0
    return (img - mean) / std


def img_inverse_normalize(img: np.ndarray):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img


def load_frames(imgs: np.ndarray, min_distance: int, max_distance: int):
    seq_len = len(imgs)
    least_frames_num = max_distance + 1
    if seq_len >= least_frames_num:
        idx_cur = random.randint(0, seq_len - least_frames_num)
        interval = random.randint(min_distance, max_distance)
        idx_fut = idx_cur + interval
    else:
        ids = random.sample(range(seq_len), 2)
        idx_cur, idx_fut = min(ids), max(ids)

    return imgs[idx_cur], imgs[idx_fut]


def load_frames_act(imgs: np.ndarray, acts: np.ndarray, min_distance: int, max_distance: int):
    seq_len = len(imgs)
    least_frames_num = max_distance + 1
    if seq_len >= least_frames_num:
        idx_cur = random.randint(0, seq_len - least_frames_num)
        interval = random.randint(min_distance, max_distance)
        idx_fut = idx_cur + interval
    else:
        ids = random.sample(range(seq_len), 2)
        idx_cur, idx_fut = min(ids), max(ids)

    acts = acts[idx_cur:idx_fut]
    acts = np.pad(acts, ((0, max_distance - len(acts)), (0, 0)), mode="constant", constant_values=0)
    term_dist = np.concatenate([
        np.linspace(1, 0, idx_fut - idx_cur, endpoint=False),
        np.zeros(max_distance - (idx_fut - idx_cur)),
    ])
    return imgs[idx_cur], imgs[idx_fut], acts, term_dist[..., None]


def process_file(
    load_fn: Callable[[str | Path], np.ndarray],
    transform_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] = paired_random_resized_crop(),
    min_distance: int = 4,
    max_distance: int = 48,
    repeated_sampling: int = 2,
):
    def process_fn(file: str):
        imgs = load_fn(file)

        src_imgs = []
        tgt_imgs = []
        for _ in range(repeated_sampling):
            src_img, tgt_img = load_frames(imgs, min_distance, max_distance)
            src_img, tgt_img = transform_fn(src_img, tgt_img)
            src_imgs.append(img_normalize(src_img))
            tgt_imgs.append(img_normalize(tgt_img))

        src_imgs = np.stack(src_imgs, axis=0).astype(np.float32)
        tgt_imgs = np.stack(tgt_imgs, axis=0).astype(np.float32)
        return src_imgs, tgt_imgs

    return process_fn


def process_file_act(
    load_fn: Callable[[str | Path], np.ndarray],
    transform_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] = paired_random_resized_crop(),
    min_distance: int = 4,
    max_distance: int = 48,
    repeated_sampling: int = 2,
):
    def process_fn(file: str):
        imgs, acts = load_fn(file)
        src_imgs = []
        tgt_imgs = []
        action_seqs = []
        term_seqs = []

        for _ in range(repeated_sampling):
            src_img, tgt_img, act_history, term_dist = load_frames_act(imgs, acts, min_distance, max_distance)
            src_img, tgt_img = transform_fn(src_img, tgt_img)
            src_imgs.append(img_normalize(src_img))
            tgt_imgs.append(img_normalize(tgt_img))
            action_seqs.append(act_history)
            term_seqs.append(term_dist)

        src_imgs = np.stack(src_imgs, axis=0).astype(np.float32)
        tgt_imgs = np.stack(tgt_imgs, axis=0).astype(np.float32)
        action_seqs = np.stack(action_seqs, axis=0).astype(np.float32)
        term_seqs = np.stack(term_seqs, axis=0).astype(np.float32)
        return src_imgs, tgt_imgs, action_seqs, term_seqs

    return process_fn


def create_sample_fn(
    cfg: RSPDataConfig,
    load_fn: Callable[[str], np.ndarray],
    transform_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] = None,
):
    if transform_fn is None:
        transform_fn = paired_random_resized_crop(cfg.flip_p, cfg.img_size, cfg.scale, cfg.ratio)
    process_fn = process_file(load_fn, transform_fn, cfg.min_distance, cfg.max_distance, cfg.repeated_sampling)

    def sample_fn(file: str):
        src_imgs, tgt_imgs = tf.numpy_function(process_fn, inp=[file], Tout=[tf.float32, tf.float32])
        src_imgs.set_shape([cfg.repeated_sampling, *cfg.img_size, 3])
        tgt_imgs.set_shape([cfg.repeated_sampling, *cfg.img_size, 3])
        return src_imgs, tgt_imgs

    return sample_fn


def create_sample_act_fn(
    cfg: RSPDataConfig,
    load_fn: Callable[[str], np.ndarray],
    transform_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] = None,
):
    if transform_fn is None:
        transform_fn = paired_random_resized_crop(cfg.flip_p, cfg.img_size, cfg.scale, cfg.ratio)
    process_fn = process_file_act(load_fn, transform_fn, cfg.min_distance, cfg.max_distance, cfg.repeated_sampling)

    def sample_fn(file: str):
        out_types = [tf.float32, tf.float32, tf.float32, tf.float32]
        src_imgs, tgt_imgs, action_seqs, term_seqs = tf.numpy_function(process_fn, inp=[file], Tout=out_types)
        src_imgs.set_shape([cfg.repeated_sampling, *cfg.img_size, 3])
        tgt_imgs.set_shape([cfg.repeated_sampling, *cfg.img_size, 3])
        return src_imgs, tgt_imgs, action_seqs, term_seqs

    return sample_fn


def sample_furnuture_fn(cfg: RSPConfig):
    def load_fn(file: str):
        with open(file, "rb") as f:
            data = pickle.load(f)

        buffer = []
        if cfg.img_version in ("fpp", "both"):
            buffer.append(data["observations"]["color_image1"])
        if cfg.img_version in ("tpp", "both"):
            buffer.append(data["observations"]["color_image2"])
        data = np.concatenate(buffer, axis=-3)
        return data

    cfg = RSPDataConfig(
        img_version=cfg.img_version,
        min_distance=cfg.min_distance,
        max_distance=cfg.max_distance,
        repeated_sampling=cfg.repeated_sampling,
        img_size=(cfg.input_size, cfg.input_size),
    )

    return create_sample_fn(cfg, load_fn)


def sample_kitchen_fn(cfg: RSPConfig):
    def load_fn(file: str):
        with open(file, "rb") as f:
            data = pickle.load(f)
        return data["observations"]

    cfg = RSPDataConfig(
        img_version=cfg.img_version,
        min_distance=cfg.min_distance,
        max_distance=cfg.max_distance,
        repeated_sampling=cfg.repeated_sampling,
        img_size=(cfg.input_size, cfg.input_size),
    )
    return create_sample_fn(cfg, load_fn)


def sample_furniture_act_fn(cfg: RSPConfig):
    def load_fn(file: str):
        with open(file, "rb") as f:
            data = pickle.load(f)

        buffer = []
        if cfg.img_version in ("fpp", "both"):
            buffer.append(data["observations"]["color_image1"])
        if cfg.img_version in ("tpp", "both"):
            buffer.append(data["observations"]["color_image2"])
        data = np.concatenate(buffer, axis=-3)
        return data, data["actions"]

    cfg = RSPDataConfig(
        img_version=cfg.img_version,
        min_distance=cfg.min_distance,
        max_distance=cfg.max_distance,
        repeated_sampling=cfg.repeated_sampling,
        img_size=(cfg.input_size, cfg.input_size),
    )
    return create_sample_act_fn(cfg, load_fn)


def sample_kitchen_act_fn(cfg: RSPConfig):
    def load_fn(file: str):
        with open(file, "rb") as f:
            data = pickle.load(f)
        return data["observations"], data["actions"]

    cfg = RSPDataConfig(
        img_version=cfg.img_version,
        min_distance=cfg.min_distance,
        max_distance=cfg.max_distance,
        repeated_sampling=cfg.repeated_sampling,
        img_size=(cfg.input_size, cfg.input_size),
    )
    return create_sample_act_fn(cfg, load_fn)


if __name__ == "__main__":
    from itertools import islice

    from tqdm import tqdm

    from src.common.tf_loader import tf_cross_file_batch_loader

    cfg = RSPDataConfig(
        img_version="tpp",
        min_distance=4,
        max_distance=48,
        repeated_sampling=2,
        img_size=(64, 64),
    )

    root = Path.home() / "scripted_sim_demo/kitchen/mixed"
    sample_fn = sample_kitchen_fn(cfg)
    loader = tf_cross_file_batch_loader(
        files=sorted(list(root.glob("*.pkl"))), sample_fn=sample_fn, batch_size=64, drop_last=True, shuffle=True
    )

    loader = loader.as_numpy_iterator()
    pbar = tqdm(islice(loader, 50))
    for batch in pbar:
        src_imgs, tgt_imgs = batch
        pbar.set_description(f"src_imgs: {src_imgs.shape}, tgt_imgs: {tgt_imgs.shape}")

    print("Done!")
