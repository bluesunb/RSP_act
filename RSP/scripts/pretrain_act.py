import datetime
import os
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np
import optax
from boxprint import bprint
from flax.jax_utils import pad_shard_unpad, replicate, unreplicate
from tqdm import tqdm

import src.common.logger as logger
from src.common.context import load_model, make_rngs, prepare_ckpt, save_model
from src.common.imports import resolve_import
from src.common.log_writer import LogWriter
from src.common.metrics import Average, MultiMetric
from src.common.tf_loader import tf_cross_file_batch_loader
from src.common.train_state import TrainState
from src.common.utils import func_timeit, reduce_array_to_scalar, tabulate_info
from src.RSP.config import RSPConfig, config_diff
from src.RSP.datasets import img_inverse_normalize, sample_furniture_act_fn, sample_kitchen_act_fn
from src.RSP.rsp_act import RNG_KEYS, RSP, action_recon_loss, kl_dist_loss, unpatchify
from src.RSP.evaluation import evaluate_action, record_video

Array = np.ndarray | jax.Array


def plot_extra_info(extra_info: Dict[str, jp.ndarray]):
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    axes = axes.flatten()
    for ax, (key, value) in zip(axes, extra_info.items()):
        fig.colorbar(ax.pcolormesh(value[0]))
        ax.set_title(key)
    
    plt.tight_layout()

    # figure to image
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


def get_sample_input(cfg: RSPConfig) -> Dict[str, jax.Array]:
    return {
        "src_img": jp.zeros((cfg.batch_size, cfg.input_size, cfg.input_size, 3), dtype=jp.float32),
        "tgt_img": jp.zeros((cfg.batch_size, cfg.input_size, cfg.input_size, 3), dtype=jp.float32),
    }


@func_timeit
def prepare_dataset_logger(cfg: RSPConfig):
    # Prepare dataset
    if cfg.data_type == "furniture":
        sample_fn = sample_furniture_act_fn(cfg)
        files = sorted(list(Path(cfg.data_dir).rglob("*_stacked.pkl")))
    elif cfg.data_type == "kitchen":
        sample_fn = sample_kitchen_act_fn(cfg)
        files = sorted(list(Path(cfg.data_dir).rglob("*.pkl")))
    else:
        raise NotImplementedError(f"Data type {cfg.data_type} is not supported.")

    loader = tf_cross_file_batch_loader(
        files, sample_fn, batch_size=cfg.batch_size, drop_last=cfg.drop_last, shuffle=cfg.shuffle_files
    )
    
    # Set trajectory info
    tmp_batch = next(iter(loader))
    cfg.set_traj_info(act_size=tmp_batch[2].shape[-1], seq_len=cfg.max_distance)

    # Prepare logger
    if cfg.log_freq > 0:
        assert cfg.exp_name is not None, "exp_name must be provided for logging."
        cfg.version = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_writer = LogWriter(cfg)
    else:
        log_writer = None

    os.makedirs(cfg.save_dir, exist_ok=True)
    cfg.save()
    return loader, log_writer


@func_timeit
def prepare_model_params(cfg: RSPConfig, resume: bool = False):
    rng = jax.random.PRNGKey(cfg.seed)
    model: RSP = resolve_import(cfg.model)(
        act_size=cfg.act_size,
        seq_len=cfg.seq_len,
        img_size=cfg.input_size,
        stoch=cfg.stoch,
        discrete=cfg.discrete,
        mask_rate=cfg.mask_rate,
        noise_scale=cfg.noise_scale,
    )
    assert model.patch_size == cfg.patch_size, f"Patch size mismatch: {model.patch_size} != {cfg.patch_size}"
    rng, rngs = make_rngs(rng, RNG_KEYS)

    if resume:
        tmp_manager = prepare_ckpt(Path(cfg.save_dir) / "ckpt", monitor="loss", best_mode="min", keep_n=5)
        variables = load_model(tmp_manager, epoch_idx=cfg.resume_epoch)
        params = variables.pop("params")
        extra_variables = variables.pop("extra_variables")
        return model, params, extra_variables

    example_input = get_sample_input(cfg)
    variables = model.init({"params": rng, **rngs}, **example_input, train=True)
    params = variables.pop("params")
    return model, params, variables


@func_timeit
def prepare_optimizer_scheduler(cfg: RSPConfig, params: Dict[str, jp.ndarray]):
    @optax.inject_hyperparams
    def create_tx(schedule):
        return optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(schedule, b1=0.9, b2=0.95, weight_decay=cfg.weight_decay),
        )

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        decay_steps=cfg.train_steps,
        end_value=cfg.min_lr,
    )

    tx = create_tx(schedule)
    return tx


def maybe_reduce(pytree):
    pytree = reduce_array_to_scalar(unreplicate(pytree))
    return jax.tree.map(reduce_array_to_scalar, pytree)


def batch_to_inputs(batch):
    batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), batch)
    batch = dict(zip(["src_img", "tgt_img", "actions", "term_dist"], batch))
    return batch


def init_train_metrics():
    return MultiMetric.create(
        loss=Average("loss"),
        loss_post_act=Average("loss_post_act"),
        loss_prior_act=Average("loss_prior_act"),
        loss_post_term=Average("loss_post_term"),
        loss_prior_term=Average("loss_prior_term"),
        loss_kl=Average("loss_kl"),
        kl=Average("kl"),
    )


def calc_loss(batch: Dict[str, jp.ndarray], outputs: Dict[str, jp.ndarray], cfg: RSPConfig) -> Dict[str, jp.ndarray]:
    kl_loss, kl_value = kl_dist_loss(
        outputs["post_dist"], outputs["prior_dist"], freebit=cfg.kl_freebit, balance=cfg.kl_balance
    )
    post_act_loss = action_recon_loss(batch["actions"], outputs["tgt_pred_post_act"])
    prior_act_loss = action_recon_loss(batch["actions"], outputs["tgt_pred_prior_act"])
    post_term_loss = action_recon_loss(batch["term_dist"], outputs["tgt_pred_post_term"])
    prior_term_loss = action_recon_loss(batch["term_dist"], outputs["tgt_pred_prior_term"])
    loss = post_act_loss + post_term_loss + cfg.kl_scale * kl_loss

    loss_info = {
        "loss": loss,
        "loss_post_act": post_act_loss,
        "loss_prior_act": prior_act_loss,
        "loss_post_term": post_term_loss,
        "loss_prior_term": prior_term_loss,
        "loss_kl": kl_loss,
        "kl": kl_value,
    }

    extra_info = {
        "tgt_orig_post_act": batch["actions"],
        "tgt_pred_post_act": outputs["tgt_pred_post_act"],
        "tgt_pred_prior_act": outputs["tgt_pred_prior_act"],
        "tgt_diff_post_act": jp.abs(outputs["tgt_pred_post_act"] - outputs["tgt_pred_post_term"]),
        "tgt_diff_prior_act": jp.abs(outputs["tgt_pred_prior_act"] - outputs["tgt_pred_prior_term"]),
        "tgt_orig_post_term": batch["term_dist"],
        "tgt_pred_post_term": outputs["tgt_pred_post_term"],
        "tgt_pred_prior_term": outputs["tgt_pred_prior_term"],
        "tgt_diff_post_term": jp.abs(outputs["tgt_pred_post_term"] - batch["term_dist"]),
        "tgt_diff_prior_term": jp.abs(outputs["tgt_pred_prior_term"] - batch["term_dist"]),
    }
    
    return loss, (loss_info, extra_info)


def train_step(
    state: TrainState,
    batch: Dict[str, jp.ndarray],
    metrics: MultiMetric,
    cfg: RSPConfig,
    pmap_axis: str = None
) -> Tuple[TrainState, MultiMetric, Dict[str, float]]:
    rng, step_rng = jax.random.split(state.rng)
    step_rng, rngs = make_rngs(step_rng, RNG_KEYS)

    def loss_fn(params):
        outputs, updates = state(batch["src_img"], batch["tgt_img"], train=True, params=params, rngs=rngs, mutable=["pos_emb"])
        loss, (loss_info, extra_info) = calc_loss(batch, outputs, cfg)
        return loss, (updates, loss_info, extra_info)

    grads, (updates, loss_info, extra_info) = jax.grad(loss_fn, has_aux=True)(state.params)
    grads = jax.lax.pmean(grads, axis_name=pmap_axis)
    loss_info = jax.lax.pmean(loss_info, axis_name=pmap_axis)
    metrics = metrics.update(**loss_info)
    state = state.apply_gradients(grads=grads, extra_variables=updates, rng=rng)

    print_info = {"loss": loss_info["loss"]}
    return state, metrics, print_info, loss_info, extra_info


def train(
    cfg: RSPConfig,
    train_step_fn: callable,
    state: TrainState,
    dataloader: Iterable,
    num_steps: int,
    train_metrics: MultiMetric,
    log_writer: Optional[LogWriter] = None,
    ckpt_manager=None,
):
    pbar = tqdm(enumerate(islice(dataloader, num_steps + 2), start=1), total=num_steps, position=0)
    for step, batch in pbar:
        batch = batch_to_inputs(batch)
        state, train_metrics, print_info, loss_info, extra_info = train_step_fn(state, batch, train_metrics)
        print_info = maybe_reduce(print_info)
        loss_info = maybe_reduce(loss_info)
        extra_info = maybe_reduce(extra_info)
        pbar.set_postfix(print_info)

        if log_writer is not None and step % cfg.log_freq == 0:
            step_metrics = maybe_reduce(train_metrics).compute()
            for key, value in step_metrics.items():
                log_writer.log_scalar(f"loss/{key}", reduce_array_to_scalar(value), step)

            if step % (cfg.log_freq * 10) == 0:
                imgs = plot_extra_info(extra_info)
                log_writer.log_image("extra_info", imgs, step)
            
            log_writer.log_scalar("lr", state.opt_state.hyperparams["schedule"][0].item(), step)

        if log_writer is not None and (step==1 or step % cfg.eval_freq == 0 or step == num_steps):
            policy_fn = jax.jit(unreplicate(state).__call__, static_argnames=["method"])
            ep_records = evaluate_action("kitchen-mixed-v0", policy_fn, num_episodes=1, render=True, seed=cfg.seed)
            # frames = ep_records["frames"][0].transpose(0, 2, 3, 1)
            clip = record_video(ep_records)
            clip.write_videofile(f"tmp.mp4")
            log_writer.log_video("video", "tmp.mp4", step)

        if ckpt_manager is not None and (step % cfg.save_freq == 0 or step == num_steps):
            step_metrics = maybe_reduce(train_metrics).compute()
            save_model(ckpt_manager, unreplicate(state), step_metrics, step)

    return state, train_metrics


def main(cfg: RSPConfig):
    n_devices = jax.device_count()
    rng = jax.random.PRNGKey(cfg.seed)

    effective_batch_size = cfg.batch_size * cfg.accum_iter * n_devices
    if cfg.lr is None:
        cfg.lr = cfg.base_lr * effective_batch_size / 256

    loader, log_writer = prepare_dataset_logger(cfg)
    model, params, extra_variables = prepare_model_params(cfg)
    tx = prepare_optimizer_scheduler(cfg, params)
    ckpt_manager = prepare_ckpt(Path(cfg.save_dir) / "ckpt", monitor="loss", best_mode="min", keep_n=5)

    logger.info()
    bprint(
        f"\n{'Model':<40} {model.__class__.__name__}\n"
        + f"{'Device count':<40} {n_devices}\n"
        + f"{'Log dir':<40} {cfg.save_dir}\n"
        + "\n".join(f"{k:<40} {v}" for k, v in config_diff(cfg).items())
        + "\n"
        + f"{'Effective Batch Size':<40} {effective_batch_size}\n"
        + f"{'Base Learning Rate':<40} {cfg.lr * 256 / effective_batch_size:.2e}\n"
        + f"{'Actual Learning Rate':<40} {cfg.lr:.2e}",
        title="Configurations",
        width=100,
        print_func=logger.info,
    )

    state = TrainState.create(model_def=model, params=params, tx=tx, extra_variables=extra_variables, rng=rng)

    rng, rngs = make_rngs(rng, RNG_KEYS)
    rngs = {"params": rng, **rngs}
    tabulate_info(params, Path(cfg.save_dir))

    state = replicate(state)
    train_metrics = replicate(init_train_metrics())
    axis_name = "batch"
    p_train_step = jax.pmap(partial(train_step, cfg=cfg, pmap_axis=axis_name), axis_name=axis_name)
    p_train_step = pad_shard_unpad(p_train_step, static_argnums=(0, 2), static_return=True)

    state, train_metrics = train(
        cfg=cfg,
        train_step_fn=p_train_step,
        state=state,
        dataloader=loader,
        num_steps=cfg.train_steps,
        train_metrics=train_metrics,
        log_writer=log_writer,
        ckpt_manager=ckpt_manager,
    )

    if log_writer is not None:
        log_writer.finalize("success")

    return state


if __name__ == "__main__":
    logger.set_log_level("DEBUG")

    train_steps = 15000
    warmup_steps = int(train_steps * 0.10)
    save_freq = train_steps // 10
    eval_freq = 1500

    cfg = RSPConfig(
        model="src.RSP.rsp_act.rsp_vit_tiny_patch16",
        data_type="kitchen",
        data_dir=str(Path.home() / "Datasets/kitchen/mixed"),
        exp_name="kitchen-act",
        base_lr=1.5e-4 * 1,
        train_steps=train_steps,
        warmup_steps=warmup_steps,
        log_freq=10,
        eval_freq=eval_freq,
        save_freq=save_freq,
        min_distance=8,
        max_distance=48,
        input_size=64,
        patch_size=16,
        batch_size=32,
        repeated_sampling=2,
        shuffle_files=True,
    )

    main(cfg)
