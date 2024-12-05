from functools import partial
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import d4rl
import gym
import jax
import numpy as np

from src.common.context import load_model, make_rngs, prepare_ckpt
from src.RSP.config import RSPConfig
from src.RSP.rsp_act import RNG_KEYS, RSP, rsp_vit_tiny_patch16
from moviepy.editor import ImageSequenceClip
from src.RSP.datasets import img_normalize, img_inverse_normalize


def kitchen_render(kitchen_env, wh=64):
    from dm_control.mujoco import engine
    camera = engine.MovableCamera(kitchen_env.sim, wh, wh)
    camera.set_pose(distance=1.8, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)
    img = camera.render()
    return img


def evaluate_action(
    env_name: str,
    policy_fn: callable,
    num_episodes: int,
    render: bool = False,
    min_seq_len: int = 8,
    max_seq_len: int = 48,
    seed: int = 0,
    **kwargs,
):
    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)
    ep_records = defaultdict(list)

    obs = env.reset()
    obs_img = kitchen_render(env, wh=64)

    for e in range(num_episodes + 1):
        done = False
        step = 0
        obs = env.reset()
        record = defaultdict(list)
        pbar = tqdm(total=env._max_episode_steps + 10, desc=f"Episode {e}/{num_episodes}", leave=True)

        while not done:
            obs_img = kitchen_render(env, wh=64)
            
            if not record["action"]:
                action_prev = np.ones((max_seq_len, env.action_space.shape[0]))
            else:
                action_prev = np.stack(record["action"][-max_seq_len + 1:])
                action_prev = np.pad(action_prev, ((0, max_seq_len - action_prev.shape[0]), (0, 0)))
                assert action_prev.shape[0] == max_seq_len
                assert (action_prev[-1] == 0).all()
            
            rng, rngs = make_rngs(rng, RNG_KEYS)
            # pickle에서 가져온 img가 이미 0~1이고, obs_img가 0~255이면 nomalize 해도 결과가 달라질 수 있음
            imgs, actions = policy_fn(img_normalize(obs_img)[None], action_prev[None], rngs=rngs)
            actions = actions[0, :min_seq_len]

            record["src_img"].append(obs_img)
            record["tgt_img"].append(img_inverse_normalize(imgs[0]))
            
            for action in actions:
                next_obs, reward, done, info = env.step(action)
                record["action"].append(action)
                record["reward"].append(reward)
                record["done"].append(done)
            
                step += 1
                pbar.update(1)

                if render and e == num_episodes and step % 3 == 0:
                    frames = kitchen_render(env, wh=200).transpose(2, 0, 1)
                    record["frames"].append(frames)
                
                if done:
                    break
            
        record = {k: np.stack(v) for k, v in record.items()}
        for k, v in record.items():
            ep_records[k].append(v)
            
        # print(f"Steps: {step} | Reward: {record['reward'].sum()}")
        pbar.set_postfix({"Steps": step, "Reward": record["reward"].sum()})

    return ep_records


def record_video(ep_records):
    frames = ep_records["frames"][0].transpose(0, 2, 3, 1)
    frames = list(frames)
    clip = ImageSequenceClip(frames, fps=30)
    return clip


if __name__ == "__main__":
    from moviepy.editor import ImageSequenceClip

    save_dir = "output_dir/rsp_vit_tiny_patch16/kitchen-act/2024-11-29_12-35-27/ckpt"
    save_dir = Path(save_dir)

    cfg = RSPConfig.load(save_dir.parent)
    model = rsp_vit_tiny_patch16(
        act_size=cfg.act_size,
        seq_len=cfg.seq_len,
        img_size=cfg.input_size,
        stoch=cfg.stoch,
        discrete=cfg.discrete,
        mask_rate=cfg.mask_rate,
        noise_scale=cfg.noise_scale
    )

    tmp_manager = prepare_ckpt(save_dir, monitor="loss", best_mode="min", keep_n=5)
    state_dict = load_model(tmp_manager, epoch_idx=-1)
    state_dict.pop("metrics", None)

    variables = {"params": state_dict["params"], **state_dict.get("extra_variables", {})}
    rng = jax.random.PRNGKey(0)
    rng, rngs = make_rngs(rng, RNG_KEYS)

    policy_fn = jax.jit(partial(model.apply, variables), static_argnames=["method"])
    ep_records = evaluate_action(
        "kitchen-mixed-v0",
        policy_fn,
        num_episodes=4,
        render=True,
        seed=0
    )

    frames = ep_records["frames"][0].transpose(0, 2, 3, 1)
    frames = list(frames)
    # frames = frames.transpose(0, 2, 3, 1)
    # frames = [f for f in frames]
    clip = ImageSequenceClip(frames, fps=30)
    clip.write_videofile("output.mp4")