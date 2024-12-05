from typing import Dict, Optional

import gym
import numpy as np
import src.common.logger as logger


def make_env(env_name: str):
    import d4rl

    print(d4rl.__version__)
    wrapped_env = gym.make(env_name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = env_name
    return env


def qlearning_dataset(
    env: gym.Env,
    dataset: Optional[Dict[str, np.ndarray]] = None,
    terminate_on_end: bool = False,
    disable_goal: bool = False,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    1. Get the dataset from the environment
    2. Extract the observations, actions, next_observations, rewards, terminals, and goals (if available)
    3. If the dataset has timeouts, extract the final timesteps
    4. Exclude the last timestep if terminate_on_end is False

    Args:
        env:                The environment
        dataset:            The dataset
        terminate_on_end:   If False, exclude the last timestep that terminates the episode.
                            This is useful for q-learning that uses the both the current and next states,
                            so the ambigous last timestep is not used.
        disable_goal:       Whether to disable the goal

    Returns:
        Dict[str, np.ndarray]: The dataset expected to be used for q-learning
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    goals = dataset["infos/goal"] if not disable_goal and "infos/goal" in dataset else None
    use_timeouts = "timeouts" in dataset
    if use_timeouts:
        final_timesteps = dataset["timeouts"][:-1]
    else:
        max_ep_steps = env._max_episode_steps
        final_timesteps = np.zeros_like(dataset["rewards"])
        final_timesteps[max_ep_steps - 1 :: max_ep_steps] = True

    dataset = {
        "observations": dataset["observations"][:-1].astype(np.float32),
        "actions": dataset["actions"][:-1].astype(np.float32),
        "next_observations": dataset["observations"][1:].astype(np.float32),
        "rewards": dataset["rewards"][:-1].astype(np.float32),
        "terminals": dataset["terminals"][:-1].astype(bool),
    }  # Drop last step to avoid accessing undefined next MDP component

    if goals is not None:
        dataset["goals"] = goals[:-1].astype(np.float32)

    if not terminate_on_end:
        dataset = {k: v[~final_timesteps] for k, v in dataset.items()}

    return dataset


def prepare_dataset(
    # env: gym.Env,
    env_name: str,
    clip_to_eps: bool = True,
    eps: float = 1e-5,
    dataset: Optional[Dict[str, np.ndarray]] = None,
    filter_terminals: bool = True,
    disable_goal: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Make sure the dataset is ready for training by:
    1. Loading the dataset if not provided
    2. Clipping the rewards to [-1 + eps, 1 - eps]
    3. Removing the ambigous terminal state if needed

    Args:
        env_name:           The environment name
        clip_to_eps:        Whether to clip the rewards to [-1 + eps, 1 - eps]
        eps:                The epsilon value for clipping
        dataset:            The dataset
        filter_terminals:   Whether to remove the terminal states
        disable_goal:       Whether to disable the goal
    """
    if dataset is None:
        logger.info(f"Loading dataset for {env_name}")
        # dataset: final timestep is filtered out
        env = make_env(env_name)
        dataset = qlearning_dataset(env, terminate_on_end=False, disable_goal=disable_goal)

    if clip_to_eps:
        lim = 1 - eps
        dataset["rewards"] = np.clip(dataset["rewards"], -lim, lim)

    dataset["terminals"][-1] = 1  # Ensure the last timestep is terminal
    if filter_terminals:
        non_term_ids = np.nonzero(~dataset["terminals"])[0]
        term_ids = np.nonzero(dataset["terminals"])[0]
        term_warning_ids = term_ids - 1
        new_dataset = {}

        for k, v in dataset.items():
            if k == "terminals":
                v[term_warning_ids] = 1  # set the terminal flag for the previous timestep
            new_dataset[k] = v[non_term_ids]  # filter out terminal steps

        dataset = new_dataset

    return dataset


def antmaze_preprocess(dataset: Dict[str, np.ndarray], adjust_goal: bool = False) -> np.ndarray:
    """
    Preprocessor for AntMaze dataset.

    Args:
        dataset:        The dataset
        adjust_goal:    If True, adjust the goal to be the mean of the last 10 steps
    """
    logger.info("Preprocessing for AntMaze")
    # >>> Fixing terminals
    terminals = np.zeros_like(dataset["rewards"])
    dataset["terminals"][:] = 0

    ep_changed = (
        np.linalg.norm(  # obs skips the terminal states, but next_obs doesn't
            dataset["observations"][1:] - dataset["next_observations"][:-1], axis=-1
        ) > 1e-6)

    ep_changed = ep_changed.astype(np.float32)
    terminals[:-1] = ep_changed
    terminals[-1] = 1

    convolve = np.vectorize(np.convolve, excluded=[1, "mode"], signature="(a)->(b)")
    if adjust_goal and "goals" in dataset:
        k = np.ones(10) / 10  # 10-step moving average
        mean_obs = convolve(dataset["observations"][..., :2].T, k, mode="full").T[:, len(dataset["observations"])]
        mean_obs(mean_obs * terminals[:, None])[terminals == 1]  # mean over the last 10 steps
        near_term_ids = np.searchsorted(np.where(terminals)[0], np.arange(len(terminals)))
        dataset["goals"] = mean_obs[near_term_ids]

    dataset["terminals"] = terminals.copy()
    return dataset


def maze2d_preprocess(dataset, **kwargs):
    logger.info("Preprocessing for Maze2D")
    # >>> Fixing terminals
    terminals = np.zeros_like(dataset["rewards"])
    ep_unchanged = np.linalg.norm(dataset["goals"][1:] - dataset["goals"][:-1], axis=-1) > 1e-6

    terminals[:-1] = ep_unchanged.astype(np.float32)
    terminals[-1] = 1

    # >>> Adjusting final dataset step
    last_term_id = np.where(terminals)[0][-2]
    dataset["goals"][last_term_id:] = dataset["goals"][-1] + np.random.randn(2) * 1e-6

    # >>> Transposing xy for Maze2D
    tmp_obs = dataset["observations"].reshape(-1, 2)
    tmp_obs = np.fliplr(tmp_obs).reshape(dataset["observations"].shape)
    dataset["observations"] = tmp_obs
    dataset["actions"] = np.fliplr(dataset["actions"])

    if "goals" in dataset:
        dataset["goals"] = np.fliplr(dataset["goals"])

    dataset["rewards"] = terminals.copy()
    dataset["terminals"] = terminals

    return dataset
