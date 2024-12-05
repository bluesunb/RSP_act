import numpy as np
from typing import Dict, Optional, Any

from src.common.d4rl.d4rl_utils import prepare_dataset
from src.common.imports import resolve_import

BASE_DATA_FORMATS = {
    "antmaze": {
        "env_name": "antmaze-large-play-v2",
        "preprocess": "src.common.d4rl.d4rl_utils.antmaze_preprocess",
        "batch": "src.PFR.datasets.trajectory.Trajectory",
    },
    "maze2d": {
        "env_name": "maze2d-large-v0",
        "preprocess": "src.datasets.d4rl_utils.maze2d_preprocess",
        "batch": "src.datasets.struct.TrajectoryBatch",
    },
}


def get_dataset(
    env_name: str,
    clip_to_eps: bool = True,
    eps: float = 1e-5,
    dataset: Optional[Dict[str, np.ndarray]] = None,
    filter_terminals: bool = False,
    disable_goal: bool = False,
    preprocess_fn: Optional[str] = None,
    batch_type: Optional[str] = None,
    prepare: bool = True,
    *,
    data_formats: Dict[str, Dict[str, Any]] = BASE_DATA_FORMATS,
    **kwargs,
) -> Any:
    dataset_info = data_formats.get(
        env_name.split("-")[0],
        {"env_name": env_name, "preprocess": preprocess_fn, "batch": batch_type}
    )
    preprocess_fn = preprocess_fn or dataset_info["preprocess"]
    batch_type = batch_type or dataset_info["batch"]

    if (preprocess_fn := resolve_import(preprocess_fn)) is None:
        preprocess_fn = lambda x: x  # noqa: E731

    if (batch_type := resolve_import(batch_type)) is None:
        batch_type = dict

    if prepare:
        dataset = prepare_dataset(
            env_name=dataset_info["env_name"],
            clip_to_eps=clip_to_eps,
            eps=eps,
            dataset=dataset,
            filter_terminals=filter_terminals,
            disable_goal=disable_goal,
        )

    dataset = preprocess_fn(dataset, **kwargs)
    construct = batch_type.create if hasattr(batch_type, "create") else batch_type
    return construct(**dataset)
