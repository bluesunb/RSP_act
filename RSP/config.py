import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
from src.common.config import BaseConfig


@dataclass
class RSPConfig(BaseConfig):
    # General parameters
    batch_size: int = 64
    drop_last: bool = True
    shuffle_files: bool = True
    train_steps: int = 400
    accum_iter: int = 1
    seed: int = 0
    
    # Logging parameters
    logger_type: str = "wandb"
    _log_dir: str = "output_dir"
    exp_name: str = None
    version: str = None
    infer_with_name: bool = False
    log_freq: int = 4
    eval_freq: int = -1
    save_freq: int = 10
    resume_epoch: int = -1
    num_eval_episodes: int = 3

    # Model parameters
    model: str = "src.RSP.rsp.rsp_vit_small_patch16"
    input_size: int = 224
    patch_size: int = 16
    norm_pixel_loss: bool = False
    mask_rate: float = 0.75
    noise_scale: float = 0.5
    kl_scale: float = 0.01
    kl_balance: float = 0.2
    kl_freebit: float = 0.1
    stoch: int = 32
    discrete: int = 32

    act_patch_size: int = 4

    # Optimizer parameters
    weight_decay: float = 0.05
    lr: Optional[float] = None
    base_lr: float = 1.5e-4
    min_lr: float = 0.0
    warmup_steps: int = 40

    # Dataset parameters
    data_type: str = "furniture"
    data_dir: str = str(Path.home() / "scripted_sim_demo/furniture/")
    img_version: str = "tpp"
    min_distance: int = 4
    max_distance: int = 48
    repeated_sampling: int = 2
    act_size: int = None
    seq_len: int = None

    @property
    def save_dir(self):
        paths = [self.log_dir, self.exp_name, self.version]
        paths = filter(None, paths)
        return os.path.join(*paths)
    
    @property
    def log_dir(self):
        model_name = self.model.split(".")[-1]
        return os.path.join(self._log_dir, model_name)
    
    def set_traj_info(self, act_size: int, seq_len: int):
        self.act_size = act_size
        self.seq_len = seq_len

def config_diff(user_cfg: RSPConfig) -> dict:
    default_cfg = RSPConfig()
    diff = {k: v for k, v in user_cfg.to_dict().items() if getattr(default_cfg, k) != v}
    return diff
