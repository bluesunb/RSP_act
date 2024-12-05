import os
from dataclasses import asdict, dataclass

import yaml

import src.common.logger as logger


@dataclass
class BaseConfig:
    def __str__(self):
        str_repr = self.__class__.__name__ + ":\n"
        for k, v in asdict(self).items():
            str_repr += f"  {k}: {v}\n"
        return str_repr

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def load(cls, load_dir: str):
        yaml_cfg_path = os.path.join(load_dir, "config.yaml")
        with open(yaml_cfg_path, "r") as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**cfg_dict)

    def save(self, save_dir: str = None):
        if save_dir is None:
            save_dir = self.save_dir

        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"Directory {save_dir} does not exist.")

        yaml_cfg_path = os.path.join(save_dir, "config.yaml")
        txt_cfg_path = os.path.join(save_dir, "config.txt")

        with open(yaml_cfg_path, "w") as f:
            yaml.dump(self.to_dict(), f)
            logger.info(f"Config saved to {yaml_cfg_path}")

        with open(txt_cfg_path, "w") as f:
            f.write(self.__str__())
            logger.info(f"Config saved to {txt_cfg_path}")


def config_diff(user_cfg: BaseConfig) -> dict:
    default_cfg = user_cfg.__class__()
    diff = {k: v for k, v in user_cfg.to_dict().items() if getattr(default_cfg, k) != v}
    return diff