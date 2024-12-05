from numbers import Number
from typing import Any, Dict, Literal, Optional, Union

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np
from flax.traverse_util import flatten_dict
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import src.common.logger as logger
from src.common.config import BaseConfig
from src.common.utils import reduce_array_to_scalar, plot_to_img

Array = Union[np.ndarray, jax.Array]


class LogWriter:
    def __init__(self, cfg: BaseConfig):
        self.cfg = cfg
        self.writer = self.create_writer()

    def create_writer(self):
        if self.cfg.logger_type.lower() == "wandb":
            logger = WandbLogger(
                project=self.cfg.exp_name.replace("/", "_"),
                name=self.cfg.version, 
                save_dir=self.cfg.log_dir, 
                version=self.cfg.version, 
                config=self.cfg.to_dict()
            )
        elif self.cfg.logger_type.lower() == "tensorboard":
            logger = TensorBoardLogger(
                save_dir=self.cfg.log_dir, 
                name=self.cfg.exp_name.replace("/", "_"), 
                version=self.cfg.version
            )
            logger.log_hyperparams(self.cfg.to_dict())
        else:
            raise ValueError(f"Invalid logger type: {self.cfg.logger_type}")
        return logger

    def infer_type(self, metrics: Dict[str, Any]) -> Literal["scalar", "heatmap", "image", "unknown"]:
        """
        Infer the representation type of the metrics.
        This type can be one of the following: `scalar`, `heatmap`, `image`, `unknown`.
        - `scalar`:     A scalar value.
        - `heatmap`:    2d array representing a heatmap.
        - `image`:      3d array representing an image.
        - `unknown`:    Unknown(undefined) type.

        Args:
            metrics:    Dictionary of metrics.

        Returns:
            Dictionary of metrics types.
        """
        infer_use_name = self.cfg.infer_with_name

        def check_type(k: str, v: Any):
            if isinstance(v, Number):
                return "scalar"
            elif isinstance(v, Array):
                if v.size == 1:
                    return "scalar"
                elif v.ndim == 2:
                    return "heatmap" if not k.endswith("img") or (infer_use_name and k.endswith("map")) else "image"
                elif v.ndim == 3:
                    return "image"
            return "unknown"

        metrics_type = jax.tree.map(check_type, metrics)
        return metrics_type

    # def log_metrics(self, metrics: Dict[str, Any], step_or_epoch: int = 0, prefix: str = ""):
    #     metrics = flatten_dict(metrics, sep="_")
    #     metrics_type = self.infer_type(metrics)
    #     metrics_to_log = {k: reduce_array_to_scalar(v) for k, v in metrics.items() if metrics_type[k] != "unknown"}

    #     for k, v in metrics_to_log.items():
    #         save_key = f"{prefix}/{k}" if prefix else k
    #         if metrics_type[k] == "image":
    #             self.log_image(save_key, image=v, step=step_or_epoch)
    #         elif metrics_type[k] == "heatmap":
    #             # FIXME: This is a temporary solution to log heatmaps.
    #             self.log_figure(save_key, figure=plt.pcolormesh(v).figure, step=step_or_epoch)
    #         else:
    #             self.writer.log_metrics({save_key: v}, step=step_or_epoch)
    
    def log_scalar(self, metric_key: str, metric_value: Union[float, jp.ndarray], step: int, postfix: str = "") -> None:
        self.writer.log_metrics({metric_key + postfix: metric_value}, step)

    def log_image(self, key: str, image: Array, step: int = None, post_fix: str = "") -> None:
        if isinstance(image, jax.Array):
            image = jax.device_get(image)

        log_key = key + post_fix
        if isinstance(self.writer, TensorBoardLogger):
            self.writer.experiment.add_image(tag=log_key, img_tensor=image, global_step=step, dataformats="HWC")
        elif isinstance(self.writer, WandbLogger):
            self.writer.log_image(key=log_key, images=[image], step=step)
        else:
            raise ValueError(f"Unsupported logger type: {type(self.writer)}")

    def log_figure(self, key: str, figure: plt.Figure, step: int = None, postfix: str = ""):
        plot_img = plot_to_img(figure)
        self.log_image(key, plot_img, step, postfix)

    def log_embedding(
        self,
        key: str,
        encodings: np.ndarray,
        step: int = None,
        metadata: Optional[Any] = None,
        images: Optional[np.ndarray] = None,
        postfix: str = "",
    ):
        """
        Logs embeddings to the logger using the logger tool.

        Args:
            key:            Name of the embedding.
            encodings:      Encodings to log.
            step:           Step number.
            metadata:       Metadata for the embeddings.
            images:         Images for the embeddings.
            postfix:        Postfix to append to the log key.
        """
        import torch as th

        log_key = key + postfix
        if isinstance(self.writer, TensorBoardLogger):
            images = np.transpose(images, (0, 3, 1, 2))
            images = th.from_numpy(images)
            self.writer.experiment.add_embedding(
                tag=log_key, mat=encodings, metadata=metadata, label_img=images, global_step=step
            )
        elif isinstance(self.writer, WandbLogger):
            logger.warning("Wandb does not support logging embeddings.")
        else:
            raise ValueError(f"Unsupported logger type: {type(self.writer)}")
        
    def log_video(self, key: str, video: np.ndarray, step: int = None, postfix: str = "", fps: int = 30, format: str = "mp4"):
        log_key = key + postfix
        if isinstance(self.writer, TensorBoardLogger):
            self.writer.experiment.add_video(tag=log_key, vid_tensor=video, global_step=step, fps=fps)
        elif isinstance(self.writer, WandbLogger):
            self.writer.log_video(key=log_key, videos=[video], step=step, fps=[fps], format=[format])
        else:
            raise ValueError(f"Unsupported logger type: {type(self.writer)}")

    def finalize(self, status: Literal["success", "failure"]):
        self.writer.finalize(status)

    @property
    def log_dir(self) -> str:
        return self.writer.log_dir
