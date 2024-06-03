# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional
from pathlib import Path
import wandb
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only


def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    return None
    # raise Exception("You are using wandb related callback, but WandbLogger was not found for some reason...")


class ModelCheckpointWB(ModelCheckpoint):
    def save_checkpoint(self, trainer) -> None:
        super().save_checkpoint(trainer)
        if not hasattr(self, "_last_artifact"):
            self._last_artifact = {}
        if not hasattr(self, "_best_artifact"):
            self._best_artifact = {}
        logger = get_wandb_logger(trainer)
        if self.current_score is None:
            self.current_score = trainer.callback_metrics.get(self.monitor)
        if logger is not None:
            self._scan_and_log_checkpoints(logger)

    @rank_zero_only
    def _scan_and_log_checkpoints(self, wb_logger: WandbLogger) -> None:
        metadata = {
            "ModelCheckpoint": {
                k: getattr(self, k)
                for k in [
                    "monitor",
                    "mode",
                    "save_last",
                    "save_top_k",
                    "save_weights_only",
                    "_every_n_train_steps",
                    "_every_n_val_epochs",
                ]
                # ensure it does not break if `ModelCheckpoint` args change
                if hasattr(self, k)
            },
        }
        if self.best_model_path not in self._best_artifact and os.path.isfile(self.best_model_path):  # a new best model
            for k in self._best_artifact.keys():
                self._best_artifact[k].delete(True)
            metadata["score"] = self.best_model_score.item()
            metadata["original_filename"] = Path(self.best_model_path).name
            artifact = wandb.Artifact(name=wb_logger.experiment.id, type="model", metadata=metadata)
            artifact.add_file(self.best_model_path, name="model.ckpt")
            wb_logger.experiment.log_artifact(artifact, aliases=["latest", "best"])
            artifact.wait()
            self._best_artifact = {self.best_model_path: artifact}
        else:  # log latest
            for k in self._last_artifact.keys():
                self._last_artifact[k].delete(True)
            metadata["score"] = self.current_score.item()
            metadata["original_filename"] = Path(self.last_model_path).name
            artifact = wandb.Artifact(name=wb_logger.experiment.id, type="model", metadata=metadata)
            artifact.add_file(self.last_model_path, name="model.ckpt")
            wb_logger.experiment.log_artifact(artifact, aliases=["latest"])
            artifact.wait()
            self._last_artifact = {self.last_model_path: artifact}

