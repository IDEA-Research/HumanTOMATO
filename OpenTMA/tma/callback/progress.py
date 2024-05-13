import logging
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
import psutil

logger = logging.getLogger()


class ProgressLogger(Callback):
    """
    A custom callback class for PyTorch Lightning that logs progress information during training.
    """

    def __init__(self, metric_monitor: dict, precision: int = 3):
        # Metric to monitor
        self.metric_monitor = metric_monitor
        self.precision = precision

    def on_train_start(
        self, trainer: Trainer, pl_module: LightningModule, **kwargs
    ) -> None:
        # Log a message when training starts
        logger.info("Training started")

    def on_train_end(
        self, trainer: Trainer, pl_module: LightningModule, **kwargs
    ) -> None:
        # Log a message when training ends
        logger.info("Training done")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule, **kwargs
    ) -> None:
        # Log a message when a validation epoch ends
        if trainer.sanity_checking:
            logger.info("Sanity checking ok.")

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule, padding=False, **kwargs
    ) -> None:
        # Log a message when a training epoch ends
        # Format for logging metrics
        metric_format = f"{{:.{self.precision}e}}"
        # Start the log line with the epoch number
        line = f"Epoch {trainer.current_epoch}"
        if padding:
            line = f"{line:>{len('Epoch xxxx')}}"  # Right padding
        metrics_str = []

        losses_dict = trainer.callback_metrics
        for metric_name, dico_name in self.metric_monitor.items():
            # If the metric is in the dictionary, format it and add it to the log line
            if dico_name in losses_dict:
                metric = losses_dict[dico_name].item()
                metric = metric_format.format(metric)
                metric = f"{metric_name} {metric}"
                metrics_str.append(metric)

        # If there are no metrics, return
        if len(metrics_str) == 0:
            return

        # Add the current memory usage to the log line
        memory = f"Memory {psutil.virtual_memory().percent}%"
        line = line + ": " + "   ".join(metrics_str) + "   " + memory
        logger.info(line)
