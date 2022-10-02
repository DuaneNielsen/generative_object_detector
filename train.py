import torch
import flash
from flash.image import ImageClassificationData, ImageClassifier
import pytorch_lightning as pl
import wandb
from wandb.plot import confusion_matrix
from torchvision.transforms.functional import to_tensor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
import argparse
import random
from torchmetrics import ConfusionMatrix


def load_labels(datamodule):
    val_loader = datamodule.val_dataloader()
    ds = val_loader.dataset
    return ds.labels


class WandbImagePredCallback(pl.Callback):
    """Logs the input images and output predictions of a module.
    Predictions and labels are logged as class indices."""

    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        dm = trainer.datamodule
        for batch in dm.val_dataloader():
            N = batch['input'].shape[0]
            x = batch['input'].to(pl_module.device)
            y = batch['target']
            pred = torch.argmax(pl_module(x), 1)
            for i in range(N):
                trainer.logger.experiment.log({
                    "val/examples": [
                        wandb.Image(x[i], caption=f"Pred:{dm.labels[pred[i]]}, Label:{dm.labels[y[i]]}")
                    ],
                    "global_step": trainer.global_step
                })


class WandbConfusionMatrixCallback(pl.Callback):
    """Logs the input images and output predictions of a module.
    Predictions and labels are logged as class indices."""

    def __init__(self):
        super().__init__()
        self.cm = None

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.cm = ConfusionMatrix(num_classes=len(trainer.datamodule.labels))

    def on_validation_epoch_end(self, trainer, pl_module):
        dm = trainer.datamodule
        labels = trainer.datamodule.labels
        for batch in dm.val_dataloader():
            x = batch['input'].to(pl_module.device)
            y = batch['target']
            pred = torch.argmax(pl_module(x), dim=1)
            self.cm.update(pred.to(self.cm.device), y.to(self.cm.device))

        trainer.logger.experiment.log({
            'confusion': wandb.Table(data=self.cm.confmat.tolist(), columns=labels),
            'global_step': trainer.global_step})

        self.cm.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    data_root_dir = '/mnt/data/data'

    wandb_logger = WandbLogger(project="traffic cone classifier", log_model=False)
    wandb_image_f = WandbImagePredCallback()

    checkpoint_f = ModelCheckpoint(dirpath=f"checkpoints//{wandb_logger.experiment.name}/",
                                   save_top_k=1, mode='max',
                                   monitor="val_accuracy")

    datamodule = ImageClassificationData.from_folders(
        train_folder=f"{data_root_dir}/traffic_cones/train/",
        val_folder=f"{data_root_dir}/traffic_cones/val/",
        batch_size=8,
        transform_kwargs={"image_size": (196, 196), "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    )

    # Init or load model
    if args.resume is None:
        model = ImageClassifier(backbone="resnet18", labels=datamodule.labels,
                                #lr_scheduler=("MultiStepLR", {"milestones": [10, 50], "gamma": 0.1}),
                                learning_rate=1e-3
                                )
    else:
        model = ImageClassifier.load_from_checkpoint(args.resume)

    # Create the trainer and finetune the model
    trainer = flash.Trainer(max_epochs=50, gpus=torch.cuda.device_count(),
                            strategy=DDPPlugin(find_unused_parameters=False),
                            logger=wandb_logger,
                            enable_checkpointing=True,
                            callbacks=[LearningRateMonitor(), checkpoint_f, wandb_image_f, WandbConfusionMatrixCallback()])
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")

    with open('labels.txt', 'w') as f:
        for label in load_labels(datamodule):
            f.write(label + '\n')