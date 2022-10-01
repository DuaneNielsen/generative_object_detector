import torch
import flash
from flash.image import ImageClassificationData, ImageClassifier
import pytorch_lightning as pl
import wandb
from torchvision.transforms.functional import to_tensor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
import argparse


class WandbImagePredCallback(pl.Callback):
    """Logs the input images and output predictions of a module.
    Predictions and labels are logged as class indices."""

    def __init__(self, num_samples=32):
        super().__init__()
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loader = trainer.datamodule.val_dataloader()
        for i in range(self.num_samples):
            x = to_tensor(val_loader.dataset[i]['input']).unsqueeze(0).to(pl_module.device)
            y = val_loader.dataset[i]['target']
            pred = torch.argmax(pl_module(x), 1)
            trainer.logger.experiment.log({
                "val/examples": [
                    wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                ],
                "global_step": trainer.global_step
            })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    data_root_dir = '/mnt/data/data'

    wandb_logger = WandbLogger(project="traffic cone classifier", log_model=False)
    wandb_image_f = WandbImagePredCallback(num_samples=4)

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
                                lr_scheduler=("MultiStepLR", {"milestones": [10, 50], "gamma": 0.1}),)
    else:
        model = ImageClassifier.load_from_checkpoint(args.resume)

    # Create the trainer and finetune the model
    trainer = flash.Trainer(max_epochs=50, gpus=torch.cuda.device_count(),
                            strategy=DDPPlugin(find_unused_parameters=False),
                            logger=wandb_logger,
                            enable_checkpointing=True,
                            callbacks=[LearningRateMonitor(), checkpoint_f, wandb_image_f])
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")