import argparse
import os
import flash
from flash.image import ImageClassificationData, ImageClassifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='a pytorch lightning checkpoint file')
    args = parser.parse_args()

    data_root_dir = '/mnt/data/data'

    datamodule = ImageClassificationData.from_folders(
        predict_folder=f"{data_root_dir}/traffic_cones/train/",
        #predict_folder=f"{data_root_dir}/traffic_cones/val",
        batch_size=16,
        transform_kwargs={"image_size": (196, 196), "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    )

    model = ImageClassifier.load_from_checkpoint(args.checkpoint)

    trainer = flash.Trainer()

    predictions = trainer.predict(model, datamodule=datamodule)

    if os.path.exists("preds.txt"):
        os.remove("preds.txt")

    with open('preds.txt', 'a') as f:
        for batch in predictions:
            for item in batch:
                filename = item['metadata']['filepath']
                pred = item['preds']
                pred_str = [i.item() for i in pred]
                f.write(f'{filename} {pred_str}\n')
