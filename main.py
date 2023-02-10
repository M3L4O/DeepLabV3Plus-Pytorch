from tqdm import tqdm
import network
import argparse
import numpy as np
from os import path as ph

from datasets.data import SegmentationDataSet
from utils.engine import Engine


import torch
from torch.utils.data import DataLoader

from torchmetrics import Dice, JaccardIndex


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument(
        "--image_dir",
        type=str,
        default="./datasets/data",
        help="Caminho para as imagens.",
    )
    parser.add_argument("--mask_dir", type=str, help="Caminho para as máscaras.")
    parser.add_argument(
        "--csv_dir",
        type=str,
        help="Caminho para a pasta contendo os CSVs.",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image_id",
        help="Coluna contendo o ID da imagem.",
    )

    parser.add_argument(
        "--image_sufix", default=".jpg", type=str, help="Sufixo para a imagem."
    )
    parser.add_argument(
        "--mask_sufix",
        default="_segmentation.png",
        type=str,
        help="Sufixo para a máscara.",
    )
    parser.add_argument(
        "--image_size", default=224, type=int, help="Tamanho da imagem."
    )
    parser.add_argument(
        "--model_path", type=str, help="Caminho até onde o modelo ficará salvo."
    )
    # Train Options

    parser.add_argument("--batch_size", type=int, default=8, help="Tamanho do batch.")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Número máximo de épocas."
    )
    return parser


def get_dataloaders(opts) -> tuple[DataLoader, DataLoader]:
    train_ds = SegmentationDataSet(
        ph.join(opts.csv_dir, "train.csv"),
        opts.image_column,
        opts.image_sufix,
        opts.mask_sufix,
        opts.image_dir,
        opts.mask_dir,
        opts.image_size,
    )
    validation_ds = SegmentationDataSet(
        ph.join(opts.csv_dir, "validation.csv"),
        opts.image_column,
        opts.image_sufix,
        opts.mask_sufix,
        opts.image_dir,
        opts.mask_dir,
        opts.image_size,
    )

    train_dl = DataLoader(
        train_ds, batch_size=opts.batch_size, shuffle=True, drop_last=True
    )
    validation_dl = DataLoader(validation_ds, batch_size=1, shuffle=False)

    return train_dl, validation_dl


def main():
    opts = get_argparser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    train_dl, validation_dl = get_dataloaders(opts)

    model = network.modeling.deeplabv3_resnet50(num_classes=1)

    criterion = torch.nn.MSELoss(reduction="mean")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    metrics = {
        "dice": Dice(num_classes=1, average="samples"),
        "jaccard": JaccardIndex(task="binary"),
    }
    best_loss = np.inf

    for epoch in range(opts.epochs):
        print("####################")
        print(f"       Epoch: {epoch}   ")
        print("####################")

        train_metrics_results = dict()
        val_metrics_results = dict()
        for bx, data in tqdm(enumerate(train_dl), total=len(train_dl)):
            train_metrics_results = Engine.train_batch(
                model, data, optimizer, criterion, metrics
            )
        print("###################")
        print("Train Results:")
        print(train_metrics_results)
        print("###################")

        for bx, data in tqdm(enumerate(validation_dl), total=len(validation_dl)):
            val_metrics_results = Engine.validate_batch(model, data, criterion, metrics)

        print("###################")
        print("Validation Results:")
        print(val_metrics_results)
        print("###################")

        print(val_metrics_results)

        if best_loss > val_metrics_results["loss"]:
            print(
                f"The loss decreased from {best_loss} from {val_metrics_results['loss']}"
            )
            torch.save(model, opts.model_path)


if __name__ == "__main__":
    main()
