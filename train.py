import argparse
import yaml
import wandb
from pathlib import Path

from torch.utils.data import DataLoader, Subset
import torch

from tts_modules.DataUtils.LJSpeech import LJSpeechDataset
from tts_modules.DataUtils.Collator import LJSpeechCollator
from tts_modules.FastSpeech.Aligner.GraphemeAligner import GraphemeAligner
from tts_modules.FastSpeech.FastSpeech import FastSpeech
from tts_modules.Featurizer.MelSpectrogram import get_featurizer
from tts_modules.trainer import FastSpeechTrainer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

root = Path(__name__).parent


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="path to yaml config for training", required=True)
    parser.add_argument("--resume", "-r", help="path to model checkpoint")
    return parser


def parse_yaml(path_to_config):
    with open(path_to_config, 'r') as yaml_config:
        train_params = yaml.safe_load(yaml_config)
    return train_params


def load_checkpoint(path_to_checkpoint, device):
    checkpoint = torch.load(path_to_checkpoint, map_location=device)


def main(args):
    train_config = parse_yaml(args.config)
    wandb.init(
        project=train_config["wandb_project"],
        config=train_config
    )
    dataset = LJSpeechDataset(root / "data")
    vocab_size = dataset.vocab_size
    dataset = Subset(dataset, torch.arange(train_config["batch_size"]))
    train_dataloader = DataLoader(dataset, batch_size=train_config["batch_size"], collate_fn=LJSpeechCollator())
    validation_dataloader = DataLoader(dataset, batch_size=train_config["batch_size"], collate_fn=LJSpeechCollator())

    fastspeech_model = FastSpeech(vocab_size=vocab_size, **train_config["model"]).to(device)
    optimizer = torch.optim.Adam(fastspeech_model.parameters(), lr=train_config["optimizer"]["lr"], betas=(0.9, 0.98),
                                 eps=10e-9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **train_config["scheduler"])
    trainer = FastSpeechTrainer(
        model=fastspeech_model,
        optimizer=optimizer,
        scheduler=scheduler,
        aligner=GraphemeAligner().to(device),
        featurizer=get_featurizer().to(device),
        config=train_config,
        resume=args.resume,
        device=device
    )
    trainer.train(train_dataloader, validation_dataloader)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
