import time
import toml
import pprint
import logging
import argparse

import torch
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split

from transformers import ViTMAEConfig, ViTMAEForPreTraining

from src.trainer import Trainer
from src.datasets import CustomCIFAR10Dataset

from src.utils import select_device


if __name__ == '__main__':
    tic = time.time()
    device = select_device()

    # ----------------------> configurations <----------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str)

    args = parser.parse_args()
    config = toml.load(args.config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(config["LOG_DIR"] + "info.md"), logging.StreamHandler()])

    logging.info(pprint.pformat(config))

    # ----------------------> set random seeds <------------
    torch.manual_seed(config["RANDOM_SEED"])
    torch.cuda.manual_seed(config["RANDOM_SEED"])

    # ---------------------- datasets ----------------------
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(config["IMAGE_SIZE"]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_valid = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.CIFAR10(root=config["DATASOURCE"], train=True)
    train_data, valid_data = random_split(dataset, [40000, 10000])

    test_data = datasets.CIFAR10(root=config["DATASOURCE"], train=False)

    train_dataset = CustomCIFAR10Dataset(train_data, transform=transform_train)
    valid_dataset = CustomCIFAR10Dataset(valid_data, transform=transform_valid)
    test_dataset = CustomCIFAR10Dataset(test_data, transform=transform_test)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False
    )

    logging.info(f"Train -> {len(train_dataset)}, {len(train_dataloader)} (samples, batches)")
    logging.info(f"Valid -> {len(valid_dataset)}, {len(valid_dataloader)} (samples, batches)")
    logging.info(f"Test  -> {len(test_dataset)},  {len(test_dataloader)}  (samples, batches)")

    # ----------------------> define model <----------------------
    vitmaeconfig = {
        'image_size': config["IMAGE_SIZE"],
        'patch_size': config["PATCH_SIZE"],
        'num_channels': config["NUM_CHANNELS"],
        'mask_ratio': config["MASK_RATIO"],
        #
        'hidden_size': config["HIDDEN_SIZE"],
        'intermediate_size': config["INTERMEDIATE_SIZE"],
        'num_hidden_layers': config["NUM_HIDDEN_LAYERS"],
        'num_attention_heads': config["NUM_ATTENTION_HEADS"],
        #
        'hidden_dropout_prob': config["HIDDEN_DROPOUT_PROB"],
        'attention_probs_dropout_prob': config["ATTENTION_PROBS_DROPOUT_PROB"],
        #
        'decoder_hidden_size': config["DECODER_HIDDEN_SIZE"],
        'decoder_intermediate_size': config["DECODER_INTERMEDIATE_SIZE"],
        'decoder_num_hidden_layers': config["DECODER_NUM_HIDDEN_LAYERS"],
        'decoder_num_attention_heads': config["DECODER_NUM_ATTENTION_HEADS"],
    }
    model = ViTMAEForPreTraining(config=ViTMAEConfig(**vitmaeconfig)).to(device)

    logging.info(model)
    logging.info(pprint.pformat(model.config))

    logging.info(f"Parameters: {sum(param.numel() for param in model.parameters())}")
    logging.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["LEARNING_RATE"],
        betas=(config["BETA_1"], config["BETA_2"]),
        weight_decay=config["WEIGHT_DECAY"]
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config["LEARNING_RATE"],
        total_steps=len(train_dataloader) * config["NUM_EPOCHS"],
        pct_start=config["PCT_START"],
        div_factor=config["DIV_FACTOR"],
        final_div_factor=config["FINAL_DIV_FACTOR"],
    ) if config["SCHEDULER"] else None
    logging.info(f"Scheduler: {scheduler}")

    # ----------------------> train and evaluate <----------------------
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        log_dir=config["LOG_DIR"],
        device=device,
    )

    trainer.fit(
        num_epochs=config["NUM_EPOCHS"],
        train_batches=train_dataloader,
        valid_batches=valid_dataloader,
        save_checkpoint=config["SAVE_MODELS"],
    )

    test_loss = trainer.evaluate(dataloader=test_dataloader)
    logging.info(f"Test Loss: {test_loss:.4f}")

    toc = time.time()
    logging.info(f"Elapsed time -> {((toc - tic) / 60):.4f} minutes")
