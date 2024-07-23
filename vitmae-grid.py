import os
import time
import toml
import pprint
import logging
import argparse

import numpy as np

import torch

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from transformers import ViTMAEConfig, ViTMAEForPreTraining
from src.models import ViTMAEEmbeddingsMasking

from src.trainer import Trainer
from src.datasets import CustomDataset ,collate_fn_train, collate_fn_valid_test

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
    np.random.seed(config["RANDOM_SEED"])
    torch.manual_seed(config["RANDOM_SEED"])
    torch.cuda.manual_seed(config["RANDOM_SEED"])

    # enable for debugging (slows down compute)
    torch.backends.cudnn.deterministic = False

    # ----------------------> datasets <----------------------
    os.environ["TRAIN_DATASET"] = os.path.expandvars(config["TRAIN_DATASET"])
    os.environ["VALID_DATASET"] = os.path.expandvars(config["VALID_DATASET"])
    os.environ["TEST_DATASET"] = os.path.expandvars(config["TEST_DATASET"])

    training_dataset = np.load(os.environ["TRAIN_DATASET"])
    validation_dataset = np.load(os.environ["VALID_DATASET"])
    testing_dataset = np.load(os.environ["TEST_DATASET"])

    # --> using random crop <--
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(64),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    # ])
    # transform = transforms.Compose([transforms.RandomCrop(64)])

    # train_dataset = CustomDataset(training_dataset, transform=transform_train)
    # train_eval_dataset = CustomDataset(training_dataset, transform=transform)
    # valid_dataset = CustomDataset(validation_dataset, transform=transform)
    # test_dataset = CustomDataset(testing_dataset, transform=transform)

    # train_dataloader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True)
    # train_eval_dataloader = DataLoader(train_eval_dataset, batch_size=config["BATCH_SIZE"], shuffle=True)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=config["BATCH_SIZE"], shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=config["BATCH_SIZE"], shuffle=False)
    # --> using random crop <--

    # --> using collate functions <--
    train_dataset = CustomDataset(training_dataset)
    valid_dataset = CustomDataset(validation_dataset)
    test_dataset = CustomDataset(testing_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        collate_fn=collate_fn_train
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        collate_fn=collate_fn_valid_test
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        collate_fn=collate_fn_valid_test
    )
    # --> using collate functions <--

    logging.info(f"Train -> {len(train_dataset)}, {len(train_dataloader)} (samples, batches)")
    logging.info(f"Valid -> {len(valid_dataset)}, {len(valid_dataloader)} (samples, batches)")
    logging.info(f"Test  -> {len(test_dataset)},  {len(test_dataloader)}  (samples, batches)")

    # ----------------------> define model <----------------------
    vitmaeconfig = {
        'image_size': config["IMAGE_SIZE"],
        'patch_size': config["PATCH_SIZE"],
        'num_channels': config["NUM_CHANNELS"],
        #
        'mask_ratio': config["MASK_RATIO"],
        'norm_pix_loss': config["NORM_PIX_LOSS"],
        'layer_norm_eps': config["LAYER_NORM_EPS"],
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

    model.vit.embeddings = ViTMAEEmbeddingsMasking(
        config=ViTMAEConfig(**vitmaeconfig),
        device=device,
        masking_type=config["MASK_TYPE"],
    ).to(device)

    logging.info(model)
    logging.info(pprint.pformat(model.config))

    logging.info(f"Parameters: {sum(param.numel() for param in model.parameters())}")
    logging.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=(config["BETA_1"], config["BETA_2"]),
        lr=config["LEARNING_RATE"],
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
        train_eval_batches=train_dataloader,
        num_eval_batches=len(valid_dataloader),
        save_checkpoint=config["SAVE_MODELS"],
        log_batch_loss=True,
    )

    test_loss = trainer.evaluate(dataloader=test_dataloader)
    logging.info(f"Test Loss: {test_loss:.4f}")

    toc = time.time()
    logging.info(f"Elapsed time -> {((toc - tic) / 60):.4f} minutes")