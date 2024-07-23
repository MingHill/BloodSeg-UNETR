import os 
import time 
import toml 
import logging 
import argparse
import pprint



from src.unetr_trainer import UNETR_TRAINER
from src.utils import select_device
from src.datasets import UnetCustomDataset, unet_valid_collate, unet_train_collate
from src.unetr_8x8 import CustomUNETR8 
from src.unetr_4x4_model import CustomUNETR 
from src.unetr_2x2 import CustomUNETR2


from transformers import ViTMAEConfig, ViTConfig, ViTMAEForPreTraining, ViTModel 

import numpy as np 
import torch
from torch import nn 
from torch.utils.data import DataLoader

import monai
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric

if __name__ == '__main__': 
    tic = time.time()
    device = select_device()

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type = str)

    args = parser.parse_args()
    config = toml.load(args.config)

    logging.basicConfig(
        level = logging.INFO, 
        format = "%(asctime)s - %(levelname)s - %(message)s", 
        handlers = [logging.FileHandler(config["LOG_DIR"] + "info.md"), logging.StreamHandler()])
    
    logging.info(pprint.pformat(config))

    # ----- Setting Random Seed -------- 
    np.random.seed(config["RANDOM_SEED"])
    torch.manual_seed(config["RANDOM_SEED"])
    torch.cuda.manual_seed(config["RANDOM_SEED"])

    torch.backends.cudnn.deterministic = False

    # -------- DATASETS & DATALOADER --------- 
    os.environ["TRAIN_DATASET"] = os.path.expandvars(config["TRAIN_DATASET"])
    os.environ["VALID_DATASET"] = os.path.expandvars(config["VALID_DATASET"])
    os.environ["TEST_DATASET"] = os.path.expandvars(config["TEST_DATASET"])

    train_data = np.load(os.environ["TRAIN_DATASET"])
    validation_data = np.load(os.environ["VALID_DATASET"])
    test_data = np.load(os.environ["TEST_DATASET"])

    train_images, train_labels = train_data['images'], train_data['labels']
    valid_images, valid_labels = validation_data['images'], validation_data['labels']
    test_images, test_labels = test_data['images'], test_data['labels']
    
    train_dataset = UnetCustomDataset(train_images, train_labels)
    valid_dataset = UnetCustomDataset(valid_images, valid_labels)
    test_dataset = UnetCustomDataset(test_images, test_labels)

    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=config["BATCH_SIZE"], 
        shuffle=True, 
        collate_fn=unet_train_collate
    )
    

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=config["BATCH_SIZE"], 
        shuffle=False,
        collate_fn=unet_valid_collate
    )

    test_dataloader = DataLoader(
        dataset=test_dataset, 
        batch_size=config["BATCH_SIZE"], 
        shuffle=False, 
        collate_fn=unet_valid_collate
    )

    logging.info(f"Train -> {len(train_dataset)}, {len(train_dataloader)} (samples, batches)")
    logging.info(f"Valid -> {len(valid_dataset)}, {len(valid_dataloader)} (samples, batches)")
    logging.info(f"Test  -> {len(test_dataset)},  {len(test_dataloader)}  (samples, batches)")

    # ------- TRANSFER WEIGHT OF EXSISTING VITMAE ENCODER INTO UNETR VIT ENCODER ---------
    vitmaeconfig = { 
        "attention_probs_dropout_prob": 0.0,
        "decoder_hidden_size": 192,
        "decoder_intermediate_size": 768,
        "decoder_num_attention_heads": 6,
        "decoder_num_hidden_layers": 6,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "hidden_size": 192,
        "image_size": 64,
        "initializer_range": 0.02,
        "intermediate_size": 768,
        "layer_norm_eps": 1e-06,
        "mask_ratio": 0.75,
        "model_type": "vit_mae",
        "norm_pix_loss": 1,
        "num_attention_heads": 6,
        "num_channels": 16,
        "num_hidden_layers": 6,
        "patch_size": 2,
        "qkv_bias": True,
        "transformers_version": "4.41.2"
    }

    # Configuration for VIT encoder for UNETR
    vitconfig = { 
        "hidden_size": vitmaeconfig["hidden_size"],
        "num_hidden_layers": vitmaeconfig["num_hidden_layers"],
        "num_attention_heads": vitmaeconfig["num_attention_heads"],
        "intermediate_size": vitmaeconfig["intermediate_size"],
        "hidden_act": vitmaeconfig["hidden_act"],
        "hidden_dropout_prob": vitmaeconfig["hidden_dropout_prob"],
        "attention_probs_dropout_prob": vitmaeconfig["attention_probs_dropout_prob"],
        "initializer_range": vitmaeconfig["initializer_range"],
        "layer_norm_eps": vitmaeconfig["layer_norm_eps"],
        "image_size": vitmaeconfig["image_size"],
        "patch_size": vitmaeconfig["patch_size"],
        "num_channels": vitmaeconfig["num_channels"],
        "qkv_bias": vitmaeconfig["qkv_bias"],
        "encoder_stride": vitmaeconfig["patch_size"],
    }
    # Extracting Pretrained VITMAE Encoder

    vitmae_model = ViTMAEForPreTraining(config = ViTMAEConfig(**vitmaeconfig))
    
    # Comment below to run a non Pretrained model
    pretrained_model_path = config["PRE_TRAINED_MODEL"]
    logging.info(f"Pretrained Model Path : {pretrained_model_path}")
    checkpoint = torch.load(pretrained_model_path)
    vitmae_model.load_state_dict(checkpoint['model_state_dict'])

    # Transfer to new VITCONFIG
    vitmae_encoder = vitmae_model.vit
    vit = ViTModel(config=ViTConfig(**vitconfig)).to(device)
    vit.load_state_dict(vitmae_encoder.state_dict(), strict=False)


    # Select which model 
    unet_model = CustomUNETR2(encoder=vit,
                             num_classes = config["NUM_CLASSES"],
                             feature_size=config["FEATURE_SIZE"]).to(device)
    
    # unet_model = CustomUNETR(encoder=vit,
    #                          num_classes = config["NUM_CLASSES"],
    #                          feature_size=config["FEATURE_SIZE"]).to(device)
    

    # unet_model = CustomUNETR8(encoder=vit,
    #                          num_classes = config["NUM_CLASSES"],
    #                          feature_size=config["FEATURE_SIZE"]).to(device)
    
    logging.info(unet_model)
    logging.info(pprint.pformat(unet_model.config))

    logging.info(f"Parameters: {sum(param.numel() for param in unet_model.parameters())}")
    logging.info(f"Trainable parameters: {sum(p.numel() for p in unet_model.parameters() if p.requires_grad)}\n")

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    adam = torch.optim.Adam(unet_model.parameters(),
                                 betas=([config["BETA_1"], config["BETA_2"]]),
                                 lr=config["LEARNING_RATE"],
                                 weight_decay = config["WEIGHT_DECAY"])

    # adamw = torch.optim.AdamW(unet_model.parameters,
    #                           lr = config["LEARNING_RATE"],
    #                           betas=(config["BETA_1"], config["BETA_2"]),
    #                           weight_decay=config["WEIGHT_DECAY"]
    #                         )
    

    trainer = UNETR_TRAINER(
        model = unet_model, 
        optimizer=adam, 
        criterion = criterion, 
        log_dir = config["LOG_DIR"], 
        device = device
    )

    trainer.fit(
        num_epochs=config["NUM_EPOCHS"],
        train_batches=train_dataloader, 
        valid_batches=valid_dataloader,
        save_checkpoint= config["SAVE_MODELS"], 
        log_batch_loss=True,
        train_eval_batches=train_dataloader
    )

    test_loss, class_report = trainer.test(test_dataloader)
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Classification report : \n {class_report}")

    toc = time.time()
    logging.info(f"Elapsed time -> {((toc - tic) / 60):.4f} minutes")



