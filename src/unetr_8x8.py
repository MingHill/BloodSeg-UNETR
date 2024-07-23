import torch 
from torch import nn 
import monai
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock # https://docs.monai.io/en/stable/networks.html#unetr-block
from monai.networks.nets import UNETR # https://docs.monai.io/en/stable/networks.html#unetr
from transformers import ViTMAEConfig, ViTMAEForPreTraining, ViTConfig, ViTModel
import numpy as np
from src.datasets import UnetCustomDataset, unet_valid_collate
from torch.utils.data import DataLoader
from monai.networks.blocks.dynunet_block import UnetOutBlock
from src.unetr_trainer import UNETR_TRAINER
from src.utils import select_device


class CustomUNETR8(nn.Module): 
    def __init__(self, encoder, num_classes = int, feature_size = int, norm_name = "instance", res_block = True, spatial_dims = 2, in_channels = 16): 
        super().__init__()
        
        self.in_channels = in_channels
        self.encoder = encoder
        self.encoder_config = self.encoder.config
        self.hidden_size = self.encoder_config.hidden_size
        self.patch_size = self.encoder_config.patch_size
        self.feat_size = [self.encoder_config.image_size // self.patch_size, self.encoder_config.image_size // self.patch_size]
        self.config = {
            'num_class': num_classes, 
            'feature_size': feature_size, 
            'norm_name': norm_name, 
            'in_channels': in_channels, 
            'encoder_config': self.encoder.config, 
            'feat_size': self.feat_size,
            'res_block': res_block, 
            'spatial_dims': spatial_dims, 
            'hidden_size': self.hidden_size
        }


        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels = self.in_channels,
            out_channels = feature_size, 
            kernel_size = 3, 
            stride = 1, 
            norm_name=norm_name, 
            res_block=res_block
        )

        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims, 
            in_channels=self.hidden_size, 
            out_channels=feature_size * 2,
            num_layer=1, 
            kernel_size=3, 
            stride=1,
            upsample_kernel_size=2, 
            norm_name=norm_name,
            conv_block=True, 
            res_block=res_block
        )

        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims, 
            in_channels=self.hidden_size, 
            out_channels=feature_size * 4,
            num_layer=0, 
            kernel_size=3, 
            stride=1,
            upsample_kernel_size=2, 
            norm_name=norm_name,
            conv_block=True, 
            res_block=res_block
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims, 
            in_channels=self.hidden_size,
            out_channels=feature_size*4, 
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims, 
            in_channels=feature_size * 4,
            out_channels=feature_size*2, 
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size*2, 
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=num_classes
        )
        

        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x): 
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x 


    def forward(self, input): # Input shape: [16, 16, 64, 64] (B, C, H, W)

        vit_out = self.encoder(input, output_hidden_states = True) 
        hidden_states = vit_out.hidden_states 
        last_hidden_state = vit_out.last_hidden_state # torch.Size([16, 257, 192])
        last_hidden_state_no_cls = last_hidden_state[:, 1:, :]  # torch.Size([16, 256, 192])
        hidden_states_no_cls = [state[:, 1:, :] for state in hidden_states]

        enc1 = self.encoder1(input) # enc1 shape: torch.Size([16, 4, 64, 64]) (B, FS, H, W)
        
        x2 = hidden_states_no_cls[2] 
        enc2 = self.encoder2(self.proj_feat(x2)) # proj(x2) shape: torch.Size([16, 192, 16, 16])
        # enc2 shape: torch.Size([16, 8, 128, 128])

        '''Added these blocks for an 8x8 patch'''
        x3 = hidden_states_no_cls[4]
        enc3 = self.encoder3(self.proj_feat(x3))

        dec3 = self.proj_feat(last_hidden_state_no_cls)

        dec2 = self.decoder4(dec3, enc3) # torch.Size([16, 192, 16, 16])
        
        dec1 = self.decoder3(dec2, enc2) # torch.Size([16, 64, 32, 32])
        
        out = self.decoder2(dec1, enc1) # torch.Size([16, 32, 64, 64])
        logits = self.out(out) # torch.Size([16, 16, 64, 64])

        return logits


if __name__ == "__main__": 
    
    # original VITMAE config (retrieve from logs/info)
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
    "layer_norm_eps": 1e-05,
    "mask_ratio": 0.5,
    "model_type": "vit_mae",
    "norm_pix_loss": 1,
    "num_attention_heads": 6,
    "num_channels": 16,
    "num_hidden_layers": 6,
    "patch_size": 4,
    "qkv_bias": True,
    "transformers_version": "4.42.3"
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

    

    '''Extracting Pretrained VITMAE Encoder'''
    pretrained_model_path = "/home/mhill/Projects/cathepsin/logs/vitmae-grid/2/model.pth"
    checkpoint = torch.load(pretrained_model_path)
    vitmae_model = ViTMAEForPreTraining(config = ViTMAEConfig(**vitmaeconfig))
    vitmae_model.load_state_dict(checkpoint['model_state_dict'])

    '''Transfer to new VITCONFIG'''
    vitmae_encoder = vitmae_model.vit
    vit = ViTModel(config=ViTConfig(**vitconfig))
    vit.load_state_dict(vitmae_encoder.state_dict(), strict=False)

    ''' Loading Train Data '''

    train_data = np.load("/home/mhill/Projects/cathepsin/data/unet_training_dataset.npz")
    train_images = train_data['images']
    train_labels = train_data['labels']
    train_dataset = UnetCustomDataset(train_images, train_labels)

    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=16, 
        collate_fn=unet_valid_collate
    )
    ''' Load Valid Data '''
    valid_data = np.load('/home/mhill/Projects/cathepsin/data/unet_validation_dataset.npz')
    valid_images = valid_data['images']
    valid_labels = valid_data['labels']
    valid_dataset = UnetCustomDataset(valid_images, valid_labels)

    valid_dataloader = DataLoader( 
        dataset = valid_dataset, 
        batch_size=16, 
        collate_fn=unet_valid_collate
    )

    train_next = next(iter(train_dataloader)) # torch.Size([16, 16, 64, 64]) (B, C, H, W)
    valid_next = next(iter(valid_dataloader)) # label_next shape : torch.Size([16, 64, 64])

    print("\n======== Dataset Lengths ========= \n")
    print(f"Length of Train Dataset: {len(train_dataset)}")
    print(f"Length of Validation Dataset: {len(valid_dataset)} \n")
    
    print("======== Batch Shapes ========== \n")
    print(f"Train Image {train_next[0].shape}")
    print(f"Train label {train_next[1].shape}")
    print(f"Valid Image {valid_next[0].shape}")
    print(f"Valid label {valid_next[1].shape} \n")
    print("Starting Training \n_____________________________________ \n")
    device = select_device()
    unet_model = CustomUNETR8(encoder=vit, num_classes = 16, feature_size=32).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=1e-3)
    
    trainer = UNETR_TRAINER(model = unet_model,
                           optimizer=optimizer,
                            criterion=criterion, 
                            device = 'cuda')
    
    model = trainer.fit(num_epochs=5, 
                train_batches=train_dataloader,
                valid_batches=valid_dataloader,
                train_eval_batches=train_dataloader)

 
    