import random

import numpy as np

import torch
import torch.nn as nn

from transformers import ViTMAEConfig, ViTMAEForPreTraining
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEEmbeddings


# see modeling_vit_mae from transformers package
class ViTMAEEmbeddingsMasking(ViTMAEEmbeddings):
    def __init__(self, config, device, masking_type='random'):
        super().__init__(config)
        self.device = device
        self.masking_type = masking_type

    def grid_masking(self, sequence, noise=None):
        batch_size, num_patches, hidden_size = sequence.shape

        # calculate the image grid dimensions
        n = int(torch.sqrt(torch.tensor(num_patches)).item())

        # create a 2d grid mask in a checkerboard pattern
        grid_mask_2d = np.indices((n, n)).sum(axis=0) % 2

        # convert to tensor and move to device
        grid_mask_2d = torch.tensor(grid_mask_2d, dtype=torch.float32).to(self.device)

        # flatten the 2d grid mask to apply to the linear sequence of patches
        mask = grid_mask_2d.flatten()

        # repeat the mask for each batch
        mask = mask.repeat(batch_size, 1)

        masked_sequence = sequence.to(self.device) * mask.unsqueeze(-1)

        ids_restore = torch.arange(num_patches, device=sequence.device).unsqueeze(0).repeat(batch_size, 1)

        return masked_sequence, mask, ids_restore

    def random_masking(self, sequence, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)  # ascend: small is keep, large is removed
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is removed
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def forward(self, pixel_values, noise=None, interpolate_pos_encoding=False):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        if interpolate_pos_encoding:
            position_embeddings = self.interpolate_pos_encoding(embeddings, height, width)
        else:
            position_embeddings = self.position_embeddings

        # add position embeddings w/o cls token
        embeddings = embeddings + position_embeddings[:, 1:, :]

        # masking: length -> length * config.mask_ratio
        if self.masking_type == 'grid':
            embeddings, mask, ids_restore = self.grid_masking(embeddings, noise)
        elif self.masking_type == 'random':
            embeddings, mask, ids_restore = self.random_masking(embeddings, noise)
        else:
            raise ValueError(
                "Invalid masking type. Expected 'grid' or 'random', got {}".format(self.masking_type))

        # append cls token
        cls_token = self.cls_token + position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore


if __name__ == '__main__':
    vitmaeconfig = {
        'image_size': 64,
        'patch_size': 4,
        'num_channels': 16,
        #
        'mask_ratio': 0.50,
        'norm_pix_loss': True,
        #
        'hidden_size': 192,
        'intermediate_size': 768,
        'num_hidden_layers': 12,
        'num_attention_heads': 12,
        #
        'hidden_dropout_prob': 0.0,
        'attention_probs_dropout_prob': 0.0,
        #
        'decoder_hidden_size': 192,
        'decoder_intermediate_size': 768,
        'decoder_num_hidden_layers': 12,
        'decoder_num_attention_heads': 12,
    }
    model = ViTMAEForPreTraining(config=ViTMAEConfig(**vitmaeconfig))

    model.vit.embeddings = ViTMAEEmbeddingsMasking(
        config=ViTMAEConfig(**vitmaeconfig),
        device=torch.device('cpu'),
        masking_type='grid'
    )

    print(model)
    print(model.config)

    print(f"Parameters: {sum(param.numel() for param in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

    sample = torch.randn(1, vitmaeconfig["num_channels"], vitmaeconfig["image_size"], vitmaeconfig["image_size"])
    outputs = model(sample)

    logits = outputs.logits
    print(logits.shape)
    print(logits)

    loss = outputs.loss
    print(loss)