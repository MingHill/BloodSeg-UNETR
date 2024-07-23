import pytest
import numpy as np

import torch
from transformers import ViTMAEForPreTraining


def select_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    return device


# see modeling_vit_mae.py from transformer package
def extract_patch_mean_var(patches):
    mean = patches.mean(dim=-1, keepdim=True)
    var = patches.var(dim=-1, keepdim=True)
    return mean, var


# see modeling_vit_mae.py from transformer package
def denormalize_patch_values(model, patches, mean, var):
    return patches * (var + 1.0e-6) ** 0.5 + mean


# see https://github.com/NielsRogge/Transformers-Tutorials/blob/master/ViTMAE/ViT_MAE_visualization_demo.ipynb
@pytest.mark.usefixtures(['patchify','unpatchify'])
def extract_mae(model, dataset, image_index, device, denormalize=True):
    model = model.to(device)
    pixel_values = dataset[image_index].unsqueeze(dim=0).to(device)

    # Patchify pixel values and normalize
    # patchified_pixel_values = model.patchify(pixel_values)
    # mean, var = extract_patch_mean_var(patchified_pixel_values)

    with torch.inference_mode():
        model.eval()
        outputs = model(pixel_values, output_hidden_states = True)  # assuming model handles normalization internally
        logits = outputs.logits

    hidden_states = outputs.hidden_states
    print(f"Output Hidden state size {outputs.hidden_states[0].shape}")
    mask = outputs.mask
    mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size ** 2 * model.config.num_channels)
    mask = model.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    # original image
    x = torch.einsum('nchw->nhwc', pixel_values).detach().cpu()

    # reconstructed image
    if denormalize:
        patchified_pixel_values = model.patchify(pixel_values)
        mean, var = extract_patch_mean_var(patchified_pixel_values)
        y = denormalize_patch_values(model, logits, mean, var)
        y = model.unpatchify(y)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()
    else:
        y = model.unpatchify(logits)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # image masked
    image_masked = x * (1 - mask)

    # reconstructed image with visible patches
    image_reconstructed_visible = x * (1 - mask) + y * mask

    mae = {
        'original_image': x.squeeze(dim=0).numpy(),
        'image_masked': image_masked.squeeze(dim=0).numpy(),
        'reconstructed_image': y.squeeze(dim=0).numpy(),
        'image_reconstructed_visible': image_reconstructed_visible.squeeze(dim=0).numpy()
    }

    return mae
