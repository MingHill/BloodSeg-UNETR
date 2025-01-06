import numpy as np

from matplotlib import pyplot as plt


def plot_mae(mae_dict, resolution="super", fname=None):
    original_image = mae_dict["original_image"].transpose(2, 0, 1)
    image_masked = mae_dict["image_masked"].transpose(2, 0, 1)
    reconstructed_image = mae_dict["reconstructed_image"].transpose(2, 0, 1)

    if resolution == "super":
        x = (
            original_image[:12]
            .reshape(2, 2, 3, 64, 64)
            .transpose(3, 0, 4, 1, 2)
            .reshape(128, 128, 3)
        )
        m = (
            image_masked[:12]
            .reshape(2, 2, 3, 64, 64)
            .transpose(3, 0, 4, 1, 2)
            .reshape(128, 128, 3)
        )
        y = (
            reconstructed_image[:12]
            .reshape(2, 2, 3, 64, 64)
            .transpose(3, 0, 4, 1, 2)
            .reshape(128, 128, 3)
        )
    elif resolution == "low":
        x = original_image[12:15].transpose(1, 2, 0)
        m = image_masked[12:15].transpose(1, 2, 0)
        y = reconstructed_image[12:15].transpose(1, 2, 0)
    elif resolution == "violet":
        x = original_image[15]
        m = image_masked[15]
        y = reconstructed_image[15]
    else:
        raise ValueError(
            "Invalid value for resolution. Expected 'super', 'low', or 'violet'."
        )

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 12))

    ax[0].imshow(
        np.clip(m, 0, 1), cmap="gray" if resolution == "violet" else None
    )  # masked
    ax[0].axis("off")

    ax[1].imshow(
        np.clip(y, 0, 1), cmap="gray" if resolution == "violet" else None
    )  # reconstructed
    ax[1].axis("off")

    ax[2].imshow(
        np.clip(x, 0, 1), cmap="gray" if resolution == "violet" else None
    )  # original
    ax[2].axis("off")

    plt.tight_layout(pad=0.25)
    if fname is not None:
        plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.show()


# function to view output prediction in view_out.py
def visualize_predictions(num_images, images, predictions, labels):
    plt.figure(figsize=(15, num_images * 5))
    for i in range(num_images):
        # Original image
        plt.subplot(num_images, 3, i * 3 + 1)
        image = images[i].transpose(1, 2, 0)[:, :, :3]
        # plt.imshow(images[i].transpose(02, 01, 0))  # Assuming the image shape is (C, H, W)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

        # Ground truth mask
        plt.subplot(num_images, 3, i * 3 + 2)
        plt.imshow(labels[i])
        plt.title("Ground Truth Mask")
        plt.axis("off")

        # Predicted mask
        plt.subplot(num_images, 3, i * 3 + 3)
        plt.imshow(predictions[i][:, :])
        plt.title("Predicted Mask")
        plt.axis("off")
    plt.show()


# function to visualize inference on in inference.ipynb
def visualize_inference(images, predictions, num_images=1):
    plt.figure(figsize=(15, num_images * 5))
    for i in range(num_images):
        # Original image
        plt.subplot(num_images, 3, i * 3 + 1)
        image = images[i].transpose(1, 2, 0)[:, :, :3]
        # plt.imshow(images[i].transpose(02, 01, 0))  # Assuming the image shape is (C, H, W)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

        # Predicted mask
        plt.subplot(num_images, 3, i * 3 + 3)
        plt.imshow(predictions[i][:, :])
        plt.title("Predicted Mask")
        plt.axis("off")
    plt.show()
