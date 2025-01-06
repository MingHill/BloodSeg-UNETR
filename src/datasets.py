import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, data, num_channels=16):
        super().__init__()
        self.num_channels = num_channels
        self.data = data

    def __getitem__(self, index):
        return self.data[index][: self.num_channels, :, :]

    def __len__(self):
        return len(self.data)


# class UnetLabelDataset(Dataset):
#     def __init__(self, data):
#         super().__init__()
#         self.data = data

#     def __getitem__(self, index):
#         return self.data[index]

#     def __len__(self):
#         return len(self.data)


class UnetCustomDataset(Dataset):
    def __init__(self, images, labels, num_channels=16):
        self.images = images
        self.labels = labels
        self.num_channels = num_channels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx][: self.num_channels, :, :]
        label = self.labels[idx]
        return image, label


class CustomCIFAR10Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index][0]

        if self.transform is not None:
            image = self.transform(image)

        return image


class CustomTransform:
    def __init__(self):
        self.transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        )

    def __call__(self, image, label):
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        image = self.transform(image)

        torch.random.manual_seed(seed)
        label = self.transform(label)

        return image, label


def create_patches(image, patch_size=64, stride=64):
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    # image shape: [c, h, w]
    c, h, w = image.size()

    # patches shape: [c, num_patches_h, num_patches_w, patch_size, patch_size]
    patches = image.unfold(1, patch_size, stride).unfold(2, patch_size, stride)

    # patches shape: [num_patches_h, num_patches_w, c, patch_size, patch_size]
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()

    # patches shape: [num_patches, c, patch_size, patch_size]
    patches = patches.view(-1, c, patch_size, patch_size)

    return patches


def collate_fn_train(batch):
    transform = transforms.Compose(
        [
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
        ]
    )

    stride = 64
    patch_size = 64
    transformed_patches = []

    for sample in batch:
        sample_patches = create_patches(sample, patch_size, stride)
        sample_transformed_patches = [transform(patch) for patch in sample_patches]
        transformed_patches.append(torch.stack(sample_transformed_patches))

    stacked_transformed_patches = torch.cat(transformed_patches, dim=0) / 255

    return stacked_transformed_patches


def collate_fn_valid_test(batch):
    stride = 64
    patch_size = 64
    patches = []

    for sample in batch:
        sample_patches = create_patches(sample, patch_size, stride)
        patches.append(sample_patches)

    stacked_patches = torch.cat(patches, dim=0) / 255

    return stacked_patches


def unet_image_collate_fn_valid_test(batch):
    batch = [torch.tensor(item) for item in batch]
    stacked_batch = torch.stack(batch)
    stacked_batch = stacked_batch.float() / 255.0
    return stacked_batch


def unet_valid_collate(batch):
    images, labels = zip(*batch)
    images = unet_image_collate_fn_valid_test(images)
    labels = [torch.tensor(label) for label in labels]
    labels = torch.stack(labels, dim=0).long()
    return images, labels


def unet_train_collate(batch):
    custom_transform = CustomTransform()
    images, labels = zip(*batch)

    transformed_images = []
    transformed_labels = []

    for img, lbl in zip(images, labels):
        img_tensor = torch.tensor(img).float() / 255.0
        lbl_tensor = torch.tensor(lbl).long()

        transformed_img, transformed_lbl = custom_transform(img_tensor, lbl_tensor)

        transformed_images.append(transformed_img)
        transformed_labels.append(transformed_lbl)

    images = torch.stack(transformed_images)
    labels = torch.stack(transformed_labels)

    return images, labels


def unet_inference(batch):
    batch = [torch.tensor(item) for item in batch]
    stacked_batch = torch.stack(batch)
    stacked_batch = stacked_batch.float() / 255.0
    return stacked_batch


if __name__ == "__main__":
    # transform_train = transforms.Compose([
    #     # transforms.Resize((48, 48)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    # ])
    #
    # transform_valid = transforms.Compose([
    #     # transforms.Resize((48, 48)),
    #     transforms.ToTensor(),
    # ])
    #
    # transform_test = transforms.Compose([
    #     # transforms.Resize((48, 48)),
    #     transforms.ToTensor(),
    # ])
    #
    # dataset = datasets.CIFAR10(root="/Users/martin/Sites/datasets/third-party/pytorch-datasets", train=True)
    # train_data, valid_data = random_split(dataset, [40000, 10000])
    #
    # test_data = datasets.CIFAR10(root="/Users/martin/Sites/datasets/third-party/pytorch-datasets", train=False)
    #
    # train_dataset = CustomCIFAR10Dataset(train_data, transform=transform_train)
    # valid_dataset = CustomCIFAR10Dataset(valid_data, transform=transform_valid)
    # test_dataset = CustomCIFAR10Dataset(test_data, transform=transform_test)
    #
    # train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    ###################################################################################################################

    training_dataset = np.load(
        "/home/mhill/Projects/cathepsin/data/training_dataset.npy"
    )
    validation_dataset = np.load(
        "/home/mhill/Projects/cathepsin/data/validation_dataset.npy"
    )
    testing_dataset = np.load("/home/mhill/Projects/cathepsin/data/testing_dataset.npy")

    train_dataset = CustomDataset(training_dataset)
    valid_dataset = CustomDataset(validation_dataset)
    test_dataset = CustomDataset(testing_dataset)

    train_dataloader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_train
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_valid_test
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_valid_test
    )

    for data in train_dataloader:
        print(data.size())
        break

    for data in valid_dataloader:
        print(data.size())
        break

    for data in test_dataloader:
        print(data.size())
        break

    # for data in train_dataset:
    #     plt.imshow(data[:3, :, :].transpose(01, 02, 0))
    #     # plt.imshow(data.permute(01, 02, 0))
    #     plt.axis('off')
    #     plt.show()
    #     break
    #
    # for data in valid_dataset:
    #     plt.imshow(data[:3, :, :].transpose(01, 02, 0))
    #     # plt.imshow(data.permute(01, 02, 0))
    #     plt.axis('off')
    #     plt.show()
    #     break
    #
    # for data in test_dataset:
    #     plt.imshow(data[:3, :, :].transpose(01, 02, 0))
    #     # plt.imshow(data.permute(01, 02, 0))
    #     plt.axis('off')
    #     plt.show()
    #     break
