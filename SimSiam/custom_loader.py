import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import random
import torchvision.transforms as T
import numpy as np
from arguments import get_args
from augmentations import get_aug


class CifarRandomEntryDataset(Dataset):
    def __init__(self, h5_path, transform=None, augment_images=False, range_of_images=(-1, -1)):
        super().__init__()
        self.file = h5py.File(h5_path, 'r', libver='latest', swmr=True)
        self.data = self.file["images"]
        
        self.transform = transform
        self.augment_images = augment_images
        self.range_of_images = range_of_images

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        slice_ = self.data[idx]  # shape: (M, C, H, W)

        original_image = slice_[0]
        if self.range_of_images[0] == -1:
            self.range_of_images[0] = 0
        if self.range_of_images[1] == -1:
            self.range_of_images[1] = slice_.shape[0] - 1
        random_idx = random.randint(self.range_of_images[0], self.range_of_images[1])
        random_entry = slice_[random_idx]
        label = random_idx

        # Convert to torch
        original_image = torch.from_numpy(original_image).float()  # Already in [-1,1]
        random_entry = torch.from_numpy(random_entry).float()     # Already in [-1,1]

        if self.transform is not None and self.augment_images:
            original_image, _ = self.transform(original_image)
            random_entry, _ = self.transform(random_entry)

        label = torch.tensor(label, dtype=torch.long)
        return (random_entry, original_image), label

def create_dataloader(
    h5_path,
    batch_size=32,
    shuffle=True,
    transform=None,
    augment_images=False,
    num_workers=4,
    pin_memory=True,
    range_of_images=(-1, -1) #Set as -1 to use all synthetic images. 
):
    dataset = CifarRandomEntryDataset(
        h5_path=h5_path,
        transform=transform,
        augment_images=augment_images,
        range_of_images=range_of_images
    )
    
    # persistent_workers keeps workers alive between epochs to avoid re-forking
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )


if __name__ == "__main__":
    args = get_args()

    train_loader = create_dataloader(
        h5_path="latest_cifar10_dataset.h5",
        batch_size=1,
        shuffle=True,
        transform=get_aug(train=True, **args.aug_kwargs),
        augment_original_image=True, #Need to verify this works.
        num_workers=8,
        range_of_images=(1, 1) #Set as -1 to use all synthetic images. 
    )
    for idx, ((images1, images2), labels) in enumerate(train_loader):
        print(images1.shape)
        print(images2.shape)
        print(labels.shape)
        break
