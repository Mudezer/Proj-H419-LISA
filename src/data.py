import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

class SegmentationDataset(Dataset):
    '''''
    This class is used to create a dataset for image segmentation.
    takes as input:
    - image_dir: the directory where the images are stored
    - mask_dir: the directory where the masks are stored
    - transform: the transformations to be applied to the images and masks
    output:
    - either the image and mask as numpy arrays of type float32
    - or the image and mask as torch tensors after the application of the transformations
    '''
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        # load the image and mask as numpy arrays of type float32
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # convert the mask to binary (0 and 1) => 0: background, 1: object (binary segmentation = binary classification)
        mask[mask == 255.0] = 1.0

        # apply the transformations to the image and mask
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask