import random
import numpy as np
import albumentations as A

from . import T4StackDataset

from torch.utils.data import Dataset


class T4MissingPlanesDataset(Dataset):

    def __init__(self, embryo_dir_globs=['PLACEHOLDER', 'PLACEHOLDER'],
                 plane_size=(400, 400), use_normalisation=True, use_augmentations=False, verbose=False):
        self.stack_dataset = T4StackDataset(embryo_dir_globs=embryo_dir_globs, plane_size=plane_size,
                                            use_normalisation=use_normalisation, use_augmentations=False, verbose=verbose)
        self.use_augmentations = use_augmentations
        self.augmentations = [A.RandomRotate90(p=1),
                              A.Flip(p=1)]
                            #   A.GaussNoise(var_limit=0.005, p=1),
                            #   A.Cutout(num_holes=30, max_h_size=20, max_w_size=20, p=1)]

    def __len__(self):
        return len(self.stack_dataset)

    def __getitem__(self, idx):
        plane_imgs = self.stack_dataset[idx]
        try:
            interval = random.randint(1, 1)
            label_idx = random.randint(interval, len(plane_imgs) - interval - 1)
            label = plane_imgs[label_idx][1]
            image = np.zeros(
                (self.stack_dataset.plane_size[0],
                self.stack_dataset.plane_size[1], 2)
            )
            image[:, :, 0] = plane_imgs[label_idx + interval][1]
            image[:, :, 1] = plane_imgs[label_idx - interval][1]
            if self.use_augmentations:
                image, label = self.__augment__(image, label)
        except Exception as e:
            # If there's an error, print some info
            print(len(plane_imgs))
            print(idx)
            print(e)
            return self.__getitem__(idx-1)
        return image, label

    def __augment__(self, image, label):
        # choose augmentations randomly
        aug = A.Compose(random.sample(self.augmentations,
                                      random.randint(0, len(self.augmentations))))
        augmented = aug(image=image, mask=label)
        return augmented['image'], augmented['mask']
