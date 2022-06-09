import os
import glob
import numpy as np
import albumentations as A
import random

from torch.utils.data import Dataset
from skimage import io, exposure
from skimage.transform import resize
from skimage.draw import disk

# Deal with truncated images from PLACEHOLDER
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class T4StackDataset(Dataset):
    """
    Dataset of Embryo Stacks
    """

    def __init__(self, embryo_dir_globs=['PLACEHOLDER', 'PLACEHOLDER'],
                 plane_size=(400, 400), use_normalisation=True, use_augmentations=False, verbose=False):
        self.use_augmentations = use_augmentations
        self.use_normalisation = use_normalisation
        self.verbose = verbose
        self.plane_size = plane_size
        self.augmentations = [A.RandomRotate90(p=1),
                              A.Flip(p=1),
                              A.GaussNoise(var_limit=0.005, p=1),
                              A.Cutout(num_holes=30, max_h_size=20, max_w_size=20, p=1)]

        # List all directories that contain embryo data using the patterns provided
        embryo_dirs = [
            path for embryo_dir_glob in embryo_dir_globs for path in glob.glob(embryo_dir_glob)
        ]

        self.data = [
            dict(sorted({
                int(os.path.splitext(os.path.basename(plane_path))[0][1:]): plane_path
                for plane_path in glob.glob(os.path.join(embryo_dir, '*.jpg'))
            }.items()))
            for embryo_dir in embryo_dirs
        ]

        # Generate mask for hiding the well outline
        rr, cc = disk((plane_size[0]/2, plane_size[1]/2), plane_size[0]/2)
        self.circle_mask = np.zeros(plane_size)
        self.circle_mask[rr, cc] = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        stack = self.data[idx]
        # Load planes
        depths = [depth for depth, _ in stack.items()]
        plane_imgs = [io.imread(plane_path, as_gray=True)
                      for _, plane_path in stack.items()]
        # Get image dimensions
        w, h = plane_imgs[0].shape
        # Crop out the exterior of the well
        w_rm, h_rm = int(w * 0.0625), int(h * 0.0625)
        plane_imgs = [img[w_rm:-w_rm, h_rm:-h_rm] for img in plane_imgs]
        # Resize the images
        plane_imgs = [resize(img, self.plane_size) for img in plane_imgs]
        # Mask out the edges of the well
        plane_imgs = [img * self.circle_mask for img in plane_imgs]
        # Normalise
        if self.use_normalisation:
            plane_imgs = self._normalise_stack_(plane_imgs)
        # Augment
        if self.use_augmentations:
            plane_imgs = self.__augment__(plane_imgs)
        return list(zip(depths, plane_imgs))

    def _normalise_stack_(self, plane_imgs):
        """ Apply normalisation across the whole stack by combining planes
            into single image and applying normalisation to that. """
        # Combine the images
        combined_img = np.zeros(
            (self.plane_size[0] * len(plane_imgs), self.plane_size[1]))
        for i in range(len(plane_imgs)):
            combined_img[
                i * self.plane_size[0]:(i + 1) * self.plane_size[0],
                0:self.plane_size[1]
            ] = plane_imgs[i]
        # Normalise combined image
        combined_img = exposure.equalize_adapthist(
            combined_img, clip_limit=0.01)
        # Unpack the combined image into planes
        return [combined_img[i * self.plane_size[0]:(i + 1) * self.plane_size[0], 0:self.plane_size[1]]
                for i in range(len(plane_imgs))]

    def __augment__(self, plane_imgs):
        # choose augmentations randomly
        aug = A.Compose(random.sample(self.augmentations,
                                      random.randint(0, len(self.augmentations))))
        combined_img = np.zeros(
            (self.plane_size[0], self.plane_size[1], len(plane_imgs))
        )
        for i in range(len(plane_imgs)):
            combined_img[:, :, i] = plane_imgs[i]
        augmented = aug(image=combined_img)['image']
        return [augmented[:, :, i] for i in range(len(plane_imgs))]
