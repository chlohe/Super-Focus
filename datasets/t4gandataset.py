import torch

from . import T4MissingPlanesDataset

from torch.utils.data import Dataset


class T4GANDataset(Dataset):

    def __init__(self, generator_model, embryo_dir_globs=['PLACEHOLDER*', 'PLACEHOLDER'],
                 plane_size=(400, 400), use_normalisation=True, use_augmentations=False, verbose=False):
        self.generator_model = generator_model
        self.generator_model = self.generator_model.cuda(
        ) if torch.cuda.is_available() else self.generator_model
        self.generator_model.eval()
        self.missing_planes_dataset = T4MissingPlanesDataset(embryo_dir_globs=embryo_dir_globs, plane_size=plane_size,
                                                             use_normalisation=use_normalisation, use_augmentations=use_augmentations, verbose=verbose)

    def __len__(self):
        return len(self.missing_planes_dataset)

    def __getitem__(self, idx):
        image, real = self.missing_planes_dataset[idx]
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        image = image.cuda() if torch.cuda.is_available() else image
        fake = self.generator_model.forward(
            image
        ).detach().squeeze().cpu().numpy()
        return real, fake
