import os
import glob
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io

# Imports for image preprocessing
from skimage import io, exposure
from skimage.transform import resize
from skimage.draw import disk

# For calculating SSIM
from ssim import ssim

# Allow loading of truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_stack(stack_dir_path):
    """ Load a stack from the containing directory path. """
    img_paths = glob.glob(os.path.join(stack_dir_path, '*.jpg'))
    # Create list of images tagged with their focal depths
    imgs = [
        (
            int(os.path.splitext(os.path.basename(img_path))[0][1:]),
            io.imread(img_path, as_gray=True)
        )
        for img_path in img_paths
    ]
    # Sort by focal depth
    imgs = sorted(imgs, key=lambda x: x[0])
    # Remove the focal depth info
    imgs = [img for _, img in imgs]
    return imgs

def normalise_stack(imgs, plane_size):
    """ Apply normalisation across the whole stack by combining planes
        into single image and applying normalisation to that. """
    # Combine the images
    combined_img = np.zeros(
        (plane_size[0] * len(imgs), plane_size[1]))
    for i in range(len(imgs)):
        combined_img[i * plane_size[0]:(i + 1) * plane_size[0], 0:plane_size[1]] = imgs[i]
    # Normalise combined image
    combined_img = exposure.equalize_adapthist(
        combined_img, clip_limit=0.01)
    # Unpack the combined image into planes
    return [combined_img[i * plane_size[0]:(i + 1) * plane_size[0], 0:plane_size[1]]
            for i in range(len(imgs))]


def preprocess_stack(imgs, plane_size=(400, 400), crop_proportion=0.0625):
    """ Preprocess a stack. """
    # Generate mask for hiding the well outline
    rr, cc = disk((plane_size[0]/2, plane_size[1]/2), plane_size[0]/2)
    circle_mask = np.zeros(plane_size)
    circle_mask[rr, cc] = 1
    # Center-crop, resize and apply mask to all images
    h, w = imgs[0].shape
    h_rm, w_rm = int(h * crop_proportion), int(w * crop_proportion)
    imgs = [img[h_rm:-h_rm, w_rm:-w_rm] for img in imgs]
    imgs = [resize(img, plane_size) for img in imgs]
    imgs = [img * circle_mask for img in imgs]
    # Normalise stack
    imgs = normalise_stack(imgs, plane_size)
    return imgs

def predict_with_model(model, first, second):
    image = np.zeros((400, 400, 2))
    image[:, :, 0] = first
    image[:, :, 1] = second
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    image = image.cuda() if torch.cuda.is_available() else image
    pred = model.forward(image)
    pred = pred.squeeze().detach().cpu()
    return pred

rr, cc = disk((200, 200), 200)
circle_mask = np.zeros((400, 400))
circle_mask[rr, cc] = 1

# Find all the stacks of interest
stack_dirs = glob.glob('PLACEHOLDER')[20:10020]
scores = []
model_paths = [
    # TODO: ADD SOME MODELS
]
k = 0
for j, model_path in enumerate(model_paths):
    os.mkdir(str(j+4))
    os.mkdir(os.path.join(str(j+4), 'real'))
    os.mkdir(os.path.join(str(j+4), 'fake'))
    model = torch.load(model_path)
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()
    # Loop through and generate
    for i, stack_dir in enumerate(stack_dirs):
        k += 1
        # Load the stack
        stack = preprocess_stack(load_stack(stack_dir))
        # Generate them all
        for idx in range(2, len(stack)):
            plane = predict_with_model(
                model, stack[idx - 1], stack[idx - 2]
            ) * circle_mask
            actual_plane = stack[idx]
            io.imsave(os.path.join(str(j+4), 'real', f'{k}.jpg'), plane.numpy())
            io.imsave(os.path.join(str(j+4), 'fake', f'{k}.jpg'), stack[idx])
