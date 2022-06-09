import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io

# Imports for image preprocessing
from skimage import io, exposure
from skimage.transform import resize
from skimage.draw import disk


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


def predict_with_model(model, first, second):
    image = np.zeros((400, 400, 2))
    image[:, :, 0] = first
    image[:, :, 1] = second
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    image = image.cuda() if torch.cuda.is_available() else image
    pred = model.forward(image)
    pred = pred.squeeze().detach().cpu()
    return pred


def generate_stack_from_path(decreasing_model, increasing_model, stack_dir):
    # Load the stack
    stack = preprocess_stack(load_stack(stack_dir))
    # Set output directory
    output_path = os.path.basename(stack_dir)
    os.mkdir(output_path)
    focals = ['F-75', 'F-60', 'F-45', 'F-30', 'F-15',
              'F0', 'F15', 'F30', 'F45', 'F60', 'F75']
    # Get new focal planes if we are missing some
    if len(stack) == 7:
        focals = ['F-45', 'F-30', 'F-15', 'F0', 'F15', 'F30', 'F45']
        plane_plus_60 = predict_with_model(
            increasing_model, stack[6], stack[5]
        )
        plane_plus_75 = predict_with_model(
            increasing_model, plane_plus_60, stack[6]
        )
        plane_minus_60 = predict_with_model(
            decreasing_model, stack[0], stack[1]
        )
        plane_minus_75 = predict_with_model(
            decreasing_model, plane_minus_60, stack[0]
        )
        # Save the new stack
        io.imsave(os.path.join(output_path, 'F60.jpg'), plane_plus_60.numpy())
        io.imsave(os.path.join(output_path, 'F75.jpg'), plane_plus_75.numpy())
        io.imsave(os.path.join(output_path, 'F-60.jpg'),
                  plane_minus_60.numpy())
        io.imsave(os.path.join(output_path, 'F-75.jpg'),
                  plane_minus_75.numpy())
    # Save the new stack
    for i, img in enumerate(stack):
        io.imsave(os.path.join(output_path, f'{focals[i]}.jpg'), img)


if __name__ == '__main__':
    # Load models
    decreasing_model = torch.load('PLACEHOLDER')
    decreasing_model = decreasing_model.cuda(
    ) if torch.cuda.is_available() else decreasing_model
    decreasing_model.eval()
    increasing_model = torch.load('PLACEHOLDER')
    increasing_model = increasing_model.cuda(
    ) if torch.cuda.is_available() else increasing_model
    increasing_model.eval()
    # Find all the stacks of interest
    stack_dirs = glob.glob('PLACEHOLDER')
    # Loop through and generate
    failures = []
    for stack_dir in stack_dirs:
        try:
            generate_stack_from_path(
                decreasing_model, increasing_model, stack_dir
            )
        except Exception:
            failures.append(stack_dir)
    # Print failure cases
    print(f'Failures: {failures}')
