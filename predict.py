import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io

from models import MPNet
from datasets import T4StackDataset

model = torch.load('PLACEHOLDER')
model = model.cuda() if torch.cuda.is_available() else model
model.eval()


def predict_middle_focal(upper, lower):
    image = np.zeros((400, 400, 2))
    image[:, :, 0] = upper
    image[:, :, 1] = lower
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    image = image.cuda() if torch.cuda.is_available() else image
    pred = model.forward(image)
    pred = pred.squeeze().detach().cpu()
    return pred


def generate_interpolated_stack(stack):
    new_stack = []
    for i in range(len(stack) - 1):
        new_stack.append(stack[i])
        new_stack.append(predict_middle_focal(stack[i], stack[i + 1]))
    new_stack.append(stack[len(stack) - 1])
    return new_stack


if __name__ == '__main__':
    dataset = T4StackDataset(use_augmentations=False)
    stack = [x[1] for x in dataset[414]]
    stack = generate_interpolated_stack(stack)
    fig, ax = plt.subplots(3, math.ceil(len(stack) / 3))
    fig.set_size_inches(21, 9)
    for i in range(len(stack)):
        ax[math.floor(i / 7), i % 7].imshow(stack[i])
    fig.savefig(f'test.jpg')
    plt.close()