import wandb
import torch

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from matplotlib import pyplot as plt

from datasets import T4PlaneExtrapolationDataset
from models import PerceptualAE

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)

def train(model, loader, optimiser, scheduler,
          max_epoch=20, log_interval=5):
    model.train()
    batches = 0

    for epoch in tqdm(range(max_epoch)):
        for batch in tqdm(loader):
            batches += 1
            # Load batch
            _, img = batch
            img = img.cuda() if torch.cuda.is_available() else img
            img = img.unsqueeze(1).float()
            # Train
            optimiser.zero_grad()
            pred = model.forward(img)
            loss = torch.mean((pred - img) ** 2)
            loss.backward()
            optimiser.step()
            # Logging
            if batches % log_interval:
                wandb.log({
                    'Batches': batches
                })
                wandb.log({
                    'MSE Loss': loss.detach().cpu().item()
                })
            if batches % 100 == 0:
                plt.figure()
                plt.imshow(pred.detach().cpu()[0].squeeze())
                plt.savefig(f'ae_bigger_{batches}.jpg')
                plt.close()
            if batches % 1000 == 0:
                torch.save(model, f'model_ae_bigger_{batches}.ckpt')
            # Step LR
            scheduler.step()


if __name__ == '__main__':
    wandb.init(project='PLACEHOLDER', entity='PLACEHOLDER')
    model = PerceptualAE()
    model = model.cuda() if torch.cuda.is_available() else model
    model.apply(weights_init)
    dataset = T4PlaneExtrapolationDataset(
        embryo_dir_globs=['PLACEHOLDER'], 
        use_augmentations=True
    )
    loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=4
    )
    optimiser = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )
    scheduler = optim.lr_scheduler.StepLR(optimiser, 3000, 0.1)
    train(model, loader, optimiser, scheduler)