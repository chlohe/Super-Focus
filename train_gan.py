import torch
import random
import wandb

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt

from datasets import T4PlaneExtrapolationDataset, T4MissingPlanesDataset
from models import MPNet, Discriminator
from ssim import ssim

REAL = [0.9, 0]
FAKE = [0, 0.9]


conv_small = nn.Conv2d(1, 32, 3, stride=3)
conv_small = conv_small.cuda() if torch.cuda.is_available() else conv_small
conv_big = nn.Conv2d(32, 64, 3, stride=3)
conv_big = conv_big.cuda() if torch.cuda.is_available() else conv_big
conv_bigger = nn.Conv2d(64, 64, 3, stride=3)
conv_bigger = conv_bigger.cuda() if torch.cuda.is_available() else conv_bigger

nn.init.uniform_(conv_small.weight, -1, 1)
nn.init.uniform_(conv_big.weight, -1, 1)
nn.init.uniform_(conv_bigger.weight, -1, 1)



def get_feature_maps(x):
    small_map = F.relu(conv_small(x), inplace=True)
    big_map = F.relu(conv_big(small_map), inplace=True)
    bigger_map = F.relu(conv_bigger(big_map), inplace=True)
    return small_map, big_map, bigger_map


def load_perceptual_ae():
    m = torch.load('PLACEHOLDER').encoder
    m.eval()
    return m

feature_model = load_perceptual_ae()


def feature_loss(pred, target, features='ae'):
    if features == 'random':
        # Feature loss with random features
        small_pred_map, big_pred_map, bigger_pred_map = get_feature_maps(pred)
        small_target_map, big_target_map, bigger_target_map = get_feature_maps(
            target)
        small_diff = ((small_pred_map - small_target_map) ** 2).mean()
        big_diff = ((big_pred_map - big_target_map) ** 2).mean()
        bigger_diff = ((bigger_pred_map - bigger_target_map) ** 2).mean()
        loss = small_diff + big_diff + bigger_diff
    elif features == 'ae':
        # Feature loss with autoencoder features - this is what is used in the final paper
        pred_map = feature_model.forward(pred)
        target_map = feature_model.forward(target)
        diff = ((pred_map - target_map) ** 2).mean()
        loss = diff
    else:
        raise Exception('Invalid feature source for feature loss!')
    return loss


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)


def train(model, discriminator_model, loader, optimiser, discriminator_optimiser,
          max_epoch=2, log_interval=5, checkpoint_interval=500, gen_train_interval=10, scheduler_interval=15000,
          scheduler=None, discriminator_scheduler=None):
    # Init model weights
    model.train()
    discriminator_model.train()

    # Training Loop
    batches = 0
    for epoch in tqdm(range(max_epoch)):
        for batch in tqdm(loader):
            batches += 1
            # Load batch
            img, target = batch
            img = img.cuda() if torch.cuda.is_available() else img
            target = target.cuda() if torch.cuda.is_available() else target
            img = img.permute(0, 3, 1, 2).float()
            target = target.unsqueeze(1).float()
            # =============== Train generator ==========================
            optimiser.zero_grad()
            # Forward pass
            pred = model.forward(img)
            discriminator_pred = discriminator_model.forward(pred)
            # Set up adversarial target
            adversarial_target = torch.tensor(
                [REAL for _ in range(len(img))]
            ).float()
            adversarial_target = adversarial_target.cuda(
            ) if torch.cuda.is_available() else adversarial_target
            # Compute losses
            mse_loss = torch.mean((pred - target) ** 2)
            feat_loss = feature_loss(pred, target, features='ae')
            ssim_loss = -ssim(pred, target)
            adv_loss = F.binary_cross_entropy(
                discriminator_pred[:len(img)],
                adversarial_target
            )
            if batches > 500:
                # Enable adversarial training
                loss = adv_loss + 100 * mse_loss  + 10 * feat_loss
            else:
                # Pretraining
                loss = 100 * mse_loss + 10 * feat_loss
            # Optimiser step
            loss.backward()
            optimiser.step()
            # ============== Train discriminator ======================
            if batches % gen_train_interval == 0:
                discriminator_optimiser.zero_grad()
                # Randomly choose whether to train on a real or fake batch
                if batches > 500:
                    if random.random() < 0.5:
                        discriminator_pred = discriminator_model.forward(
                            torch.cat([pred.detach()])
                        )
                        discriminator_target = torch.cat([
                            torch.tensor([FAKE for _ in range(len(img))])
                        ]).float()
                    else:
                        discriminator_pred = discriminator_model.forward(
                            torch.cat([target])
                        )
                        discriminator_target = torch.cat([
                            torch.tensor([REAL for _ in range(len(img))])
                        ]).float()
                else:
                    discriminator_pred = discriminator_model.forward(
                        torch.cat([target, pred.detach()])
                    )
                    discriminator_target = torch.cat([
                        torch.tensor([REAL for _ in range(len(img))] +
                                     [FAKE for _ in range(len(img))])
                    ]).float()
                discriminator_target = discriminator_target.cuda(
                ) if torch.cuda.is_available() else discriminator_target
                discriminator_loss = F.binary_cross_entropy(
                    discriminator_pred,
                    discriminator_target
                )
                discriminator_loss.backward()
                discriminator_optimiser.step()
                if batches % log_interval == 0:
                    wandb.log({
                        'Batches': batches
                    })
                    wandb.log({
                        'Discriminator Loss': discriminator_loss.detach().cpu().item()
                    })
                    wandb.log({
                        'Discriminator Accuracy':
                        (torch.argmax(discriminator_target, axis=1) == torch.argmax(
                            discriminator_pred.squeeze(), axis=1)).float().sum() / len(discriminator_target)
                    })
            # Logging
            if batches % log_interval == 0:
                wandb.log({
                    'Generator Loss': loss.detach().cpu().item()
                })
                wandb.log({
                    'MSE Loss': mse_loss.detach().cpu().item()
                })
                wandb.log({
                    'Feature Loss': feat_loss.detach().cpu().item()
                })
                wandb.log({
                    'Adversarial Loss': adv_loss.detach().cpu().item()
                })
                wandb.log({
                    'SSIM':
                    ssim(pred, target).detach().cpu().item()
                }) 
            # Uncomment to plot pictures
            if batches % (log_interval * 40) == 1:
                # Plot images
                fig, ax = plt.subplots(2, 2)
                input_img = img.detach().cpu()[0]
                pred_img = pred.detach().cpu()[0].squeeze()
                target_img = target.detach().cpu()[0].squeeze()
                ax[0, 0].imshow(input_img[0, :, :])
                # ax[0, 0].imshow(start_model.forward(img).detach().cpu()[0].squeeze())
                ax[0, 1].imshow(input_img[1, :, :])
                ax[1, 0].imshow(pred_img)
                ax[1, 1].imshow(target_img)
                fig.savefig(f'mid_{epoch}_{batches}.jpg', dpi=300)
                plt.close()
            if batches % checkpoint_interval == 0:
                torch.save(model, f'mid_model_{batches}.ckpt')
                torch.save(discriminator_model,
                        f'mid_discriminator_model_{batches}.ckpt')
            # LR scheduling
            if batches % scheduler_interval == 0 and batches != 0:
                print("LR DROPPED")
                if scheduler is not None:
                    scheduler.step()
                if discriminator_scheduler is not None:
                    discriminator_scheduler.step()


if __name__ == '__main__':
    wandb.init(project='PLACEHOLDER', entity='PLACEHOLDER')
    model = MPNet()
    model = model.cuda() if torch.cuda.is_available() else model
    model.apply(weights_init)
    discriminator_model = Discriminator()
    discriminator_model = discriminator_model.cuda(
    ) if torch.cuda.is_available() else discriminator_model
    discriminator_model.apply(weights_init)
    dataset = T4MissingPlanesDataset(
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
        lr=1e-3
    )
    discriminator_optimiser = optim.Adam(
        filter(lambda p: p.requires_grad, discriminator_model.parameters()),
        lr=1e-3
    )
    scheduler = optim.lr_scheduler.StepLR(optimiser, 1, 0.1)
    discriminator_scheduler = optim.lr_scheduler.StepLR(
        discriminator_optimiser, 1, 0.1)
    train(model, discriminator_model, loader,
          optimiser, discriminator_optimiser, scheduler=scheduler, discriminator_scheduler=discriminator_scheduler, max_epoch=80, 
          scheduler_interval=15000
        )
