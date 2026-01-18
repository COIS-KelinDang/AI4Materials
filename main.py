from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import torch
import torch.optim as optim
import os
from network import Patch_Trans_Discriminator, Conv_Trans_Generator
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.backends import cudnn
import random
from torch.autograd import Variable
import utils
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--image_size", type=int, default=256, help="Size of image for discriminator input.")
parser.add_argument("--initial_size", type=int, default=8, help="Initial size for generator.")
parser.add_argument("--patch_size", type=int, default=16, help="Patch size for generated image.")
parser.add_argument('--train_dataset', type=str, default=r'datasets_new')
parser.add_argument('--weights_save', type=str, default=r'weights')
parser.add_argument('--log_path', type=str, default=r'logs')
parser.add_argument("--lr_gen", type=float, default=0.0001, help="Learning rate for generator.")
parser.add_argument("--lr_dis", type=float, default=0.0004, help="Learning rate for discriminator.")
parser.add_argument("--inner_g", type=int, default=3)
parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay.")
parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension.")
parser.add_argument("--num_head", type=int, default=4, help="the heads num of atention.")
parser.add_argument("--mlp_ratio", type=int, default=4, help="hidden layer ratio params.")
parser.add_argument("--batch_size", type=int, default=40, help="Batch size for generator.")
parser.add_argument("--epoch", type=int, default=2000, help="Number of epoch.")
parser.add_argument("--g_dim", type=int, default=1024, help="Embedding dimension.")
parser.add_argument("--d_dim", type=int, default=384, help="Embedding dimension.")
parser.add_argument("--optim", type=str, default="Adam", help="Choose your optimizer")
parser.add_argument("--diff_aug", type=str, default="translation,cutout,color", help="Data Augmentation")
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()

generator = Conv_Trans_Generator(
    depth1=1,
    depth2=1,
    depth3=2,
    initial_size=args.initial_size,
    dim=args.g_dim,
    heads=args.num_head,
    mlp_ratio=args.mlp_ratio,
    drop_rate=0.5,
)
generator.to(args.device)

discriminator = Patch_Trans_Discriminator(
    diff_aug=args.diff_aug,
    image_size=args.image_size,
    patch_size=args.patch_size,
    input_channel=3,
    num_classes=1,
    dim=args.d_dim,
    depth=7,
    heads=args.num_head,
    mlp_ratio=args.mlp_ratio,
    drop_rate=0.0,
)
discriminator.to(args.device)

generator = torch.nn.DataParallel(generator)
discriminator = torch.nn.DataParallel(discriminator)

if args.optim == "Adam":
    optim_gen = optim.Adam(
        generator.parameters(),
        lr=args.lr_gen,
        betas=(0.5, 0.999))

    optim_dis = optim.Adam(
        discriminator.parameters(),
        lr=args.lr_dis,
        betas=(0.5, 0.999))

elif args.optim == "SGD":
    optim_gen = optim.SGD(
        generator.parameters(),
        lr=args.lr_gen,
        momentum=0.9,
    )

    optim_dis = optim.SGD(
        discriminator.parameters(),
        lr=args.lr_dis,
        momentum=0.9,
    )

elif args.optim == "RMSprop":
    optim_gen = optim.RMSprop(
        generator.parameters(),
        lr=args.lr_gen,
        eps=1e-08,
        weight_decay=args.weight_decay,
        momentum=0,
        centered=False,
    )

    optim_dis = optim.RMSprop(
        discriminator.parameters(),
        lr=args.lr_dis,
        eps=1e-08,
        weight_decay=args.weight_decay,
        momentum=0,
        centered=False,
    )


def compute_gradient_penalty(D, real_samples, fake_samples, phi):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(
        real_samples.get_device()
    )
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates, _ = D(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(
        real_samples.get_device()
    )
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        # create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    grad_norm = gradients.norm(2, dim=1)
    # 防止 grad_norm 为 nan
    grad_norm = torch.clamp(grad_norm, min=0, max=10)
    gradient_penalty = ((grad_norm - phi) ** 2).mean()

    # gradients = gradients.contiguous().view(gradients.size(0), -1)
    # gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty


gen_scheduler = torch.optim.lr_scheduler.StepLR(optim_gen, 5, gamma=0.98)
dis_scheduler = torch.optim.lr_scheduler.StepLR(optim_dis, 5, gamma=0.98)

train_set = utils.TrainGANDataset(args.train_dataset, args.image_size, train=True)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=args.batch_size,
    shuffle=True
)

real_label = 1
fake_label = 0
adversarial_loss = torch.nn.BCELoss()

d_loss_list, g_loss_list = [], []
for epoch in range(args.epoch):
    generator = generator.train()
    discriminator = discriminator.train()

    d_loss_all, g_loss_all = 0, 0
    for index, imgs in enumerate(train_loader):
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(args.device)
        fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).to(args.device)

        # Configure input
        real_imgs = Variable(imgs.type(torch.FloatTensor)).to(args.device)

        # -----------------
        #  Train Generator
        # -----------------
        for _ in range(args.inner_g):
            optim_gen.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, args.latent_dim)))).to(args.device)

            # Generate a batch of images
            gen_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            validity, _ = discriminator(gen_imgs)
            g_loss = adversarial_loss(F.sigmoid(validity), valid)

            g_loss.backward()
            optim_gen.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optim_dis.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = adversarial_loss(F.sigmoid(real_pred), valid)

        # Loss for fake images
        fake_pred, _ = discriminator(gen_imgs.detach())
        d_fake_loss = adversarial_loss(F.sigmoid(fake_pred), fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        d_adv_acc = 0.5 * sum(sum(torch.round(F.sigmoid(fake_pred)) == fake)) / batch_size + 0.5 * sum(
            sum(torch.round(F.sigmoid(real_pred)) == valid)) / batch_size
        g_adv_acc = sum(sum(torch.round(F.sigmoid(fake_pred)) == fake)) / batch_size
        d_loss.backward()
        optim_dis.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, adv_acc: %d%%] [G loss: %f, adv_acc: %d%%]"
            % (epoch + 1, args.epoch, index, len(train_loader), d_loss.item(), 100 * d_adv_acc, g_loss.item(),
               100 * g_adv_acc)
        )

        d_loss_all += d_loss.item() * imgs.shape[0]
        g_loss_all += g_loss.item() * imgs.shape[0]
    d_loss_list.append(d_loss_all / len(train_set))
    g_loss_list.append(g_loss_all / len(train_set))

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, len(d_loss_list)), np.array(d_loss_list))
    plt.legend(['D Loss'])
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(0, len(g_loss_list)), np.array(g_loss_list))
    plt.legend(['G Loss'])
    plt.grid()
    plt.savefig(os.path.join(args.log_path, 'Loss_Details.png'))

    plt.figure(2)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        with torch.no_grad():
            generator.eval()
            discriminator.eval()
            test_z = Variable(torch.FloatTensor(np.random.normal(0, 1, (16, args.latent_dim)))).to(args.device)
            generated_imgs = generator(test_z).cpu().view(-1, 3, args.image_size, args.image_size)

            # 可视化结果
            fig, axes = plt.subplots(4, 4, figsize=(6, 6))
            for i, ax in enumerate(axes.flat):
                save_img = generated_imgs[i]
                save_img = save_img * 0.5 + 0.5
                ax.imshow(np.uint8(save_img.cpu().detach() * 255).transpose(1, 2, 0))
                ax.axis('off')
            plt.savefig(os.path.join(args.log_path, 'Epoch_%d_Visual_Images_fake.png' % (epoch + 1)))

    plt.figure(3)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        with torch.no_grad():
            img_set = []
            for i in range(16):
                idx = np.random.randint(0, len(train_set))
                img_set.append(train_set[idx])

            # 可视化结果
            fig, axes = plt.subplots(4, 4, figsize=(6, 6))
            for i, ax in enumerate(axes.flat):
                save_img = img_set[i]
                save_img = save_img * 0.5 + 0.5
                ax.imshow(np.uint8(save_img.cpu().detach() * 255).transpose(1, 2, 0))
                ax.axis('off')
            plt.savefig(os.path.join(args.log_path, 'Epoch_%d_Visual_Images_real.png' % (epoch + 1)))

    if epoch % 500 == 0 or epoch + 1 == args.epoch:
        torch.save(generator.state_dict(), os.path.join(args.weights_save, 'G_%d.pth' % (epoch)))
        torch.save(discriminator.state_dict(), os.path.join(args.weights_save, 'D_%d.pth' % (epoch)))