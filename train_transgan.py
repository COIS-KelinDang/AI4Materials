import torch
from torch.utils.data import DataLoader
import utils
from TransGAN.ViT_custom import Generator, Discriminator
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

parser = argparse.ArgumentParser(description='Calculate volume of a cylinder')
parser.add_argument('--train_dataset', type=str, default=r'datasets')
parser.add_argument('--run_save', type=str, default=r'runs')
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--step_size', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--weight_save', type=bool, default=True)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

def main():
    train_set = utils.TrainGANDataset(args.train_dataset, args.img_size, train=True)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size)
    generator = Generator(latent_dim=args.latent_dim, norm_layer='ln', g_act='gelu')
    generator.to(args.device)
    discriminator = Discriminator(img_size=args.img_size, df_dim=512, norm_layer='ln', act_layer='gelu')
    discriminator.to(args.device)
    generator = torch.nn.DataParallel(generator)
    discriminator = torch.nn.DataParallel(discriminator)

    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), args.lr)
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), args.lr)

    gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_optimizer, args.step_size, gamma=0.97)
    dis_scheduler = torch.optim.lr_scheduler.StepLR(dis_optimizer, args.step_size, gamma=0.97)

    for epoch in range(args.epochs):
        train_bar = tqdm(train_loader)
        iter_idx = 0
        for imgs in train_bar:
            real_imgs = imgs.type(torch.cuda.FloatTensor).to(args.device)
            z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))).to(args.device)

            real_validity = discriminator(real_imgs)
            fake_imgs = generator(z).detach()
            assert fake_imgs.size() == real_imgs.size(), f"fake_imgs.size(): {fake_imgs.size()} real_imgs.size(): {real_imgs.size()}"
            fake_validity = discriminator(fake_imgs)

            # -----训练判别器-----
            real_label = torch.full((imgs.shape[0],), 1., dtype=torch.float, device=real_imgs.get_device())
            fake_label = torch.full((imgs.shape[0],), 0., dtype=torch.float, device=real_imgs.get_device())
            real_validity = nn.Sigmoid()(real_validity.view(-1))
            fake_validity = nn.Sigmoid()(fake_validity.view(-1))
            d_real_loss = nn.BCELoss()(real_validity, real_label)
            d_fake_loss = nn.BCELoss()(fake_validity, fake_label)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            dis_optimizer.step()
            dis_optimizer.zero_grad()
            iter_idx += 1

            # -----训练生成器-----
            gen_z = torch.FloatTensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim))).to(args.device)
            gen_imgs = generator(gen_z)
            fake_validity = discriminator(gen_imgs)
            real_label = torch.full((args.batch_size,), 1., dtype=torch.float, device=real_imgs.get_device())
            fake_validity = nn.Sigmoid()(fake_validity.view(-1))
            g_loss = nn.BCELoss()(fake_validity.view(-1), real_label)
            g_loss.backward()
            gen_optimizer.step()
            gen_optimizer.zero_grad()

        gen_scheduler.step()
        dis_scheduler.step()




if __name__ == '__main__':
    main()