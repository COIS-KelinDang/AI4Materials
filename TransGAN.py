from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from copy import deepcopy
import os
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
from network import Trans_Discriminator, Conv_Trans_Generator


parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=512, help='Size of image for discriminator input.')
parser.add_argument('--initial_size', type=int, default=8, help='Initial size for generator.')
parser.add_argument('--patch_size', type=int, default=32, help='Patch size for generated image.')
parser.add_argument('--num_classes', type=int, default=1, help='Number of classes for discriminator.')
parser.add_argument('--aux_classes', type=int, default=10, help='Number of classes for discriminator.')
parser.add_argument('--lr_gen', type=float, default=0.01, help='Learning rate for generator.')
parser.add_argument('--lr_dis', type=float, default=0.01, help='Learning rate for discriminator.')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay.')
parser.add_argument('--latent_dim', type=int, default=1024, help='Latent dimension.')
parser.add_argument('--n_critic', type=int, default=5, help='n_critic.')
parser.add_argument('--max_iter', type=int, default=500000, help='max_iter.')
parser.add_argument('--num_head', type=int, default=4, help='the heads num of atention.')
parser.add_argument('--mlp_ratio', type=int, default=4, help='hidden layer ratio params.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for generator.')
parser.add_argument('--epoch', type=int, default=2000, help='Number of epoch.')
parser.add_argument('--output_dir', type=str, default='checkpoint', help='Checkpoint.')
parser.add_argument('--dim', type=int, default=128, help='Embedding dimension.')
parser.add_argument('--img_name', type=str, default="img_name", help='Name of pictures file.')
parser.add_argument('--optim', type=str, default="SGD", help='Choose your optimizer')
parser.add_argument('--loss', type=str, default="wgangp_eps", help='Loss function')
parser.add_argument('--phi', type=int, default="1", help='phi')
parser.add_argument('--beta1', type=int, default="0", help='beta1')
parser.add_argument('--beta2', type=float, default="0.99", help='beta2')
parser.add_argument('--lr_decay', type=str, default=True, help='lr_decay')
parser.add_argument('--diff_aug', type=str, default="translation,cutout,color", help='Data Augmentation')

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)
print("Device:", device)

args = parser.parse_args()

generator = Conv_Trans_Generator(depth1=1, depth2=1, depth3=2, initial_size=args.initial_size, latent_dim=args.latent_dim,
                                 dim=args.dim, heads=args.num_head, mlp_ratio=args.mlp_ratio, drop_rate=0.5)
generator.to(device)

discriminator = Trans_Discriminator(diff_aug=args.diff_aug, image_size=args.image_size, patch_size=args.patch_size,
                                    input_channel=3, num_classes=1,dim=args.dim, depth=7, heads=args.num_head, mlp_ratio=args.mlp_ratio, drop_rate=0.)
discriminator.to(device)

# --------------创建训练日志---------------
path_log = '.\log'
t = time.localtime()
time_str = 'TransGAN' + str(t.tm_mon) + '_' + str(t.tm_mday) + '_' + str(t.tm_hour) + '_' + str(t.tm_min)
path_train_log = os.path.join(path_log,time_str)
os.makedirs(path_train_log)
path_train_details_file = os.path.join(path_train_log, '_train_details.txt')
with open(path_train_details_file, 'a') as f:
    f.write('训练细节记录:\n')
path_train_eval_imgs = os.path.join(path_train_log,'eval_img')
os.makedirs(path_train_eval_imgs)
path_train_parmas = os.path.join(path_train_log, 'params')
os.makedirs(path_train_parmas)



if args.optim == 'Adam':
    optim_gen = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=args.lr_gen,
                           betas=(args.beta1, args.beta2))

    optim_dis = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr_dis,
                           betas=(args.beta1, args.beta2))

elif args.optim == 'SGD':
    optim_gen = optim.SGD(filter(lambda p: p.requires_grad, generator.parameters()),
                          lr=args.lr_gen, momentum=0.9)

    optim_dis = optim.SGD(filter(lambda p: p.requires_grad, discriminator.parameters()),
                          lr=args.lr_dis, momentum=0.9)

elif args.optim == 'RMSprop':
    optim_gen = optim.RMSprop(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr_dis, eps=1e-08,
                              weight_decay=args.weight_decay, momentum=0, centered=False)

    optim_dis = optim.RMSprop(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr_dis, eps=1e-08,
                              weight_decay=args.weight_decay, momentum=0, centered=False)

else:
    raise ValueError('未支持优化器类型!')



def compute_gradient_penalty(D, real_samples, fake_samples, phi):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(real_samples.get_device())
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(real_samples.get_device())
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty

gen_scheduler = torch.optim.lr_scheduler.StepLR(optim_gen, 5, gamma=0.98)
dis_scheduler = torch.optim.lr_scheduler.StepLR(optim_dis, 5, gamma=0.98)

# 训练TransGAN
transforms_train = transforms.Compose([transforms.Resize(size=(args.image_size, args.image_size)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transforms_test = transforms.Compose([transforms.Resize(size=(args.image_size, args.image_size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transforms_train)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transforms_test)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)
real_label = 1
fake_label = 0
dis_criterion = nn.BCELoss()
aux_criterion = nn.NLLLoss()

for epoch in range(args.epoch):
    generator = generator.train()
    discriminator = discriminator.train()
    for index, (img, labels) in enumerate(train_loader):
        # train D
        # train with real
        real_imgs = img.type(torch.cuda.FloatTensor)
        noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (img.shape[0], args.latent_dim)))
        label = np.random.randint(0, args.aux_classes, img.shape[0])
        class_onehot = np.zeros((img.shape[0], args.aux_classes))
        class_onehot[np.arange(img.shape[0]), label] = 1
        noise[np.arange(img.shape[0]), :args.aux_classes] = class_onehot[np.arange(img.shape[0])]
        optim_dis.zero_grad()
        real_valid, real_aux = discriminator(real_imgs)
        fake_imgs = generator(noise).detach()
        fake_valid, fake_aux = discriminator(fake_imgs)
        real_adv_label = torch.FloatTensor(img.shape[0]).to(device)
        real_adv_label.data.fill_(real_label)
        real_aux_label = torch.LongTensor(img.shape[0]).to(device)
        real_aux_label.data.copy_(labels)
        fake_adv_label = torch.FloatTensor(img.shape[0]).to(device)
        fake_adv_label.data.fill_(fake_label)
        fake_aux_label = torch.LongTensor(img.shape[0]).to(device)
        fake_aux_label.data.copy_(torch.from_numpy(label))
        dis_errD_real = dis_criterion(real_valid, real_adv_label)
        aux_errD_real = aux_criterion(real_aux, real_aux_label)
        dis_errD_fake = dis_criterion(fake_valid, dis_label)
        aux_errD_fake = aux_criterion(aux_output, aux_label)
        errD_real = dis_errD_real + aux_errD_real
        errD_real.backward()

        # train with fake
        gener_noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (img.shape[0], args.latent_dim)))
        generated_imgs = generator(gener_noise)
        label = np.random.randint(0, args.aux_classes, batch_size)
        noise_ = np.random.normal(0, 1, (batch_size, nz))
        class_onehot = np.zeros((batch_size, num_classes))
        class_onehot[np.arange(batch_size), label] = 1
        noise_[np.arange(batch_size), :num_classes] = class_onehot[np.arange(batch_size)]
        noise_ = (torch.from_numpy(noise_))
        noise.data.copy_(noise_.view(batch_size, nz, 1, 1))
        aux_label.data.resize_(batch_size).copy_(torch.from_numpy(label))

        fake = net_G(noise)
        dis_label.data.fill_(fake_label)
        dis_output, aux_output = net_D(fake.detach())
        dis_errD_fake = dis_criterion(dis_output, dis_label)
        aux_errD_fake = aux_criterion(aux_output, aux_label)
        errD_fake = dis_errD_fake + aux_errD_fake
        errD_fake.backward()
        D_G_z1 = dis_output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()
        schedulerD.step()


        if args.loss == 'hinge':
            loss_dis = torch.mean(nn.ReLU(inplace=True)(1.0 - real_valid)).to(device) + torch.mean(
                nn.ReLU(inplace=True)(1 + fake_valid)).to(device)
        elif args.loss == 'wgangp_eps':
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, fake_imgs.detach(), args.phi)
            loss_dis = -torch.mean(real_valid) + torch.mean(fake_valid) + gradient_penalty * 10 / (args.phi ** 2)


        loss_dis.backward()
        optim_dis.step()

        # if global_steps % n_critic == 0:
        optim_gen.zero_grad()
        gener_noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim)))
        generated_imgs = generator(gener_noise)
        fake_valid = discriminator(generated_imgs)
        gener_loss = -torch.mean(fake_valid).to(device)
        gener_loss.backward()
        optim_gen.step()

        print("[Epoch %d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
              (epoch + 1, index % len(train_loader), len(train_loader), loss_dis.item(), gener_loss.item()))
    if epoch % 10 == 0:
        # 记录测试图像
        generator.eval()
        torchvision.utils.save_image(
            real_imgs[:10].data, os.path.join(path_train_eval_imgs, '{}_real_samples.png'.format(epoch)))
        with torch.no_grad():
            gener_noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (10, args.latent_dim)))
            fake_imgs = generator(gener_noise)
        torchvision.utils.save_image(
            fake_imgs[:10].data, os.path.join(path_train_eval_imgs, '{}_fake_samples.png'.format(epoch)))
        torch.save(generator.state_dict(), os.path.join(path_train_parmas, 'generator.pth'))
        torch.save(discriminator.state_dict(), os.path.join(path_train_parmas, 'discriminator.pth'))
    #     torch.save('')
    #
    # if epoch % 20 == 0:
    #     sample_imgs = generated_imgs[:25]
    #     img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)
    #     save_image(sample_imgs, os.path.join(path_train_eval_imgs,'generated_img_{}.jpg'.format(epoch)), nrow=5,
    #                normalize=True, scale_each=True)



#
# def train(noise, generator, discriminator, optim_gen, optim_dis,
#           epoch, schedulers, img_size=32, latent_dim=args.latent_dim,
#           n_critic=args.n_critic,
#           gener_batch_size=args.gener_batch_size, device="cuda:0"):
#
#     gen_step = 0
#     generator = generator.train()
#     discriminator = discriminator.train()
#     transform = transforms.Compose(
#         [transforms.Resize(size=(img_size, img_size)), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
#          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
#     train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=30, shuffle=True)
#
#     for index, (img, _) in enumerate(train_loader):
#         real_imgs = img.type(torch.cuda.FloatTensor)
#         noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (img.shape[0], latent_dim)))
#
#         optim_dis.zero_grad()
#         real_valid = discriminator(real_imgs)
#         fake_imgs = generator(noise).detach()
#
#         fake_valid = discriminator(fake_imgs)
#
#         if args.loss == 'hinge':
#             loss_dis = torch.mean(nn.ReLU(inplace=True)(1.0 - real_valid)).to(device) + torch.mean(
#                 nn.ReLU(inplace=True)(1 + fake_valid)).to(device)
#         elif args.loss == 'wgangp_eps':
#             gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, fake_imgs.detach(), args.phi)
#             loss_dis = -torch.mean(real_valid) + torch.mean(fake_valid) + gradient_penalty * 10 / (args.phi ** 2)
#
#         loss_dis.backward()
#         optim_dis.step()
#
#         if global_steps % n_critic == 0:
#
#             optim_gen.zero_grad()
#             if schedulers:
#                 gen_scheduler, dis_scheduler = schedulers
#                 g_lr = gen_scheduler.step(global_steps)
#                 d_lr = dis_scheduler.step(global_steps)
#
#             gener_noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (gener_batch_size, latent_dim)))
#
#             generated_imgs = generator(gener_noise)
#             fake_valid = discriminator(generated_imgs)
#
#             gener_loss = -torch.mean(fake_valid).to(device)
#             gener_loss.backward()
#             optim_gen.step()
#
#             gen_step += 1
#
#         if gen_step and index % 100 == 0:
#             sample_imgs = generated_imgs[:25]
#             img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)
#             save_image(sample_imgs, f'generated_images/generated_img_{epoch}_{index % len(train_loader)}.jpg', nrow=5,
#                        normalize=True, scale_each=True)
#             tqdm.write("[Epoch %d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
#                        (epoch + 1, index % len(train_loader), len(train_loader), loss_dis.item(), gener_loss.item()))
#
#
# def validate(generator, writer_dict, fid_stat):
#     writer = writer_dict['writer']
#     global_steps = writer_dict['valid_global_steps']
#
#     generator = generator.eval()
#     fid_score = get_fid(fid_stat, epoch, generator, num_img=5000, val_batch_size=60 * 2, latent_dim=1024,
#                         writer_dict=None, cls_idx=None)
#
#     print(f"FID score: {fid_score}")
#
#     writer.add_scalar('FID_score', fid_score, global_steps)
#
#     writer_dict['valid_global_steps'] = global_steps + 1
#     return fid_score
#
#
# best = 1e4
#
# for epoch in range(args.epoch):
#
#     lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
#
#     train(noise, generator, discriminator, optim_gen, optim_dis,
#           epoch, writer, lr_schedulers, img_size=32, latent_dim=args.latent_dim,
#           n_critic=args.n_critic,
#           gener_batch_size=args.gener_batch_size)
#
#     checkpoint = {'epoch': epoch, 'best_fid': best}
#     checkpoint['generator_state_dict'] = generator.state_dict()
#     checkpoint['discriminator_state_dict'] = discriminator.state_dict()
#
#     score = validate(generator, writer_dict, fid_stat)
#
#     print(f'FID score: {score} - best ID score: {best} || @ epoch {epoch + 1}.')
#     if epoch == 0 or epoch > 30:
#         if score < best:
#             save_checkpoint(checkpoint, is_best=(score < best), output_dir=args.output_dir)
#             print("Saved Latest Model!")
#             best = score
#
# checkpoint = {'epoch': epoch, 'best_fid': best}
# checkpoint['generator_state_dict'] = generator.state_dict()
# checkpoint['discriminator_state_dict'] = discriminator.state_dict()
# score = validate(generator, writer_dict, fid_stat)  ####CHECK AGAIN
# save_checkpoint(checkpoint, is_best=(score < best), output_dir=args.output_dir)