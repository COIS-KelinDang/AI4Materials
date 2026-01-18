import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import functools
import random
import math


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class Patch_Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(Patch_Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


def DiffAugment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.2):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, ratio=0.5):
    if random.random() < 0.3:
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
    return x


def rand_rotate(x, ratio=0.5):
    k = random.randint(1, 3)
    if random.random() < ratio:
        x = torch.rot90(x, k, [2, 3])
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
    'rotate': [rand_rotate],
}


class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None,
                 dropout=0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.fc1 = nn.Linear(in_feat, hid_feat)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_feat, out_feat)
        self.droprateout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.droprateout(x)


class Attention(nn.Module):
    """
        输入尺寸为(b,n,c), b:batch_size, n:构成图片的patch数量, c:每个patch的表示向量维度
    """

    def __init__(self, dim, heads=4, attention_dropout=0., proj_dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = 1. / dim ** 0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_dropout)
        )

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        dot = (q @ k.transpose(-2, -1)) * self.scale  # (b, h, n, n), scale:放缩系数
        attn = dot.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.out(x)
        return x


class ImgPatches(nn.Module):
    def __init__(self, input_channel=3, dim=768, patch_size=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(input_channel, dim,
                                     kernel_size=patch_size, stride=patch_size)

    def forward(self, img):
        patches = self.patch_embed(img).flatten(2).transpose(1, 2)
        return patches


def UpSampling(x, H, W):
    """
        上采样模块，输入尺寸为(B,N,dim)，输出尺寸为(B,4*N,dim//4)
    """
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


class Encoder_Block(nn.Module):
    """
        输入尺寸为(b,n,dim), b:batch_size, n:构成图片的patch数量, c:每个patch的表示向量维度
        输出尺寸为(b,n,dim)
    """

    def __init__(self, dim, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, drop_rate, drop_rate)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio, dropout=drop_rate)

    def forward(self, x):
        x1 = self.ln1(x)
        x = x + self.attn(x1)
        x2 = self.ln2(x)
        x = x + self.mlp(x2)
        return x


class TransformerEncoder(nn.Module):
    """
        输入尺寸为(b,n,dim), b:batch_size, n:构成图片的patch数量, c:每个patch的表示向量维度
        输出尺寸为(b,n,mlp_ratio*dim)
    """

    def __init__(self, depth, dim, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.Encoder_Blocks = nn.ModuleList([
            Encoder_Block(dim, heads, mlp_ratio, drop_rate)
            for i in range(depth)])

    def forward(self, x):
        for Encoder_Block in self.Encoder_Blocks:
            x = Encoder_Block(x)
        return x


class Trans_Generator(nn.Module):
    """docstring for Generator"""

    def __init__(self, depth1=5, depth2=4, depth3=2, initial_size=8, dim=384, heads=4, mlp_ratio=4, drop_rate=0.):
        super(Trans_Generator, self).__init__()
        self.initial_size = initial_size
        self.dim = dim
        self.depth1 = depth1
        self.depth2 = depth2
        self.depth3 = depth3
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate = drop_rate
        self.mlp = nn.Linear(1024, (self.initial_size ** 2) * self.dim)
        self.positional_embedding_1 = nn.Parameter(torch.zeros(1, (8 ** 2), dim))
        self.positional_embedding_2 = nn.Parameter(torch.zeros(1, (8 * 2) ** 2, dim // 4))
        self.positional_embedding_3 = nn.Parameter(torch.zeros(1, (8 * 4) ** 2, dim // 16))
        self.TransformerEncoder_encoder1 = TransformerEncoder(depth=self.depth1, dim=self.dim, heads=self.heads,
                                                              mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder2 = TransformerEncoder(depth=self.depth2, dim=self.dim // 4, heads=self.heads,
                                                              mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder3 = TransformerEncoder(depth=self.depth3, dim=self.dim // 16, heads=self.heads,
                                                              mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.linear = nn.Sequential(nn.Conv2d(self.dim // 16, 3, 1, 1, 0))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, noise):
        x = self.mlp(noise).view(-1, self.initial_size ** 2, self.dim)
        x = x + self.positional_embedding_1
        H, W = self.initial_size, self.initial_size
        x = self.TransformerEncoder_encoder1(x)  # (B,N,C)
        x, H, W = UpSampling(x, H, W)  # (B,4*N,C//4)
        x = x + self.positional_embedding_2
        x = self.TransformerEncoder_encoder2(x)  # (B,4*N,C//4)
        x, H, W = UpSampling(x, H, W)  # (B,16*N,C//16)
        x = x + self.positional_embedding_3
        x = self.TransformerEncoder_encoder3(x)  # (B,16*N,C//16)
        x = self.linear(x.permute(0, 2, 1).view(-1, self.dim // 16, H, W))  # (B,16*N,C//16)
        return x


class Conv_Trans_Generator(nn.Module):
    """docstring for Generator"""

    def __init__(self, depth1=5, depth2=4, depth3=2, initial_size=8, dim=384, heads=4, mlp_ratio=4,
                 drop_rate=0.):  # ,device=device):
        super(Conv_Trans_Generator, self).__init__()
        # self.device = device
        self.initial_size = initial_size
        self.dim = dim
        self.depth1 = depth1
        self.depth2 = depth2
        self.depth3 = depth3
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate = drop_rate
        self.mlp = nn.Linear(128, (self.initial_size ** 2) * dim)
        self.positional_embedding_1 = nn.Parameter(torch.zeros(1, (8 ** 2), dim))
        self.positional_embedding_2 = nn.Parameter(torch.zeros(1, (8 * 2) ** 2, dim // 4))
        self.positional_embedding_3 = nn.Parameter(torch.zeros(1, (8 * 4) ** 2, dim // 16))
        self.TransformerEncoder_encoder1 = TransformerEncoder(depth=self.depth1, dim=self.dim, heads=self.heads,
                                                              mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder2 = TransformerEncoder(depth=self.depth2, dim=self.dim // 4, heads=self.heads,
                                                              mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder3 = TransformerEncoder(depth=self.depth3, dim=self.dim // 16, heads=self.heads,
                                                              mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.linear = nn.Sequential(nn.Conv2d(self.dim // 16, 3, 1, 1, 0))

        self.transconv1 = nn.Sequential(nn.ConvTranspose2d(self.dim // 16, 64, 4, 2, 1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True))
        self.transconv2 = nn.Sequential(nn.ConvTranspose2d(64, 128, 4, 2, 1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(True))
        self.transconv3 = nn.Sequential(nn.ConvTranspose2d(128, 256, 4, 2, 1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(True))
        # self.transconv4 = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, 2, 1),
        #                                 nn.BatchNorm2d(256),
        #                                 nn.ReLU(True))
        self.out_conv = nn.Conv2d(256, 3, 1, 1, 0)
        # self.apply(self._init_weights)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.trunc_normal_(m.weight, std=.02)
        #         if isinstance(m, nn.Linear) and m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.LayerNorm):
        #         nn.init.constant_(m.bias, 0)
        #         nn.init.constant_(m.weight, 1.0)

    def forward(self, noise):
        x = self.mlp(noise).view(-1, self.initial_size ** 2, self.dim)
        x = x + self.positional_embedding_1
        H, W = self.initial_size, self.initial_size
        x = self.TransformerEncoder_encoder1(x)  # (B,N,C)
        x, H, W = UpSampling(x, H, W)  # (B,4*N,C//4)
        x = x + self.positional_embedding_2
        x = self.TransformerEncoder_encoder2(x)  # (B,4*N,C//4)
        x, H, W = UpSampling(x, H, W)  # (B,16*N,C//16)
        x = x + self.positional_embedding_3
        x = self.TransformerEncoder_encoder3(x)  # (B,16*N,C//16)
        x = x.permute(0, 2, 1).view(-1, self.dim // 16, H, W)
        x = self.transconv1(x)
        x = self.transconv2(x)
        x = self.transconv3(x)
        # x = self.transconv4(x)
        x = self.out_conv(x)
        return x


class Patch_Trans_Discriminator(nn.Module):
    def __init__(self, diff_aug, image_size=32, patch_size=4, input_channel=3, num_classes=1,
                 dim=384, depth=7, heads=4, mlp_ratio=4, drop_rate=0.):
        super(Patch_Trans_Discriminator, self).__init__()
        if image_size % patch_size != 0:
            raise ValueError('Image size must be divisible by patch size.')
        num_patches = (image_size // patch_size) ** 2
        self.image_size = image_size
        self.dim = dim
        self.diff_aug = diff_aug
        self.patch_size = patch_size
        self.depth = depth
        # Image patches and embedding layer
        self.patches = ImgPatches(input_channel, dim, self.patch_size)

        # Embedding for patch position and class
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.adv_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.positional_embedding, std=0.2)

        self.droprate = nn.Dropout(p=drop_rate)
        self.TransfomerEncoder = TransformerEncoder(depth, dim, heads,
                                                    mlp_ratio, drop_rate)
        self.norm = nn.LayerNorm(dim)
        self.adv = nn.Linear(dim, 1)

        self.conv_block = nn.Sequential(nn.Conv2d(dim, 1, kernel_size=4, stride=2, padding=1, bias=False))

    def forward(self, x):
        x = DiffAugment(x, self.diff_aug)  # (B,C,H,W)
        b = x.shape[0]
        adv_token = self.adv_embedding.expand(b, -1, -1)
        x = self.patches(x)  # (B,N,C)
        x = torch.cat((adv_token, x), dim=1)  # (B,N+1,C)
        x += self.positional_embedding
        x = self.droprate(x)
        x = self.TransfomerEncoder(x)  # (B,N+1,C)
        x = self.norm(x)
        out_adv = self.adv(x[:, 0])

        B, N, C = x.shape
        N_conv = int(math.sqrt(N - 1))
        x_conv = x[:, 1:, :].permute(0, 2, 1).view(-1, self.dim, N_conv, N_conv)
        x_conv = self.conv_block(x_conv)
        return out_adv, x_conv


class Trans_Discriminator(nn.Module):
    def __init__(self, diff_aug, image_size=32, patch_size=4, input_channel=3, num_classes=1,
                 dim=384, depth=7, heads=4, mlp_ratio=4, drop_rate=0.):
        super(Trans_Discriminator, self).__init__()

        if image_size % patch_size != 0:
            raise ValueError('Image size must be divisible by patch size.')
        num_patches = (image_size // patch_size) ** 2
        self.diff_aug = diff_aug
        self.patch_size = patch_size
        self.depth = depth
        # Image patches and embedding layer
        self.patches = ImgPatches(input_channel, dim, self.patch_size)

        # Embedding for patch position and class
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.adv_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        self.aux_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.positional_embedding, std=0.2)

        self.droprate = nn.Dropout(p=drop_rate)
        self.TransfomerEncoder = TransformerEncoder(depth, dim, heads,
                                                    mlp_ratio, drop_rate)
        self.norm = nn.LayerNorm(dim)
        self.adv = nn.Linear(dim, 1)
        self.aux = nn.Linear(dim, num_classes)

    def forward(self, input):
        x = DiffAugment(input, self.diff_aug)  # (B,C,H,W)
        b = x.shape[0]
        # cls_token = self.class_embedding.expand(b, -1, -1)
        # aux_token = self.class_embedding.expand(b, -1, -1)
        x = self.patches(x)  # (B,N,C)
        # x = torch.cat((cls_token, aux_token, x), dim=1)   # (B,N+2,C)
        x += self.positional_embedding
        x = self.droprate(x)
        x = self.TransfomerEncoder(x)  # (B,N+2,C)
        x = self.norm(x)
        out_adv = self.adv(x[:, 0])
        out_aux = self.aux(x[:, 1])
        return out_adv, out_aux