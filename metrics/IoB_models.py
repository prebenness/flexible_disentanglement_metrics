from torch import nn
import torch
import numpy as np
import torch.nn.init as init

from metrics.utils import is_power_of_2

#Content
#Create content trainer
class TensorReconstructor(nn.Module):
    def __init__(self):
        super(TensorReconstructor, self).__init__()
        lr = 0.0001
        self.AE = AutoEncoder()
        self.print_network(self.AE, 'AutoEncoder')
        #Setup the optimizers
        beta1 = 0.5
        beta2 = 0.999
        AE_params = list(self.AE.parameters())
        self.AE_opt = torch.optim.Adam([p for p in AE_params if p.requires_grad], lr=lr, betas=(beta1, beta2))
        self.AE.apply(self.initialize_weights)

    def l2_criterion(self, inp, target):
        return torch.mean(torch.abs(inp - target) ** 2)

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
            init.xavier_uniform_(m.weight.data)

    def forward(self, train_input):
        self.eval()
        reconstruction = self.AE(train_input)
        self.train()
        return reconstruction

    def test(self, train_input, ori_images):
        self.eval()
        reconstruction = self.AE(train_input)
        mse = self.l2_criterion(reconstruction, ori_images)
        self.train()
        return mse

    def AE_update(self, train_input, ori_images):
        self.AE.zero_grad()
        reconstruction = self.AE(train_input)
        self.loss = self.l2_criterion(reconstruction, ori_images)
        self.loss.backward()
        self.AE_opt.step()
        return self.loss

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.model = []
        # 128x64x64 -> 64x32x32 -> 64x16x16
        # 32x32x32 -> 16x64x64 -> 8x128x128 -> 3x128x128
        dim = 128
        input_dim = 128
        output_dim = 3
        #Input layer
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3)]

        #Downsampling blocks
        for i in range(2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1)]

        #Upsampling blocks
        for i in range(3):
            self.model += [UpConv2dBlock(dim, dim // 2, 4, 2, 1)]
            dim //= 2

        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, relu=False)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, relu=True):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True

        #Initialize padding
        self.pad = nn.ZeroPad2d(padding)

        #Initialize normalization
        norm_dim = output_dim
        self.norm = nn.InstanceNorm2d(norm_dim, affine=True)

        #Initialize activation
        if relu:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.Tanh()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        x = self.norm(x)
        x = self.activation(x)
        return x

class UpConv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, relu=True):
        super(UpConv2dBlock, self).__init__()
        self.use_bias = True

        #Initialize normalization
        norm_dim = output_dim
        self.norm = nn.InstanceNorm2d(norm_dim, affine=True)

        #Initialize activation
        if relu:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.Tanh()
        self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


#Style
#Create style trainer
class VectorReconstructor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VectorReconstructor, self).__init__()

        # Pad targets up to next power of 2 and set this as new output dim
        new_output_dim = 2 ** np.ceil(np.log2(output_dim))
        new_output_dim = int(new_output_dim)

        self.output_dim = new_output_dim
        target_padding = new_output_dim - output_dim
        ## left, right, top, bottom
        pad = (target_padding, 0, target_padding, 0)
        self.target_padder = nn.ZeroPad2d(pad)

        lr = 0.0001
        self.DE = Decoder(input_dim=input_dim, output_dim=output_dim)
        self.print_network(self.DE, 'Decoder')
        #Setup the optimizers
        beta1 = 0.5
        beta2 = 0.999
        DE_params = list(self.DE.parameters())
        self.DE_opt = torch.optim.Adam([p for p in DE_params if p.requires_grad], lr=lr, betas=(beta1, beta2))
        self.DE.apply(self.initialize_weights)

    def l2_criterion(self, inp, target):
        target = self.target_padder(target)
        return torch.mean(torch.abs(inp - target) ** 2)

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
            init.xavier_uniform_(m.weight.data)

    def forward(self, train_input):
        self.eval()
        reconstruction = self.DE(train_input)
        self.train()
        return reconstruction

    def test(self, train_input, ori_images):
        self.eval()
        reconstruction = self.DE(train_input)
        mse = self.l2_criterion(reconstruction, ori_images)
        self.train()
        return mse

    def DE_update(self, train_input, ori_images):
        self.DE.zero_grad()
        reconstruction = self.DE(train_input)
        self.loss = self.l2_criterion(reconstruction, ori_images)
        self.loss.backward()
        self.DE_opt.step()
        return self.loss

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()
        self.model = []
        self.liner = []
        self.latent_dim = 64
        self.input_dim = input_dim
        output_channels = 3
        #Linear layers
        self.liner = nn.Sequential(
            nn.Linear(input_dim, self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim * 4 * 4),
            nn.Linear(self.latent_dim * 4 * 4, self.latent_dim * 8 * 8),
        )   # -> 128 * 128 image

        # Upsampling blocks
        internal_dim = self.latent_dim
        i = 1
        while True:      
            self.model += [UpConv2dBlock(internal_dim, internal_dim // 2, 4, 2, 1)]
            internal_dim = self.latent_dim // 2 ** i

            # Upsample until desired output_dim is reached
            if 8 * 2 ** i >= output_dim:
                break

            i += 1

        self.model += [Conv2dBlock(internal_dim, output_channels, 7, 1, 3, relu=False)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        liner_feature = self.liner(x)
        liner_feature = liner_feature.view(liner_feature.size(0), self.latent_dim, 8, 8)
        return self.model(liner_feature)
