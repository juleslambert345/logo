# code from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py

import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import mlflow.pyfunc


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.opt = opt # option of the model

        scale = opt.generator_scale
        self.scale = scale

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 32*scale * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(32*scale),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32*scale, 32*scale, 3, stride=1, padding=1),
            nn.BatchNorm2d(32*scale, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32*scale, 16*scale, 3, stride=1, padding=1),
            nn.BatchNorm2d(16*scale, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16*scale, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 32*self.scale, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.opt = opt # option of the model

        scale = opt.discriminator_scale

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 4*scale, bn=False),
            *discriminator_block(4*scale, 8*scale),
            *discriminator_block(8*scale, 16*scale),
            *discriminator_block(16*scale, 32*scale),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(32*scale * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()

        self.opt = opt # option of the model

        scale = opt.encoder_scale

        def encoder_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *encoder_block(opt.channels, 4*scale, bn=False),
            *encoder_block(4*scale, 8*scale),
            *encoder_block(8*scale, 16*scale),
            *encoder_block(16*scale, 32*scale),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Linear(32*scale * ds_size ** 2, opt.latent_dim)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


class GeneratorWrapper(mlflow.pyfunc.PythonModel):

    # Load in the model and all required artifacts
    # The context object is provided by the MLflow framework
    # It will contain all of the artifacts specified above
    def load_context(self, context):
        import torch
        import pickle

        self.cuda = True if torch.cuda.is_available() else False

        # Load in and deserialize the label encoder object
        with open(context.artifacts["opt"], 'rb') as handle:
            self.opt = pickle.load(handle)

        # Initialize the model and load in the state dict
        self.generator =  Generator(self.opt)
        self.generator.load_state_dict(torch.load(context.artifacts['generator_dict']))

        if self.cuda:
            self.generator.cuda()


    # Create a predict function for our models
    def predict(self, context, nb_logos):

        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        z = Variable(Tensor(np.random.normal(0, 1, (nb_logos, 100))))
        logos = self.generator(z)

        return logos

