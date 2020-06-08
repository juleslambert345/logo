'''
This is to create model for a basic version of conditional gan
Essentially copied the model from the dcgan_model.py file and transform into conditional
'''

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
        self.nb_label = opt.nb_label

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim+self.nb_label, 32*scale * self.init_size ** 2))#+100 is for the

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

    def forward(self, z, labels):
        out = torch.cat((z, labels), 1)
        out = self.l1(out)
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
            *discriminator_block(opt.channels+1, 4*scale, bn=False),# the +1 is for the label
            *discriminator_block(4*scale, 8*scale),
            *discriminator_block(8*scale, 16*scale),
            *discriminator_block(16*scale, 32*scale),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(32*scale * ds_size ** 2, 1), nn.Sigmoid())

        # label encoding
        self.nb_label = opt.nb_label
        self.label_encoding = nn.Sequential(nn.Linear(self.nb_label, 50), nn.LeakyReLU(0.2, inplace=True), nn.Linear(50, 32 *32), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, img, label):
        batch_size = img.shape[0]
        label_encoding = self.label_encoding(label)
        label_encoding = torch.reshape(label_encoding, (batch_size, 1, self.opt.img_size, self.opt.img_size))
        out = torch.cat((img, label_encoding), 1)
        out = self.model(out)
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
        self.adv_layer = nn.Linear(32*scale * ds_size ** 2, opt.latent_dim+opt.nb_label)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity[:,:self.opt.latent_dim], validity[:,self.opt.latent_dim:]


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
        #nb_logo should give the number of logo
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        z = Variable(Tensor(np.random.normal(0, 1, (64, 100))))
        logos = self.generator(z)

        return logos.cpu().detach()

