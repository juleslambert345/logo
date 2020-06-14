# code from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py

import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import mlflow.pyfunc
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import io

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

def get_image(image_tensor):
    '''
    take an image as tensor and return the image as vytes object
    '''

    imggrd = vutils.make_grid(image_tensor, padding=2, normalize=True)
    imgnp = imggrd.numpy()
    imgnpT = np.transpose(imgnp, (1, 2, 0))

    imgPIL = Image.fromarray((255 * imgnpT).astype("uint8"), 'RGB')

    # create file-object in memory
    file_object = io.BytesIO()

    # write PNG in file-object
    imgPIL.save(file_object, 'PNG')

    file_object.seek(0)

    return file_object


class GeneratorWrapper(mlflow.pyfunc.PythonModel):

    # Load in the model and all required artifacts
    # The context object is provided by the MLflow framework
    # It will contain all of the artifacts specified above
    def load_context(self, context):
        import torch
        import pickle

        self.cuda = True if torch.cuda.is_available() else False

        # Load in and deserialize the label encoder object
        with open(context.artifacts["opt"].replace('\\', '/'), 'rb') as handle:
            self.opt = pickle.load(handle)

        # Initialize the model and load in the state dict
        self.generator = Generator(self.opt)
        self.encoder = Encoder(self.opt)
        self.generator.load_state_dict(torch.load(context.artifacts['generator_dict'].replace('\\', '/'),
                                                  map_location=lambda storage, loc: storage))

        self.encoder.load_state_dict(torch.load(context.artifacts['encoder_dict'].replace('\\', '/'),
                                                  map_location=lambda storage, loc: storage))

        self.generator.eval()
        self.encoder.eval()

        self.transformation = transforms.Compose(
            [transforms.Resize(self.opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

        if self.cuda:
            self.generator.cuda()
            self.encoder.cuda()

    def latent_vector_interpolation(self, z1, z2, nb_logos):
        z_to_interpolate = torch.cat((torch.unsqueeze(z1, 0), torch.unsqueeze(z2, 0)), 0)
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        z_values = Variable(Tensor(nb_logos, self.opt.latent_dim).fill_(1.0), requires_grad=False)
        for i in range(nb_logos):
            ratio = i / (nb_logos - 1)
            z_values[i] = (1 - ratio) * z_to_interpolate[0] + ratio * z_to_interpolate[1]
        return z_values



    # Create a predict function for our models
    def predict(self, context, data_dic):
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        if data_dic['type_prediction'] == 'random':
            #nb_logo should give the number of logo
            nb_logos = data_dic['nb_logos']

            z_value = Variable(Tensor(np.random.normal(0, 1, (nb_logos, self.opt.latent_dim))))
            logos = self.generator(z_value)
            logos = logos.cpu().detach()
            logos = get_image(logos)
            return logos
        if data_dic['type_prediction'] == 'from_vector':
            z = data_dic['z']
            Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
            z_value = Variable(Tensor(z))
            logos = self.generator(z_value)
            logos = logos.cpu().detach()
            logos = get_image(logos)
            return logos
        if data_dic['type_prediction'] == 'random_interpolation':
            nb_logos = data_dic['nb_logos']
            z_values = np.random.normal(0, 1, (2, 100)) # shape (2, 100)
            z_values = Variable(Tensor(z_values))
            z_values = self.latent_vector_interpolation(z_values[0], z_values[1], nb_logos)
            logos = self.generator(z_values.cuda())
            logos = logos.cpu().detach()
            logos = get_image(logos)
            return logos
        if data_dic['type_prediction'] == 'interpolation_from_vector':
            nb_logos = data_dic['nb_logos']
            z_values = data_dic['z'] # shape (2, 100)
            z_values = Variable(Tensor(z_values))
            z_values = self.latent_vector_interpolation(z_values[0], z_values[1], nb_logos)
            logos = self.generator(z_values.cuda())
            logos = logos.cpu().detach()
            logos = get_image(logos)
            return logos
        if data_dic['type_prediction'] == 'encoding':
            img = data_dic['img']
            img_tensor = torch.unsqueeze(self.transformation(img), 0)
            real_imgs = Variable(img_tensor.type(Tensor))
            z = self.encoder(real_imgs)
            return z.cpu().detach().numpy()
