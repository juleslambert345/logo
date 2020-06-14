# code from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py
import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from dcgan_model import weights_init_normal, Generator, Discriminator, GeneratorWrapper
from dcgan_dataset import SimpleDataset, LabelDataset
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import mlflow
import mlflow.pytorch
from datetime import datetime
from train_encoder import train_encoder
from PIL import ImageFilter


from PIL import Image
from os.path import join
from torch.utils.data import Dataset


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training the gan")
parser.add_argument("--n_epochs_encoder", type=int, default=200, help="number of epochs of training the encoder")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--experiment_name", type=str, default='cluster5V6', help="name of folder where we save the experiment")
parser.add_argument("--generator_scale", type=int, default=16, help="scale of which we multiply the number of channel in the generator")
parser.add_argument("--discriminator_scale", type=int, default=16, help="scale of which we multiply the number of channel in the discriminator")
parser.add_argument("--encoder_scale", type=int, default=16, help="scale of which we multiply the number of channel in the discriminator")
parser.add_argument("--cluster", type=int, default=5, help="scale of which we multiply the number of channel in the discriminator")



opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
experiment_path = join('experiments', opt.experiment_name)
os.makedirs(join(experiment_path, 'model'), exist_ok=True)
os.makedirs(join(experiment_path, 'images'), exist_ok=True)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1/255):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        new_tensor = tensor + torch.randn(tensor.size()) * self.std + self.mean
        new_tensor = torch.clamp(new_tensor, 0, 1)
        return new_tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class GaussianBlur(object):
    def __init__(self, radius =2, p=0.1):
        self.radius = radius
        self.p = p

    def __call__(self, img):
        if np.random.rand()<self.p:
            return img.filter(ImageFilter.GaussianBlur(self.radius))
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(radius={0}, , p={1})'.format(self.radius, self.p)


def save_model(discriminator, generator, model_path, batches_done):
    '''
    save model
    '''
    torch.save(discriminator.state_dict(), join(model_path, 'D%d.pth' % batches_done))
    torch.save(generator.state_dict(), join(model_path, 'G%d.pth' % batches_done))


def get_model(opt, cuda):
    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(opt)
    discriminator = Discriminator(opt)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    return generator, discriminator, adversarial_loss


def save_loss_plot(batches_list, generative_loss_list, discriminative_loss_list, experiment_path):
    plt.scatter(batches_list, generative_loss_list, label="generative loss")
    plt.scatter(batches_list, discriminative_loss_list, label="discriminative loss")
    plt.legend(loc="upper left")
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.savefig(join(experiment_path, 'loss.png'))
    plt.close()

def save_model_info(discriminator, generator, experiment_path, gen_imgs, batches_done, batches_list, generative_loss_list, discriminative_loss_list):
    save_image(gen_imgs.data[:25], join(experiment_path, 'images', "%d.png" % batches_done), nrow=5, normalize=True)
    save_model(discriminator, generator, join(experiment_path, 'model'), batches_done)
    loss_df = pd.DataFrame({'batches done': batches_list,
                            'generative loss': generative_loss_list,
                            'discriminative loss': discriminative_loss_list})
    loss_df.to_csv(join(experiment_path, 'loss.csv'))
    save_loss_plot(batches_list, generative_loss_list, discriminative_loss_list, experiment_path)
    save_interpolation(25, generator, join(experiment_path, 'images'), "%d_interpolation.png" % batches_done)


def train_generator(imgs,  generator, discriminator, adversarial_loss, optimizer_G, opt):

    valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)

    optimizer_G.zero_grad()

    # Sample noise as generator input
    z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

    # Generate a batch of images
    gen_imgs = generator(z)

    # Loss measures generator's ability to fool the discriminator
    g_loss = adversarial_loss(discriminator(gen_imgs), valid)

    g_loss.backward()
    optimizer_G.step()

    return gen_imgs, g_loss


def train_discriminator(real_imgs, gen_imgs, discriminator, adversarial_loss, optimizer_D, g_loss):
    valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

    optimizer_D.zero_grad()

    # Measure discriminator's ability to classify real from generated samples
    real_loss = adversarial_loss(discriminator(real_imgs), valid)
    fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
    d_loss = (real_loss + fake_loss) / 2


    d_loss.backward()
    optimizer_D.step()
    return d_loss

def create_interpolation(nb_logo, generator):
    # Sample noise as generator input
    z_to_interpolate = Variable(Tensor(np.random.normal(0, 1, (2, opt.latent_dim))))
    z_values = Variable(Tensor(nb_logo, opt.latent_dim).fill_(1.0), requires_grad=False)
    for i in range(nb_logo):
        ratio = i/(nb_logo-1)
        z_values[i]= (1-ratio)*z_to_interpolate[0]+ratio*z_to_interpolate[1]
    gen_imgs = generator(z_values)
    return gen_imgs


def save_interpolation(nb_logo, generator, path, name_file = None):
    if name_file is None:
        name_file = "interpolation_logo.png"
    generated_images = create_interpolation(nb_logo, generator)
    save_image(generated_images.data, join(path, name_file), nrow=5, normalize=True)


def calculate_encoding(dataloader, encoder, path_experiment):

    list_encoding = []
    list_filename = []
    encoder.eval()

    for i, data in enumerate(dataloader):
        imgs, filename = data
        real_imgs = Variable(imgs.type(Tensor))
        list_encoding.append(encoder(real_imgs).data.cpu().numpy())
        list_filename+=list(filename)

    encoding = np.concatenate( list_encoding, axis=0 )


    df = pd.DataFrame(encoding, index = list_filename)
    df.to_csv(join(path_experiment, 'encoding_training_data.csv'))


def save_ml_flow(path_save_experiment, name_experiement, path_experiement_files):
    '''

    :param path_save_experiment: where we will put the folder of packaging
    :param name_experiement: name of the experiment
    :param path_experiement_files: where the files of teh experiement were saving during experiement
    '''

    now = datetime.now()
    time_stamp = now.strftime("%Y_%m_%d__%H_%M_")

    save_folder = join(path_save_experiment, time_stamp+name_experiement)


    cwd = os.getcwd()

    artifacts = {
        "generator_dict": 'file//'+join(cwd, path_experiement_files, 'G.pth').replace('\\', '/'), #mlflow seem to have some difficulty with path in windows this correction make it work
        "discriminator_dict": 'file//' + join(cwd, path_experiement_files, 'D.pth').replace('\\', '/'),
        "encoder_dict": 'file//' + join(cwd, path_experiement_files, 'E.pth').replace('\\', '/'),
        "opt": 'file//'+join(cwd, path_experiement_files, 'opt.pkl').replace('\\', '/'),
        "loss_image": 'file//' + join(cwd, path_experiement_files, 'loss.png').replace('\\', '/'),
        "loss_image_encoder": 'file//' + join(cwd, path_experiement_files, 'loss_encoder.png').replace('\\', '/'),
        "loss_csv": 'file//' + join(cwd, path_experiement_files, 'loss.csv').replace('\\', '/'),
        "generated_logo": 'file//' + join(cwd, path_experiement_files, 'generated_logo.png').replace('\\', '/'),
        "interpolation_logo": 'file//' + join(cwd, path_experiement_files, 'interpolation_logo.png').replace('\\', '/'),
        "encoding_logo": 'file//' + join(cwd, path_experiement_files, 'encoding_image.png').replace('\\', '/'),
        "encoding_data": 'file//' + join(cwd, path_experiement_files, 'encoding_training_data.csv').replace('\\', '/')
    }

    conda_env = mlflow.pytorch.get_default_conda_env()

    mlflow.pyfunc.save_model(path=save_folder,
                             python_model=GeneratorWrapper(),
                             artifacts=artifacts,
                             conda_env=conda_env,
                             code_path=['dcgan_model.py', 'meta_data.txt']
                             )


if __name__ == "__main__":

    generator, discriminator, adversarial_loss = get_model(opt, cuda)

    pickle.dump(opt, open(join(experiment_path, 'opt.pkl'), "wb"))

    # Configure data loader

    transformation = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomGrayscale(p=0.01),
                 GaussianBlur(2),
                 transforms.Resize(opt.img_size),
                 transforms.ToTensor(),
                 AddGaussianNoise(),
                 transforms.Normalize([0.5], [0.5])
                 ]
            )

    dataset_logo = SimpleDataset(join('data', 'cluster', str(opt.cluster)), transformation)

    dataloader = torch.utils.data.DataLoader(
        dataset_logo,
        batch_size=opt.batch_size,
        shuffle=True,
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr*10, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    generative_loss_list = []
    discriminative_loss_list = []
    batches_list = []

    for epoch in range(opt.n_epochs):
        for i, data in enumerate(dataloader):

            imgs, _ = data
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            gen_imgs, g_loss = train_generator(imgs, generator, discriminator, adversarial_loss, optimizer_G, opt)
            d_loss = train_discriminator(real_imgs, gen_imgs, discriminator, adversarial_loss, optimizer_D, g_loss)

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i

            generative_loss_list.append(g_loss.item())
            discriminative_loss_list.append(d_loss.item())
            batches_list.append(batches_done)

            if batches_done % opt.sample_interval == 0:
                save_model_info(discriminator, generator, experiment_path, gen_imgs, batches_done, batches_list,
                                generative_loss_list, discriminative_loss_list)

    torch.save(discriminator.state_dict(), join(experiment_path, 'D.pth'))
    torch.save(generator.state_dict(), join(experiment_path, 'G.pth'))
    save_image(gen_imgs.data[:25], join(experiment_path, "generated_logo.png"), nrow=5, normalize=True)
    save_interpolation(25, generator, experiment_path)

    encoder = train_encoder(opt, generator, experiment_path)
    calculate_encoding(dataloader, encoder, experiment_path)

    save_ml_flow('experiments_result', opt.experiment_name, experiment_path)
