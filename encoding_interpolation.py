from dcgan_model import Encoder, Generator
import pickle
from os.path import join
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image


path = join('experiments_result', '2020_06_13__06_01_cluster5V4', 'artifacts')


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

opt = pickle.load( open( join(path, "opt.pkl"), "rb" ) )

transformation = transforms.Compose(
        [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )

encoder_path = join(path, 'E.pth')
generator_path = join(path, 'G.pth')

encoder = Encoder(opt)
generator = Generator(opt)

if cuda:
    encoder.cuda()
    generator.cuda()

encoder.load_state_dict(torch.load(encoder_path))
encoder.eval()

generator.load_state_dict(torch.load(generator_path))
generator.eval()

img1 = Image.open(join('data', 'cluster', '5', '55.png'))
img2 = Image.open(join('data', 'cluster', '5', '1957.png'))

img1_tensor = torch.unsqueeze(transformation(img1),0)
img2_tensor = torch.unsqueeze(transformation(img2),0)

real_imgs1 = Variable(img1_tensor.type(Tensor))
real_imgs2 = Variable(img2_tensor.type(Tensor))

z1 = encoder(real_imgs1)
z2 = encoder(real_imgs2)

nb_logo= 10
# Sample noise as generator input
z_to_interpolate = torch.cat((z1, z2), 0)
z_values = Variable(Tensor(nb_logo, opt.latent_dim).fill_(1.0), requires_grad=False)
for i in range(nb_logo):
    ratio = i/(nb_logo-1)
    z_values[i]= (1-ratio)*z_to_interpolate[0]+ratio*z_to_interpolate[1]
gen_imgs = generator(z_values.cuda())

save_image(gen_imgs.data[:10], join('experiments', 'interpolation.png'), nrow=5, normalize=True)


