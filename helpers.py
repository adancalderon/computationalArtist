import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from torchvision.utils import save_image
import scipy.misc
import time
import random

loader = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.ToTensor()
])

unloader = transforms.ToPILImage()

def loadImage(name):
    image = Image.open(name)
    image = Variable(loader(image))
    image = image.unsqueeze(0)
    return image

def saveImage(input, path, height, width):
    image = input.data.clone().cpu()
    batch_size = image.size(0)
    image = image.view(batch_size, 3, height, width)
    image = unloader(image)
    scipy.misc.imsave(path, image)
    
def im_convert(tensor):
    ''' Presentar imagen como Tensor'''
    imagen = tensor.to('cpu').clone().detach()
    imagen = imagen.numpy().squeeze()
    imagen = imagen.transpose(1, 2, 0)
    #imagen = imagen * np.array((0.029, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    imagen = imagen.clip(0, 1)
    
    return imagen

def showImages(content, style, output):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))

    ax1.imshow(im_convert(content))
    ax1.axis('off')
    ax1.set_title('Original Picture')

    ax2.imshow(im_convert(style))
    ax2.axis('off')
    ax2.set_title('Style Image')

    ax3.imshow(im_convert(output) )
    ax3.axis('off')
    ax3.set_title('Merged Image')

    plt.show()

def save_images(input, paths, imsize):
    assert len(input) == len(paths), "Number of inputs and paths should be the same."
    N = input.size()[0]
    images = input.data.clone().cpu()
    for n in range(N):
        image = images[n]
        image = image.view(3, imsize, imsize)
        image = unloader(image)
        imageio.imwrite(paths[n], image)

class GramMatrix(nn.Module):

    def forward(self, input):
        a,b,c,d = input.size()
        features = input.view(a*b, c*d)
        G = torch.mm(features, features.t())
        return G.div(a*b*c*d)
    
class CNN(object):
    def __init__(self, style, content, pastiche, alpha, beta, lr=0.01):
        super(CNN, self).__init__()
        
        self.style = style
        self.content = content
        self.pastiche = nn.Parameter(pastiche.data)

        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.contentWeight = alpha
        self.styleWeight = beta
        self.total_loss = None

        self.loss_network = models.vgg19(weights='IMAGENET1K_V1')

        self.gram = GramMatrix()
        self.loss = nn.MSELoss()
        self.optimizerLBFGS = optim.LBFGS([self.pastiche])
        self.optimizerAdam = optim.Adam([self.pastiche], lr=lr)

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            print('Using GPU device')
            self.loss_network.cuda()
            self.gram.cuda()
        else:
            print('Using CPU device')
            self.loss_network.cpu()
            self.gram.cpu()

    def trainLBFGS(self):
        def closure():
            self.optimizerLBFGS.zero_grad()

            pastiche = self.pastiche.clone()
            pastiche.data.clamp_(0,1)
            content = self.content.clone()
            style = self.style.clone()

            content_loss = 0
            style_loss = 0

            i = 1
            not_inplace = lambda layer: nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer
            for layer in list(self.loss_network.features):
                layer = not_inplace(layer)
                if self.use_cuda:
                    layer.cuda()
                else:
                    layer.cpu()
                
                pastiche, content, style = layer.forward(pastiche), layer.forward(content), layer.forward(style)

                if isinstance(layer, nn.Conv2d):
                    name = "conv_" +str(i)

                    if name in self.content_layers:
                        content_loss += self.loss(pastiche * self.contentWeight, content.detach() * self.contentWeight)

                    if name in self.style_layers:
                        pastiche_g, style_g = self.gram.forward(pastiche), self.gram.forward(style)
                        style_loss += self.loss(pastiche_g * self.styleWeight, style_g.detach() * self.styleWeight)
                    
                if isinstance(layer, nn.ReLU):
                    i += 1
            
            total_loss = content_loss + style_loss
            self.total_loss = total_loss
            total_loss.backward()

            return total_loss
    
        self.optimizerLBFGS.step(closure)
        return self.pastiche
    
    
    def trainAdam(self, steps):
        for i in range(steps):
            self.optimizerAdam.zero_grad()

            pastiche = self.pastiche.clone()
            pastiche.data.clamp_(0, 1)
            content = self.content.clone()
            style = self.style.clone()

            content_loss = 0
            style_loss = 0

            i = 1
            not_inplace = lambda layer: nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer
            for layer in list(self.loss_network.features):
                layer = not_inplace(layer)
                if self.use_cuda:
                    layer.cuda()
                else:
                    layer.cpu()

                pastiche, content, style = layer.forward(pastiche), layer.forward(content), layer.forward(style)

                if isinstance(layer, nn.Conv2d):
                    name = "conv_" + str(i)

                    if name in self.content_layers:
                        content_loss += self.loss(pastiche * self.contentWeight, content.detach() * self.contentWeight)

                    if name in self.style_layers:
                        pastiche_g, style_g = self.gram.forward(pastiche), self.gram.forward(style)
                        style_loss += self.loss(pastiche_g * self.styleWeight, style_g.detach() * self.styleWeight)

                    if isinstance(layer, nn.ReLU):
                        i += 1

            total_loss = content_loss + style_loss
            self.total_loss = total_loss
            total_loss.backward()

            self.optimizerAdam.step()
        return self.pastiche
    

    def createArt(style, content, alpha, beta, epochs, height, width):
        device = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        loader = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor()
        ])

        unloader = transforms.ToPILImage()
       
        path = f"../data/out/test_{epochs}.png"
        startTime = time.time()
        canva = content.clone()
        model = CNN(style, content, canva, alpha, beta)

        for i in range(epochs +1):
            canva = model.trainLBFGS()
            print("EPoch: %d" % (i))
            print("Elapsed time:{:.2f} ".format(time.time() - startTime))
        
        save_image(canva, path)
        return 'olis'


 