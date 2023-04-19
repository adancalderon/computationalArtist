import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class FastStyleTransfer:
    def __init__(self, content_image_path, style_image_path, output_image_path=None):
        self.content_image = self.load_image(content_image_path)
        self.style_image = self.load_image(style_image_path)
        self.output_image_path = output_image_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.vgg19_model = self.load_vgg19_model()
        self.content_weight = 1.0
        self.style_weight = 1.0
        self.num_epochs = 500
        self.lr = 0.01

    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return transform(image).unsqueeze(0).to(self.device)


    def load_vgg19_model(self):
        vgg19_model = models.vgg19(pretrained=False)
        vgg19_model.load_state_dict(torch.load('./trained_vgg19.pth'))
        vgg19_model.eval()
        vgg19_model.to(self.device)
        return vgg19_model

    def compute_content_loss(self, content_features, output_features):
        return nn.MSELoss()(output_features.relu2_2, content_features.relu2_2)

    def compute_style_loss(self, style_features, output_features):
        style_loss = 0.0
        for ft_y, gm_s in zip(output_features, style_features):
            gm_y = self.gram_matrix(ft_y)
            style_loss += nn.MSELoss()(gm_y, gm_s)
        return style_loss

    def gram_matrix(self, input):
        b, c, h, w = input.size()
        features = input.view(b * c, h * w)
        G = torch.mm(features, features.t())
        return G.div(b * c * h * w)

    def run(self):
        content_image = self.content_image.clone()
        style_image = self.style_image.clone()
        output_image = content_image.clone().requires_grad_(True)
        optimizer = optim.Adam([output_image], lr=self.lr)

        for epoch in range(self.num_epochs):
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            content_features = self.vgg19_model(content_image)
            style_features = self.vgg19_model(style_image)
            output_features = self.vgg19_model(output_image)

            # Compute content loss
            content_loss = self.compute_content_loss(content_features.relu2_2, output_features.relu2_2)

            # Compute style loss
            style_loss = self.compute_style_loss(style_features, output_features)

            # Total loss
            total_loss = self.content_weight * content_loss + self.style_weight * style_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, self.num_epochs, total_loss.item()))
                                # Save intermediate output image
                if self.output_image_path is not None:
                    output_image = self.output_image_path.format(epoch + 1)
                    self.save_image(output_image, output_image.clone().detach())

        # Save final output image
        if self.output_image_path is not None:
            output_image = self.output_image_path.format(self.num_epochs)
            self.save_image(output_image, output_image.clone().detach())

    def save_image(self, image_path, image_tensor):
        image_tensor = image_tensor.clone().detach().cpu()
        image_tensor = image_tensor.squeeze(0)
        image_tensor = self.unnormalize(image_tensor)
        image = transforms.ToPILImage()(image_tensor)
        image.save(image_path)

    def unnormalize(self, tensor):
        tensor = tensor.clone().detach()
        tensor = tensor.cpu()
        tensor = tensor.numpy().transpose(1, 2, 0)
        tensor = tensor * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        tensor = np.clip(tensor, 0, 1)
        return torch.from_numpy(tensor.transpose(2, 0, 1))

