import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import os
import numpy as np
from PIL import Image

from model import StyleTransfer
from loss import ContentLoss, StyleLoss
from tqdm import tqdm

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def pre_processing(image: Image.Image) -> torch.Tensor:
    preprocessing = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    image_tensor = preprocessing(image).unsqueeze(0)
    return image_tensor

def post_processing(tensor: torch.Tensor) -> Image.Image:
    image = tensor.to('cpu').detach().numpy()
    image = image.squeeze()
    image = image.transpose(1, 2, 0)
    image = image * std + mean
    image = image.clip(0, 1) * 255
    image = image.astype(np.uint8)
    return Image.fromarray(image)

def train_main():
    # Load data
    content_image = Image.open('./content.jpg')
    content_image = pre_processing(content_image)

    style_image = Image.open('./style.jpg')
    style_image = pre_processing(style_image)

    # Load model and losses
    style_transfer = StyleTransfer().eval()
    content_loss = ContentLoss()
    style_loss = StyleLoss()

    # Hyperparameters
    alpha = 1
    beta = 1e6
    learning_rate = 1

    save_root = f'alpha={alpha}_beta={beta}_lr={learning_rate}_initContent_style2_LBFGS'
    os.makedirs(save_root, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    style_transfer = style_transfer.to(device)
    content_image = content_image.to(device)
    style_image = style_image.to(device)

    x = content_image.clone()
    x.requires_grad_(True)

    # Use SGD optimizer instead of LBFGS
    optimizer = optim.SGD([x], lr=learning_rate)

    def closure():
        optimizer.zero_grad()

        x_content_list = style_transfer(x, 'content')
        y_content_list = style_transfer(content_image, 'content')
        x_style_list = style_transfer(x, 'style')
        y_style_list = style_transfer(style_image, 'style')

        loss_c, loss_s = 0, 0
        for x_style, y_style in zip(x_style_list, y_style_list):
            loss_s += style_loss(x_style, y_style)
        loss_s = beta * loss_s

        for x_content, y_content in zip(x_content_list, y_content_list):
            loss_c += content_loss(x_content, y_content)
        loss_c = alpha * loss_c

        loss_total = loss_c + loss_c  # original: content twice
        loss_total.backward()
        return loss_total

    # Train loop
    num_epochs = 1000
    for epoch in tqdm(range(num_epochs)):
        # Compute gradients via closure, then update with SGD
        _ = closure()
        optimizer.step()

        # Recompute losses for reporting
        x_content_list = style_transfer(x, 'content')
        y_content_list = style_transfer(content_image, 'content')
        x_style_list = style_transfer(x, 'style')
        y_style_list = style_transfer(style_image, 'style')

        loss_c, loss_s = 0, 0
        for x_style, y_style in zip(x_style_list, y_style_list):
            loss_s += style_loss(x_style, y_style)
        loss_s = beta * loss_s

        for x_content, y_content in zip(x_content_list, y_content_list):
            loss_c += content_loss(x_content, y_content)
        loss_c = alpha * loss_c

        loss_total = loss_c + loss_c  # original: content twice

        print(f"Style Loss: {np.round(loss_s.cpu().detach().numpy(), 4)}")
        print(f"Content Loss: {np.round(loss_c.cpu().detach().numpy(), 4)}")
        print(f"Total Loss: {np.round(loss_total.cpu().detach().numpy(), 4)}")

        gen_img = post_processing(x)
        gen_img.save(os.path.join(save_root, f'{epoch}.jpg'))

if __name__ == "__main__":
    train_main()
