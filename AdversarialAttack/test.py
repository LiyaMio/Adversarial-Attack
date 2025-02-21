import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def imshow(img, title, ax):
    img = img.squeeze().numpy()
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')


def main():
    epsilon = 0.1
    model_path = 'mnist_cnn.pth'

    # 自动检测模型文件是否存在
    if not os.path.exists(model_path):
        print("模型文件未找到，开始训练...")
        train_model = True
    else:
        train_model = False

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if train_model:
        for epoch in range(1):
            for images, labels in trainloader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存至 {model_path}")
    else:
        model.load_state_dict(torch.load(model_path))
        print(f"已加载预训练模型 {model_path}")

    model.eval()
    successful = False

    for data, target in testloader:
        if successful: break

        data.requires_grad = True
        output = model(data)
        init_pred = output.argmax(dim=1)

        if init_pred.item() != target.item():
            continue

        loss = criterion(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        output_adv = model(perturbed_data)
        final_pred = output_adv.argmax(dim=1)

        if final_pred.item() != target.item():
            successful = True

    if successful:
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        imshow(data.detach(), f"Original ({target.item()})", axes[0])
        perturbation = (perturbed_data - data).detach() * 10
        imshow(perturbation, "Perturbation (x10)", axes[1])
        imshow(perturbed_data.detach(), f"Adversarial ({final_pred.item()})", axes[2])
        plt.tight_layout()
        plt.show()
    else:
        print("攻击失败！建议：1. 增大epsilon值 2. 增加训练epoch")


if __name__ == "__main__":
    main()