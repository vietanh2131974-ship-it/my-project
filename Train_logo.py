import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import h5py
import numpy as np
import torch.nn.functional as F

# Mô hình mạng neural
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 3)  # Số lượng lớp = số lượng loại hình xe

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.data = ImageFolder(folder_path, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index]
        return img, label

def train_model(model, train_loader, criterion, optimizer, epochs=50):
    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = Variable(images), Variable(labels)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Tính Accuracy
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        print(
            f'Epoch {epoch+1}/{epochs}, '
            f'Loss: {running_loss/len(train_loader):.4f}, '
            f'Accuracy: {accuracy:.2f}%'
        )

    print('Training complete!')

if __name__ == "__main__":
    # Thay đổi đường dẫn tới thư mục tiền xử lý ảnh
    preprocessed_data_folder = "preprocessed_data"

    # Áp dụng các biến đổi cho dữ liệu
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    ])

    # Tạo đối tượng CustomDataset
    dataset = CustomDataset(preprocessed_data_folder, transform=transform)

    # Chia dữ liệu thành tập huấn luyện và tập kiểm thử
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Tạo DataLoader cho tập huấn luyện
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Khởi tạo mô hình và các tham số
    model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Đào tạo mô hình
    train_model(model, train_loader, criterion, optimizer)

    # Lưu mô hình dưới dạng tệp HDF5
    with h5py.File('vehicle_model.h5', 'w') as hdf:
        for name, param in model.named_parameters():
            hdf.create_dataset(name, data=param.detach().cpu().numpy())
    print('Mô hình đã được lưu.')
