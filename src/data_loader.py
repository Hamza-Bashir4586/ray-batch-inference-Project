from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_data_loader(batch_size=20):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.CIFAR10(root="./data", download=True, transform=transform)
    half_dataset = Subset(dataset, range(len(dataset) // 2))
    return DataLoader(half_dataset, batch_size=batch_size, shuffle=False)
