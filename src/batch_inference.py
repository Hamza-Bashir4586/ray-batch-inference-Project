import ray
import torch
import numpy as np
import logging
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset

# Initialize Ray
ray.init(num_gpus=1, object_store_memory=4 * 1024 * 1024 * 1024)
logging.info("Ray initialized.")

# Load pre-trained model
model = models.resnet50(pretrained=True).eval().cuda()
model_state_dict_ref = ray.put(model.state_dict())

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
dataset = datasets.CIFAR10(root="./data", download=True, transform=transform)
half_dataset = Subset(dataset, range(len(dataset) // 2))
data_loader = DataLoader(half_dataset, batch_size=20, shuffle=False)

@ray.remote(num_gpus=0.5)
class BatchProcessor:
    def __init__(self, model_state_dict):
        self.model = models.resnet50()
        self.model.load_state_dict(model_state_dict)
        self.model.cuda().eval()

    def process(self, batch):
        batch = batch.cuda()
        with torch.no_grad():
            predictions = self.model(batch).cpu().numpy()
        return predictions

actors = [BatchProcessor.remote(model_state_dict_ref) for _ in range(2)]
results = [actor.process.remote(batch) for batch, _ in data_loader]
all_predictions = ray.get(results)

logging.info(f"Processed {len(all_predictions)} batches.")
