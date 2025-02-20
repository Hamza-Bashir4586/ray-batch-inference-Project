import torch
from src.model import load_model
from src.data_loader import get_data_loader

# Load model
model = load_model()
data_loader = get_data_loader()

# Perform inference
model.eval()
for batch, _ in data_loader:
    batch = batch.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    with torch.no_grad():
        outputs = model(batch)
        predictions = torch.argmax(outputs, dim=1)
    print(predictions)
