# Ray Batch Inference Project

## Project Overview  
Inspired by **Pinterest’s** innovative use of **Ray for batch inference**, this project explores building a scalable, **distributed image classification pipeline** using **PyTorch** and **Ray**.  

Just as Pinterest efficiently tags millions of images to enhance searchability and content discovery, this project aims to process large image batches, assigning relevant tags in parallel.  

By leveraging **Ray’s parallel processing capabilities** with **ResNet50 as a pretrained model**, the goal is to **optimize large-scale image classification pipelines** to handle high data volumes efficiently.

##Features
- **Parallelized Batch Inference with Ray** (Up to **2x faster** than standard inference)
- **ResNet50 for Image Classification**
- **GPU Acceleration for High-Speed Processing**
- **CIFAR-10 Dataset for Testing (Simulating Large-Scale Tagging)**
- **SQLite Integration for Storing Image Tags**
- **Scalable Backend Infrastructure Simulation**

##  Installation
```
pip install -r requirements.txt
```

## Usage
- **Run the batch inference script:
```
python src/batch_inference.py
```

- **Run the example usage script:
```
python examples/example_usage.py
```
-**Run tests:
```
pytest tests/
```

- Using Ray reduced inference time by ~2x compared to sequential processing.
- Average inference time per batch: 0.22 seconds.


