# test_imagenet_based_models

This repository mainly des
1. Explains how to handle ImageNet1K data for beginners
2. Treats basic PyTorch and Hugging Face models that are trained on ImageNet1K.

<br>

## Requirements
- Python 3.6+ (recommend newer than Python 3.10)
- PyTorch 1.1.0+ (recommend 2.1.0 or newer)
- torchvision
- Other dependencies: transformers

```pip install transformers```

<br>

## What is ImageNet dataset?
- One of the most authoritative benchmark datasets for computer vision tasks.
- There are various versions of ImageNet dataset. In most cases, ImageNet datset refers to ILSVRC2012.
- **Let us start from the most simple computer vision task: Image Classification.**
- Our target is ImageNet1K

## How to Download ImageNet Dataset for Image Classification
1. Go to [ImageNet Official Site](image-net.org).
2. In order to download the dataset, you must sign up and verify through email.
3. Click 'Download' menu > select '2012' from ImageNet Large-scale Visual Recognition Challenge (ILSVRC).

![Alt text](attachments/download_imagenet.png?raw=true "How to download ImageNet1K")

![Alt text](attachments/download_train_val_test_each.png?raw=true "Download each dataset")
