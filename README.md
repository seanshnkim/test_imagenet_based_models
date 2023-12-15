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

<br>

## How to Download Datasets from Hugging Face
1. Go to [access token page in Huggingface](https://huggingface.co/settings/tokens)

![Alt text](attachments/hg_token_creating_page.png?raw=true "Access Token Page in Huggingface")

2. Create a new token and copy the content.
3. Go to your terminal, and type `huggingface-cli login`.

![Alt text](attachments/token_create_terminal_output.png?raw=true "Terminal Output After Login")

4. Type in your pasted token. Since it is credential, it wouldn't show any sign of entering the token (e.g. cursor location being changed).

