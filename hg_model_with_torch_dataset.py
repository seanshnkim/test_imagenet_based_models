import torch
import torchvision
import torchvision.transforms as transforms

import logging
import os
from tqdm import tqdm

from transformers import ViTForImageClassification, ViTMSNPreTrainedModel

class ImageNet(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root,
        image_folder,
        transform=None,
        train=True,
    ):
        """
        ImageNet

        Dataset wrapper (can copy data locally to machine)

        :param root: root network directory for ImageNet data
        :param image_folder: path to images inside root network directory
        :param tar_file: zipped image_folder inside root network directory
        :param train: whether to load train data (or validation)
        :param job_id: scheduler job-id used to create dir on local machine
        :param copy_data: whether to copy data from network file locally
        """

        suffix = 'train/' if train else 'val/'
        data_path = None

        if data_path is None:
            data_path = os.path.join(root, image_folder, suffix)
        logger.info(f'data-path {data_path}')

        super(ImageNet, self).__init__(root=data_path, transform=transform)
        logger.info('Initialized ImageNet')
        

pretrained_model = 'google/vit-base-patch16-224'
model = ViTForImageClassification.from_pretrained(pretrained_model)

val_transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))])

root_path = '../msn_shoeprint_retrieval'
image_folder = 'imagenet'

val_dataset = ImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=val_transform,
        train=False)

# NOTE: batch_size=64 is the maximum batch size (approximately uses up to 7.9GB) that can fit in my GPU (RTX-2080 8GB)
val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# REVIEW: If you do not load model to GPU, you will get 
# "RuntimeError: Input type (c10:Half) and bias type (float) should be the same"
# Refer to this post: https://discuss.huggingface.co/t/is-transformers-using-gpu-by-default/8500/2
model.to(device)

total, correct = 0, 0
for data in tqdm(val_data_loader):
    with torch.cuda.amp.autocast(enabled=True):
        inputs, labels = data[0].to(device), data[1].to(device)
        # outputs.shape = (batch_size, num_labels) = torch.Size([64, 1000])
        outputs = model(inputs).logits
    '''
    Predictions and/or references don't match the expected format.
    Expected format: {'predictions': Value(dtype='int32', id=None), 'references': Value(dtype='int32', id=None)},
    Input predictions: tensor([[-0.0220, -0.2046],
    '''
    # predicted_labels.shape = (batch_size,) = torch.Size([64])
    # labels.shape = (batch_size,) = torch.Size([64])
    
    # First method (readable)
    '''
    predicted_labels = outputs.argmax(-1)
    total += inputs.shape[0]
    correct += (predicted_labels == labels).sum().item()'''
    
    # Second approach (concise)
    total += inputs.shape[0]
    correct += outputs.max(dim=1).indices.eq(labels).sum().item()
    
    del inputs, labels, outputs

# Custom accuracy: 75.66%    
print(f"Custom accuracy: {(correct/total) * 100:.2f}%")
