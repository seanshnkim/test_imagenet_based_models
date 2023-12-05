import torch
import torchvision
import torchvision.transforms as transforms

import logging
import os
import sys
from tqdm import tqdm

from transformers import ViTForImageClassification
import evaluate

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

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
        

pretrained_model = 'google/vit-base-patch16-224-in21k'
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

val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
accuracy = evaluate.load("accuracy")

# REVIEW: If you do not load model to GPU, you will get 
# "RuntimeError: Input type (c10:Half) and bias type (float) should be the same"
model.to(device)

for data in tqdm(val_data_loader):
    with torch.cuda.amp.autocast(enabled=True):
        inputs, labels = data[0].to(device), data[1].to(device)
        predicted_labels = model(inputs).logits
    
    accuracy.add_batch(predictions=predicted_labels, references=labels)

accuracy.compute()


# image = processor(Image.open(img_fpath).convert('RGB'), return_tensors="pt")
    # with torch.no_grad():
    #     logits = model(**image).logits