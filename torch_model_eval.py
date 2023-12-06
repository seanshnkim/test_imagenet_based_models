import torch
import torchvision
import torchvision.transforms as transforms

import os
import logging
from tqdm import tqdm
import yaml
import argparse
from time import localtime, strftime, time


class ImageNet(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root,
        image_folder='imagenet',
        transform=None,
        train=True,
    ):
        suffix = 'train/' if train else 'val/'
        data_path = None

        if data_path is None:
            data_path = os.path.join(root, image_folder, suffix)
        print(f'data-path {data_path}')

        super(ImageNet, self).__init__(root=data_path, transform=transform)
        print('Initialized ImageNet')
        

parser = argparse.ArgumentParser()

parser.add_argument(
    '--fname', type=str,
    help='yaml file containing config file names to launch',
    default='configs.yaml')

args = parser.parse_args()

with open(args.fname, 'r') as y_file:
    params = yaml.load(y_file, Loader=yaml.FullLoader)

pretrained_model = params['model_name']

val_transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))])

root_path = params['root_path']
image_folder = params['image_folder']
BATCH_SIZE = params['batch_size']
NUM_WORKERS = params['num_workers']

val_dataset = ImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=val_transform,
        train=False)

# NOTE: batch_size=64 is the maximum batch size (approximately uses up to 7.9GB) that can fit in my GPU (RTX-2080 8GB)
val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS)

# model = torchvision.models.resnet50(pretrained=True)
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

time_stamp = strftime("%m%d-%H%M", localtime())
model_name = pretrained_model.replace('/', '_')
log_fname = f'eval_{model_name}_{time_stamp}.log'
logging.basicConfig(filename=os.path.join('logs', log_fname), level=logging.INFO,\
                    format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()

logger.info(f"Based on torchvision model: {pretrained_model}")
logger.info(f"Dataset location: {os.path.join(root_path, image_folder)}")
logger.info(f"Batch size: {BATCH_SIZE}")
logger.info(f"Number of workers: {NUM_WORKERS}")
logger.info(f"Number of images: {len(val_dataset)}")

start_time = time()
total, correct = 0, 0

model.eval()
for data in tqdm(val_data_loader):
    with torch.cuda.amp.autocast(enabled=True):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
    
    total += inputs.shape[0]
    num_cur_correct = outputs.max(dim=1).indices.eq(labels).sum().item()
    correct += num_cur_correct
    
    logger.info("Batch accuracy: {:.2f}%".format((num_cur_correct / inputs.shape[0]) * 100) )
    
    del inputs, labels, outputs
end_time = time()

logger.info(f"\nTotal time: {end_time - start_time:.2f}s")
logger.info(f"Custom accuracy: {(correct/total) * 100:.2f}%")