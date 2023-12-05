# from transformers import TrainingArguments, ViTMSNPreTrainedModel, ViTMSNModel, ViTMSNConfig, ViTMSNForImageClassification
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

import evaluate

import os

pretrained_model = 'google/vit-base-patch16-224'
model = ViTForImageClassification.from_pretrained(pretrained_model)
processor = ViTImageProcessor.from_pretrained(pretrained_model)

accuracy = evaluate.load("accuracy")

imagenet_dir = '../msn_shoeprint_retrieval/imagenet/val'
cur_synset = 'n04067472'
data_subset_dir = os.path.join(imagenet_dir, cur_synset)

synset_dict = {}
with open("synset_label.txt") as f:
    content = f.readlines()
    synset_dict = {x.split()[0]: x.split()[1] for x in content}

for img in os.listdir(data_subset_dir):
    img_fpath = os.path.join(data_subset_dir, img)
    
    image = processor(Image.open(img_fpath).convert('RGB'), return_tensors="pt")
    
    with torch.no_grad():
        logits = model(**image).logits

    predicted_label = logits.argmax(-1).item()
    label = int(synset_dict[cur_synset])
    
    accuracy.add(predictions=predicted_label, references=label)

accuracy.compute()