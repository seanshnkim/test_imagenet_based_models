import torch

from transformers import ViTImageProcessor, ViTForImageClassification

pretrained_model = 'google/vit-base-patch16-224'
model = ViTForImageClassification.from_pretrained(pretrained_model)
processor = ViTImageProcessor.from_pretrained(pretrained_model)


# TEST (a single image)
for img in os.listdir('imagenet/val/n04067472'):
    img_fpath = os.path.join('imagenet/val/n04067472', img)
    
    image = processor(Image.open(img_fpath).convert('RGB'), return_tensors="pt")
    
    with torch.no_grad():
        logits = model(**image).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    print(model.config.id2label[predicted_label])
    
