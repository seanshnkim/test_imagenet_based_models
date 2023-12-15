import os
from glob import glob


# https://www.image-net.org/challenges/LSVRC/2012/browse-synsets.php

imagenet_dir = '../msn_shoeprint_retrieval/imagenet/val'
synsets = glob(f"{imagenet_dir}/*/")

synsets = [x.split('/')[-2] for x in synsets]
synsets.sort()
synsets_dict = {x: i for i, x in enumerate(synsets)}

# save synsets_dict
with open("synset_label.txt", "w") as f:
    for key, value in synsets_dict.items():
        f.write(f"{key} {value}\n")