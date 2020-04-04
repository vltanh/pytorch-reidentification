import os
from pathlib import Path
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torchvision as tv
import xml.etree.ElementTree as ET
import numpy as np
import json

root_dir = Path('data/AIC20_ReID')
dataset = {x: root_dir / f'image_{x}'
           for x in ['train', 'test']}
track_txts = {x: root_dir / f'{x}_track.txt'
              for x in dataset.keys()}
train_xml_path = 'train_label.xml'

xml_data = ET.parse(str(root_dir / train_xml_path),
                    parser=ET.XMLParser(encoding='iso-8859-5')).getroot()[0]
labels = dict()
for x in xml_data:
    x = x.attrib
    labels[x['imageName']] = (x['vehicleID'], x['cameraID'])

lines = open(track_txts['train']).readlines()
tracks = [x.strip().split() for x in lines if len(x.strip()) != 0]

vehs = dict()
for i, track in enumerate(tracks):
    veh_id, cam_id = zip(*[labels[img_id] for img_id in track])
    veh_id = veh_id[0]
    cam_id = cam_id[0]

    vehs.setdefault(veh_id, dict())
    vehs[veh_id][cam_id] = track

for image_name, (veh_id, cam_id) in labels.items():
    vehs[veh_id].setdefault(cam_id, [])
    if image_name not in vehs[veh_id][cam_id]:
        print(f'{image_name} is not found in {veh_id}/{cam_id}')
        vehs[veh_id][cam_id].append(image_name)

json.dump(vehs, open('data.json', 'w'))
