import json
import csv
import xml.etree.ElementTree as ET
import random
random.seed(3698)

dataset_json_path = 'list/train_image_metadata.json'
label_path = 'list/train_vehicle_type.csv'
train_xml_path = 'data/AIC20_ReID/train_label.xml'

# Load JSON
dataset = json.load(open(dataset_json_path))

# Load CSV
label = csv.reader(open(label_path))
next(label)

# Load XML
xml_data = ET.parse(train_xml_path,
                    parser=ET.XMLParser(encoding='iso-8859-5')).getroot()[0]

# Generate mapping imagename -> (vehicle id, camera id)
mapping = dict()
for x in xml_data:
    x = x.attrib
    mapping[x['imageName']] = (x['vehicleID'], x['cameraID'])

# Generate mapping label -> list of instances with corresponding label
ds = dict()
for i, line in enumerate(label):
    img_id, vtype = line
    veh_id, cam_id = mapping[img_id + '.jpg']
    ds.setdefault(vtype, set())
    ds[vtype].add(veh_id)

# Stratified split
trainval = {
    'train': dict(),
    'val': dict()
}
train_ratio = 0.7

for k, v in ds.items():
    total = len(v)
    train_sz = int(len(v) * train_ratio)
    test_sz = total - train_sz

    v = list(v)
    random.shuffle(v)
    trainval['train'][k] = v[:train_sz]
    trainval['val'][k] = v[train_sz:]

# Output split to CSV
for phase in ('train', 'val'):
    rows = [['vehicle_id', 'label']]
    for k, v in trainval[phase].items():
        rows.extend([[x, k] for x in v])
    csv.writer(open(f'cls_{phase}.csv', 'w')).writerows(rows)
