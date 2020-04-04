import json
import csv
import random
random.seed(3698)

dataset_json_path = 'list/train_image_metadata.json'
val_csv_path = 'list/cls_val.csv'
train_csv_path = 'list/cls_train.csv'
easy_mode = True
gallery_ratio = 0.8

# Stratified split
splits = {
    'gallery': dict(),
    'query': dict(),
    'train': dict()
}

# Load JSON
reference = json.load(open(dataset_json_path))

# Load CSV
dataset = csv.reader(open(val_csv_path))
next(dataset)

for veh_id, _ in dataset:
    tracks = list(reference[veh_id].keys())
    if easy_mode:
        def get_random_image(x):
            if len(x) == 1:
                return (x[0], [])
            i = random.choice(range(len(x)))
            return (x[i], x[:i] + x[i+1:])

        query, gallery = zip(
            *map(get_random_image, reference[veh_id].values()))

        splits['gallery'][veh_id] = sum(gallery, [])
        splits['query'][veh_id] = list(query)
    else:
        total = len(tracks)
        train_sz = max(1, int(total * gallery_ratio))
        test_sz = total - train_sz

        random.shuffle(tracks)
        splits['gallery'][veh_id] = tracks[:train_sz]
        splits['query'][veh_id] = tracks[train_sz:]

# Load CSV
dataset = csv.reader(open(train_csv_path))
next(dataset)

s = 0
for veh_id, _ in dataset:
    tracks = list(reference[veh_id].keys())
    if easy_mode:
        gallery = reference[veh_id].values()
        splits['train'][veh_id] = sum(gallery, [])
        s += len(splits['train'][veh_id])
    else:
        splits['train'][veh_id] = tracks
print(s)

# Output split to CSV
for split in splits.keys():
    rows = [['image_name' if easy_mode else 'camera_id', 'vehicle_id']]
    for k, v in splits[split].items():
        rows.extend([[x, k] for x in v])
    with open(f"list/reid_{split}_{'easy' if easy_mode else 'hard'}.csv", 'w') as f:
        csv.writer(f).writerows(rows)
