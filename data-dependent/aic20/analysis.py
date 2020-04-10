from itertools import combinations, product
from PIL import Image
import matplotlib.pyplot as plt
import json
import csv

root = 'data/AIC20_ReID'
folder_name = 'image_train'

# Load JSON
dataset_json_path = 'data/list/train_image_metadata.json'
dataset = json.load(open(dataset_json_path))

if False:
    ntracks_per_vehicle = 4
    nimages_per_track = 6

    # Check JSON
    for vehicle_id, tracks in dataset.items():
        fig, axes = plt.subplots(ntracks_per_vehicle, nimages_per_track)
        fig.suptitle(f'{vehicle_id}')
        for track_idx, (track_id, track) in enumerate(list(tracks.items())[:ntracks_per_vehicle]):
            for image_idx, image_filename in enumerate(track[:nimages_per_track]):
                image = Image.open(f'{root}/{folder_name}/{image_filename}')
                ax = axes[track_idx, image_idx]
                ax.imshow(image)
                ax.set_title(image_filename)
        fig.tight_layout()
        plt.show()
        plt.close()

print('Check matching image_filename and vehicle_id...')


def find_track_id(instance):
    image_filename, vehicle_id = instance
    tracks = dataset[vehicle_id]
    for track_id, track in tracks.items():
        if image_filename in track:
            return track_id
    return None


# def find_track_id(instance):
#     track_id, vehicle_id = instance
#     tracks = dataset[vehicle_id]
#     if track_id in tracks:
#         return track_id
#     return None


splits = dict()
for split in ['train', 'gallery', 'query']:
    csv_data = csv.reader(open(f'data/list/reid_{split}_easy.csv'))
    next(csv_data)
    csv_data = list(csv_data)
    track_ids = list(map(find_track_id, csv_data))
    print(split, sum(x == None for x in track_ids))

    image_filenames, vehicle_ids = list(zip(*csv_data))

    splits[split] = dict()
    splits[split]['image_filenames'] = list(image_filenames)
    splits[split]['vehicle_ids'] = list(vehicle_ids)
    splits[split]['track_ids'] = [
        f'{x}_{y}' for x, y in zip(list(vehicle_ids), list(track_ids))]


print('Check number of UNIQUE images and UNIQUE ids...')
for split, data in splits.items():
    unique_image_filenames = set(data['image_filenames'])
    unique_vehicle_ids = set(data['vehicle_ids'])
    unique_track_ids = set(data['track_ids'])
    print(split, len(unique_vehicle_ids), len(unique_track_ids),
          len(data['image_filenames']), len(unique_image_filenames))

all_filenames = set(sum([x['image_filenames'] for x in splits.values()], []))
all_vehicle_ids = set(sum([x['vehicle_ids'] for x in splits.values()], []))
all_track_ids = set(sum([x['track_ids'] for x in splits.values()], []))
print('total', len(all_filenames), len(all_vehicle_ids), len(all_track_ids))


print('Check leakage...')

for split_1, split_2 in combinations(splits.keys(), 2):
    print(split_1, split_2)
    for value in splits[split_1].keys():
        unique_image_filenames_1 = set(splits[split_1][value])
        unique_image_filenames_2 = set(splits[split_2][value])

        leakage_filenames_21 = list(filter(
            lambda x: x in unique_image_filenames_1,
            unique_image_filenames_2))
        leakage_filenames_12 = list(filter(
            lambda x: x in unique_image_filenames_2,
            unique_image_filenames_1))

        print(value, len(leakage_filenames_12), len(leakage_filenames_21))
