import pandas as pd
import json
import csv


def get_df_hard(path):
    reference = json.load(open('data/list/train_image_metadata.json'))
    camera_ids, vehicle_ids = zip(*list(csv.reader(open(path)))[1:])
    tracks = [x for camera_id, vehicle_id in zip(camera_ids, vehicle_ids)
              for x in reference[vehicle_id][camera_id]]
    labels = [vehicle_id
              for camera_id, vehicle_id in zip(camera_ids, vehicle_ids)
              for _ in range(len(reference[vehicle_id][camera_id]))]
    return pd.DataFrame({
        'image_name': tracks,
        'vehicle_id': labels
    }, columns=['image_name', 'vehicle_id'])


d_data = pd.read_csv('data/list/Train_Direction.csv')

splits = ['train', 'gallery', 'query']
data = {split: get_df_hard(f'data/list/reid_{split}_hard.csv')  # pd.read_csv(f'data/list/reid_{split}_hard.csv')
        for split in splits}
print(data)

for split in splits:
    merged = pd.merge(d_data, data[split], left_on='0', right_on='image_name')
    dfs = {y: x[['image_name', 'vehicle_id']]
           for y, x in merged.groupby('2')}

    for direction, df in dfs.items():
        print(f'{split}, {direction}, {len(df)}')
        df.to_csv(
            f'data/list/reid_direction/reid_{split}_{direction}_hard.csv', index=False)
