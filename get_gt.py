# Set up data loaders
import csv
# from datasets import ImageFolderDataset

query_folder = 'data/AIC20_ReID/image_query'
gallery_folder = 'data/AIC20_ReID/image_test'
query_csv = 'data/list/Label-Test-Query - Query.csv'
gallery_csv = 'data/list/Label-Test-Query - Test.csv'

size = (224, 224)


dict_cluster_codes = {}
id_codes = 0

query_images = []
query_cluster_codes = []

with open(query_csv, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)
    for row in csv_reader:
        image_name, cluster_code = row[0], row[5]
        cluster = cluster_code.split("_")[2]
        if int(cluster) > 0 and int(cluster) <= 50:
            query_images.append(image_name)
            if cluster_code not in dict_cluster_codes:
                dict_cluster_codes[cluster_code] = id_codes
                id_codes += 1
            query_cluster_codes.append(dict_cluster_codes[cluster_code])

gallery_images = []
gallery_cluster_codes = []

track_txt = 'data/AIC20_ReID/test_track.txt'
tracklet_lists = [[] for i in range(798)]  # number of trackets
tracklet_id = 0
lines = [line.rstrip('\n') for line in open(track_txt, 'r')]
for line in lines:
    image_names = line.split(" ")[:-1]
    for image_name in image_names:
        tracklet_lists[tracklet_id].append(image_name)
    tracklet_id += 1

with open(gallery_csv, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)
    for row in csv_reader:
        image_name, cluster_code = row[0], row[5]
        cluster = cluster_code.split("_")[2]
        if int(cluster) > 0 and int(cluster) <= 50 and "_" in image_name:
            tracklet_name = int(image_name.split("_")[0])
            for image_name in tracklet_lists[tracklet_name]:
                gallery_images.append(image_name)
                if cluster_code not in dict_cluster_codes:
                    dict_cluster_codes[cluster_code] = id_codes
                    id_codes += 1
                gallery_cluster_codes.append(dict_cluster_codes[cluster_code])

query_data = [['image_id', 'vehicle_id']] + \
    list(map(list, zip(query_images, query_cluster_codes)))
gallery_data = [['image_id', 'vehicle_id']] + \
    list(map(list, zip(gallery_images, gallery_cluster_codes)))

csv.writer(open('data/list/reid_query_gt.csv', 'w')).writerows(query_data)
csv.writer(open('data/list/reid_gallery_gt.csv', 'w')).writerows(gallery_data)

# query_dataset = ImageFolderDataset(query_folder, query_images, query_cluster_codes,
#                                        transform = transforms.Compose([
#                                         transforms.Resize(size),
#                                         transforms.ToTensor()
#                                       ]))
# gallery_dataset = ImageFolderDataset(gallery_folder, gallery_images, gallery_cluster_codes,
#                                      transform = transforms.Compose([
#                                         transforms.Resize(size),
#                                         transforms.ToTensor()
#                                       ]))

# batch_size = 8
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# gallery_loader = torch.utils.data.DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, **kwargs)
