from PIL import Image

import os


class ImageFolderDataset():
    def __init__(self, directory, transform):
        assert os.path.isdir(directory), 'Invalid directory.'
        self.directory = directory
        self.image_filenames = os.listdir(self.directory)
        self.transform = transform

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image = Image.open(os.path.join(self.directory, image_filename))
        image = self.transform(image)
        return image, image_filename

    def __len__(self):
        return len(self.image_filenames)
