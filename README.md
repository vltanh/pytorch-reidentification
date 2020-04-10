# Usage

## Train

### Train

To train, run
```
  python train.py --config path/to/config/file [--gpus gpu_id] [--debug]
```

Arguments:
```
  --config: path to configuration file
  --gpus: gpu id to be used
  --debug: to save the weights or not
```

For example:
```
  python train.py --config configs/train/debug_ircad.yaml --gpus 0 --debug
```

### Config

Modify the default configuration file (YAML format) to suit your need. The mechanism behind is exactly the same as creating an object of a class noted in ```name```, with keyword arguments noted in ```args```.

### Result

All the result will be stored in the ```runs``` folder in separate subfolders, one for each run. The result consists of the log file for Tensorboard, the network pretrained models (best metrics, best loss, and the latest iteration).

#### Training graph

This project uses Tensorboard to plot training graph. To see it, run

```
  tensorboard --logdir=logs
```

and access using the announced port (default is 6006, e.g ```http://localhost:6006```).

#### Pretrained models

The ```.pth``` files contains a dictionary:

```
  {
      'epoch':                the epoch of the training where the weight is saved
      'model_state_dict':     model state dict (use model.load_state_dict to load)
      'optimizer_state_dict': optimizer state dict (use opt.load_state_dict to load)
      'config':               full configuration of that run
  }
```

# Credit

This repository heavily borrows the code from https://github.com/adambielski/siamese-triplet with modification to fit with the template. Specifically, things taken from the repository linked above include:
- Siamese/Triplet Network
- (Online) Contrastive Loss/Triplet Loss
- Balanced Batch Sampler
- All/Random/Hard negative mining of Pairs/Triplets

Mean Average Precision (mAP) is borrowed and modified from https://github.com/CoinCheung/triplet-reid-pytorch.
