# Defines the Dataset class that defines how Samples are given out for feeding into the Batch
""" 
Must have two definite class methods implemented.
1. __len__
2. __getitem__
"""

import pandas as pd
import torch 
import numpy as np

class CustomDataset(object):
    def __init__(self, folds, img_height, img_width, mean, std):
        # Read the train_folds.csv file
        df = pd.read_csv("../input/train_folds.csv")
        df = df[df.kfold.isin(folds)]
        
        self.images = df.images.values

    # Defines the total length of the training or validation samples
    def __len__(self):
        return len(self.images)
    
    # Implemented this 
    # Remember that the torch needs the image in the form 
    # CHW but the numpy or PIL stores them as HWC
    # do transpose the image with np.transpose(2, 0, 1)
    def __getitem__(self, item_id):
        # Returns a dict of image_object and label
        return {
            "image" : None,
            "label" : None
        }


