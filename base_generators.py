# import torch
# import keras
# import numpy as np


# class BasePTDataset(torch.utils.data.Dataset):
#     pass


# class BaseKerasDataset(keras.utils.Sequence):

#     def __init__(self, data, labels, shuffle, bs):
#         self.data = data
#         self.labels = labels
#         self.shuffle = shuffle
#         self.batch_size = bs

#     def __len__(self):
#         return self.data.shape[0] // self.batch_size

#     def on_epoch_end(self):
#         self.indexes = np.arange(len(self.data.shape[0]))
#         if self.shuffle:
#             np.random.shuffle(self.indexes)
#     //////...............//////////
