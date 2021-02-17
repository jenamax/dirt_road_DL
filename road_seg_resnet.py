# -*- coding: utf-8 -*-
"""road_seg_resnet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LPjT7zxTTVQU1_NYQyhs7AJKtbbMOtZf
"""

# !pip install segmentation-models-pytorch

import cv2
import numpy as np
import glob
import numpy as np
import torch
import segmentation_models_pytorch as sm
import glob
from os import system
import random

images_names = []

images_train = []
labels_train = []

images_test = []
labels_test = []


for file in glob.glob("data1/img/*.png"):
    images_names.append(file.split("/")[-1])

images_names.sort()

for i in range(0, len(images_names)): 
    print(i, end=" ")
    if i < 0.7 * len(images_names):
      images_train.append(np.asarray(cv2.imread("data1/img/" + images_names[i]))[0:704, :, :])
      images_train[-1] = cv2.resize(images_train[-1], (512, 256))
    else:
      images_test.append(np.asarray(cv2.imread("data1/img/" + images_names[i]))[0:704, :, :])
      images_test[-1] = cv2.resize(images_test[-1], (512, 256))

print()

for i in range(0, len(images_names)): 
    print(i, end=" ")
    if i < 0.7 * len(images_names):
      labels_train.append(np.asarray(cv2.imread("data1/masks/" + images_names[i], 0))[0:704, :])
      labels_train[-1] = cv2.resize(labels_train[-1], (512, 256))
    else:
      labels_test.append(np.asarray(cv2.imread("data1/masks/" + images_names[i], 0))[0:704, :])
      labels_test[-1] = cv2.resize(labels_test[-1], (512, 256))

print()

images_names = []

for file in glob.glob("data2/img/*.png"):
    images_names.append(file.split("/")[-1])

images_names.sort()

for i in range(0, len(images_names)): 
    print(i, end=" ")
    if i < 0.7 * len(images_names):
      images_train.append(np.asarray(cv2.imread("data2/img/" + images_names[i])[0:704, :, :])) 
      images_train[-1] = cv2.resize(images_train[-1], (512, 256)) 
    else:
      images_test.append(np.asarray(cv2.imread("data2/img/" + images_names[i])[0:704, :, :]))
      images_test[-1] = cv2.resize(images_test[-1], (512, 256))

print()

for i in range(0, len(images_names)): 
    print(i, end=" ")
    if i < 0.7 * len(images_names):
      labels_train.append(np.asarray(cv2.imread("data2/masks/" + images_names[i], 0))[0:704, :])
      labels_train[-1] = cv2.resize(labels_train[-1], (512, 256))
    else:
      labels_test.append(np.asarray(cv2.imread("data2/masks/" + images_names[i], 0))[0:704, :])
      labels_test[-1] = cv2.resize(labels_test[-1], (512, 256))

from numpy.random import normal
def augment(images, labels):
  images_aug = []
  labels_aug = []
  for i in range(len(images)):
    for j in range(0, 2):
      aug = cv2.cvtColor(images[i], cv2.COLOR_BGR2HSV)
      aug[:, :, 2] += np.uint8(normal(0, 30))
      aug = cv2.cvtColor(images[i], cv2.COLOR_HSV2BGR)
      aug[:, :, 0] += np.uint8(normal(0, 30))
      aug[:, :, 1] += np.uint8(normal(0, 30))
      aug[:, :, 2] += np.uint8(normal(0, 30))
      images_aug.append(aug)
      labels_aug.append(labels[i])
  return images + images_aug, labels + labels_aug

# print()
# print(len(images_train), len(images_test))
# images_train, labels_train = augment(images_train, labels_train)
# images_test, labels_test = augment(images_test, labels_test)
# print(len(images_train), len(images_test))

n_rows,n_cols = (64*4,192*4)

rows = images_train[0].shape[1]
cols = images_train[0].shape[0]


images_train = np.asarray(images_train).transpose((0, 3, 1, 2))
images_test = np.asarray(images_test).transpose((0, 3, 1, 2))

print(images_train.shape)

labels_train = np.asarray(labels_train)
labels_test = np.asarray(labels_test)

labels_train = labels_train.astype(np.float32).reshape((labels_train.shape[0], 1, labels_train.shape[1], labels_train.shape[2]))
labels_test = labels_test.astype(np.float32).reshape((labels_test.shape[0], 1, labels_test.shape[1], labels_test.shape[2]))
print(labels_train.shape)

# from keras.callbacks import Callback

# class TestCallback(Callback):
#     def __init__(self, test_data):
#         self.test_data = test_data

#     def on_epoch_end(self, epoch, logs={}):
#         x, y = self.test_data
#         loss, acc = self.model.evaluate(x, y, verbose=0)
#         print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

import torch.utils.data as data

class DataLoaderSegmentation(data.Dataset):
    def __init__(self, images, labels):
        super(DataLoaderSegmentation, self).__init__()
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
            return torch.from_numpy(self.images[index]).float(), torch.from_numpy(self.labels[index]).float()

    def __len__(self):
        return len(self.images)

from torch.utils.data import DataLoader



print(images_train.shape)

train_dataset = DataLoaderSegmentation(
    images_train, 
    labels_train, 
)

valid_dataset = DataLoaderSegmentation(
    images_test, 
    labels_test, 
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

loss = sm.utils.losses.DiceLoss()

metrics = [
    sm.utils.metrics.IoU(threshold=0.5),
]


model = sm.Unet('resnet34', classes=1, in_channels=3, activation='sigmoid', encoder_weights='imagenet')
optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])
# model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=learn_rate), metrics=['binary_accuracy'])
# print(model.summary())
# callbacks = [EarlyStopping(monitor='loss', patience=2, verbose=1, min_delta=0.01), ModelCheckpoint("/content/drive/My Drive/road_detection/weights.h5", monitor='loss', save_best_only=True, verbose=2)]
train_epoch = sm.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=device,
)

valid_epoch = sm.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=device,
)

print("Start training")
max_score = 0

for i in range(0, 10):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, 'best_model.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')
# model.fit(x=images_train, y=labels_train, callbacks=callbacks, epochs=3, batch_size=2, validation_data=(images_test, labels_test))
print("Training finished")

import time
n = np.random.choice(len(valid_dataset))
    
image, gt_mask = valid_dataset[n]

shape = train_dataset[0][0].cpu().numpy().shape

# img = cv2.resize(cv2.imread("/content/drive/My Drive/road_detection/imgs/out290.png"), (shape[2], shape[1]))
# s = time.time()
# image = torch.from_numpy(np.asarray(img.transpose(2, 0, 1), dtype=np.float32))

# x_tensor = image.to(device).unsqueeze(0)
# pr_mask = model.predict(x_tensor)
# pr_mask = (pr_mask.squeeze().cpu().numpy().round())
# print(time.time() - s)
# res = np.asarray(pr_mask, dtype = np.uint8)
# res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
# res *= np.array([255, 0, 0], dtype=np.uint8)
# vis = np.copy(img)
# cv2.addWeighted(img, 1, res, 0.3, 0, vis)
# cv2_imshow(img)
# cv2_imshow(vis)

# import time
# model = sm.Unet('resnet34', classes=1, activation='sigmoid', encoder_weights='imagenet')
# model.load_weights("/content/drive/My Drive/road_detection/weights.h5")
#img = cv2.resize(img, (1280, 704))

# img = images_test[int(random.uniform(0, len(images_test)))] 
# img = cv2.resize(cv2.imread("/content/drive/My Drive/road_detection/img1.png"), (images_test[0].shape[1], images_test[0].shape[0]))
# s = time.time()
# res = np.reshape(model.predict(np.array([img])), (256, 512))
# print(time.time() - s)
# res[res <= 0.8] = 0
# res[res > 0.8] = 1
# res = np.asarray(res, dtype = np.uint8)
# res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
# res *= np.array([255, 0, 0], dtype=np.uint8)
# alpha = 0.7
# vis = np.copy(img)
# cv2.addWeighted(img, alpha, res, 1 - alpha, 0, vis)
# cv2_imshow(img)
# cv2_imshow(vis)