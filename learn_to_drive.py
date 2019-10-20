#!/usr/bin/env python
# coding: utf-8

import os
import json
import sys
import pandas as pd
from dataset import Drive360Loader
from torchvision import models
import torch.nn as nn
import torch
import numpy as np
from scipy import interpolate
import torch.optim as optim
import datetime
from itertools import islice

save_dir = "./"
os.chdir(save_dir)
run_val = False #run a validation at the end or not
detail_size = 1000 #How many batches to print results to console
output_states = list(range(0, 100)) #which epochs to save the submission file for
max_epoch = 5 #number of epochs to run
min_start_batch_idx = -1 #if restarting, what batch to restart running at, -1 means from the beginning
extrapolate = True #whether to extrapolate the data. If True but it is trained on the full dataset, reverts to False anyway
full_start = 100 # Each chapter in the submission file starts at 10s

#Version control when saving outputs
version_name = 'Version 21'
path_name = save_dir + version_name + '.pt'

## Loading data from Drive360 dataset.
# load the config.json file that specifies data 
# location parameters and other hyperparameters 
# required.
config = json.load(open(save_dir + 'config - ' + version_name + '.json'))

# create a train, validation and test data loader
train_loader = Drive360Loader(config, 'train')
validation_loader = Drive360Loader(config, 'validation')
test_loader = Drive360Loader(config, 'test')

# print the data (keys) available for use. See full 
# description of each data type in the documents.
print('Loaded train loader with the following data available as a dict.')
print(train_loader.drive360.dataframe.keys())


# ## Training a basic driving model
# The model that takes uses resnet34 and a simple 2 layer NN + LSTM architecture to predict canSteering and canSpeed. 
class SomeDrivingModel(nn.Module):
    def __init__(self):
        super(SomeDrivingModel, self).__init__()
        final_concat_size = 0
        
        multiplier = 0 #count for the layers depending on config inputs
        
        # Main CNNs
        if config['front']: #CNN for front images
            cnn_front = models.resnet34(pretrained=True)
            self.features_front = nn.Sequential(*list(cnn_front.children())[:-1])
            self.intermediate_front = nn.Sequential(nn.Linear(
                              cnn_front.fc.in_features, 128),
                              nn.ReLU())
            multiplier+=1
        
        if config['multi_camera']['right_left']: #CNN for side images
            cnn_ls = models.resnet34(pretrained=True)
            self.features_ls = nn.Sequential(*list(cnn_ls.children())[:-1])
            self.intermediate_ls= nn.Sequential(nn.Linear(
                              cnn_ls.fc.in_features, 128),
                              nn.ReLU())
            multiplier+=1

            cnn_rs = models.resnet34(pretrained=True)
            self.features_rs = nn.Sequential(*list(cnn_rs.children())[:-1])
            self.intermediate_rs= nn.Sequential(nn.Linear(
                              cnn_rs.fc.in_features, 128),
                              nn.ReLU())
            multiplier+=1
        
        if config['multi_camera']['rear']: #CNN for rear images
            cnn_back = models.resnet34(pretrained=True)
            self.features_back = nn.Sequential(*list(cnn_back.children())[:-1])
            self.intermediate_back= nn.Sequential(nn.Linear(
                              cnn_back.fc.in_features, 128),
                              nn.ReLU())
            multiplier+=1
                
        if config['data']: #NN for the segmentatuon map
            self.nn_data = nn.Sequential(nn.Linear(
                              20, 256), #20 input features from the map + 1 derived from location
                              nn.ReLU(),
                              nn.Linear(
                              256, 128),
                              nn.ReLU())
            multiplier+=1
        
        
        final_concat_size += 128* multiplier
        #print("Number of views is:", multiplier)

        # Main LSTM
        self.lstm = nn.LSTM(input_size=128*multiplier,
                            hidden_size=64,
                            num_layers=3,
                            batch_first=False)
        final_concat_size += 64
        
        # Angle Regressor       
        self.control_angle = nn.Sequential( 
            #Linear layers decreasing in size, with dropouts between each. 
            nn.Linear(final_concat_size, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 1)
        )
        
        # Speed Regressor
        self.control_speed = nn.Sequential(
            #Linear layers decreasing in size, with dropouts between each. 
            nn.Linear(final_concat_size, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, data):
        module_outputs = []
        lstm_i = []
        
        # Loop through temporal sequence of
        # camera images and pass 
        # through the cnn.
        #for k, v in data['cameraFront'].items():
        for k in data['cameraFront']:
            layers = []
            v_front = data['cameraFront'][k]
            #print(data)
            x = self.features_front(v_front)
            x = x.view(x.size(0), -1)
            x = self.intermediate_front(x)
            layers.append(x)
            
            if config['data']:
                v_data = data['hereData'][k]
                y = self.nn_data(v_data)
                layers.append(y)
            
            if config['multi_camera']['right_left']:
                v_ls = data['cameraLeft'][k]
                y = self.features_ls(v_ls)
                y = y.view(y.size(0), -1)
                y = self.intermediate_ls(y)
                layers.append(y)

                v_rs = data['cameraRight'][k]
                z = self.features_rs(v_rs)
                z = z.view(z.size(0), -1)
                z = self.intermediate_rs(z)
                layers.append(z)
            
            if config['multi_camera']['rear']:
                v_back = data['cameraRear'][k]
                u = self.features_back(v_back)
                u = u.view(u.size(0), -1)
                u = self.intermediate_back(u)
                layers.append(u)
            
            lstm_i.append(torch.cat(layers, dim=1))
            # feed the current camera
            # output directly into the 
            # regression networks.
            if k == 0:
                module_outputs.append(torch.cat(layers, dim=1))

        # Feed temporal outputs of CNN into LSTM
        i_lstm, _ = self.lstm(torch.stack(lstm_i))
        module_outputs.append(i_lstm[-1])
        
        # Concatenate current image CNN output 
        # and LSTM output.
        x_cat = torch.cat(module_outputs, dim=-1)

        # Feed concatenated outputs into the 
        # regession networks.
        prediction = {'canSteering': torch.squeeze(self.control_angle(x_cat)),
                      'canSpeed': torch.squeeze(self.control_speed(x_cat))}
        return prediction


# Helper function to add_results to a set of lists when creating the output file. 
def add_results(results, output):
    normalize_targets = config['target']['normalize']
    target_mean = config['target']['mean']
    target_std = config['target']['std']
    
    steering = np.squeeze(output['canSteering'].cpu().data.numpy())
    speed = np.squeeze(output['canSpeed'].cpu().data.numpy())
    if normalize_targets: #denormalize the predictions
        steering = (steering*target_std['canSteering'])+target_mean['canSteering']
        speed = (speed*target_std['canSpeed'])+target_mean['canSpeed']
    if np.isscalar(steering):
        steering = [steering]
    if np.isscalar(speed):
        speed = [speed]
    results['canSteering'].extend(steering)
    results['canSpeed'].extend(speed)

# Function to create the submission
def create_submission(file):
    model.cuda()
    model.eval()

    results = {'canSteering': [],
               'canSpeed': []}
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            for k1 in data:
                for k2 in data[k1]:
                    data[k1][k2] = data[k1][k2].cuda()
            prediction = model(data)
            add_results(results, prediction)
            if batch_idx % detail_size == 0:  
                print('Test:', batch_idx, ': ', datetime.datetime.now()) 

    df = pd.DataFrame.from_dict(results)
    df.to_csv(file, index=False)

# sample rate and start index for different manual sampling frequency datasets
# change the settings if using a different sample dataset
sequence_use = config['data_loader']['historic']['number'] * config['data_loader']['historic']['frequency']
if 'sample1' in config['data_loader']['data_dir'].lower():
    sample_start = 10 # full_start / sample_rate
    sample_rate = 10 # sample rate for sample1
    sequence_start = sequence_use * sample_rate
elif 'sample2' in config['data_loader']['data_dir'].lower():
    sample_start = 5 # full_start / sample_rate
    sample_rate = 20 # sample rate for sample2
    sequence_start = sequence_use * sample_rate
elif 'sample3' in config['data_loader']['data_dir'].lower():
    sample_start = 2.5 # full_start / sample_rate
    sample_rate = 40 # sample rate for sample3
    sequence_start = sequence_use * sample_rate 
else:
    extrapolate = False

# Code to extrapolate the dataset if tested on a dataset that is a subset of the full data. 
def extrap(extrapolate, file):
    if extrapolate:
        df_full = pd.read_csv('./test_full.csv')
        df_sample_submission = pd.read_csv(file)
        print('Full file is: ', len(df_full))
        print('Sample file is: ', len(df_sample_submission))
        
        # Read in sample submission data and extrapolate the full submission data
        full_submission = {
            'canSteering': [], 
            'canSpeed': []
        }
        chapter_sizes = df_full.groupby(['chapter']).size()
        
        # start from the 0 row in the sample submission
        chapter_start = 0
        for chapter in chapter_sizes.index:
            print('Extrapolating:', chapter-1, chapter_start, len(full_submission['canSteering']))
            tmp = df_full[df_full['chapter'] == chapter]
            for img in range(len(tmp)):
                img_i = int(tmp.iloc[img, 0].split('/')[-1].replace('.jpg','').replace('img',''))
                if img_i > full_start:
                    if img_i < (sequence_use+1) * sample_rate:
                        i_sample = chapter_start
                        canSteering = df_sample_submission.at[i_sample, 'canSteering']
                        canSpeed = df_sample_submission.at[i_sample, 'canSpeed']
                    elif img_i % sample_rate == 0:
                        i_sample = chapter_start + np.int(np.floor(img_i / sample_rate)) - sequence_use - 1
                        canSteering = df_sample_submission.at[i_sample, 'canSteering']
                        canSpeed = df_sample_submission.at[i_sample, 'canSpeed']
                    elif img_i / sample_rate > np.int(np.floor(chapter_sizes.loc[chapter] / sample_rate)):
                        #if chapter == 106:
                            #print(img_i, chapter_start, chapter_sizes.loc[chapter], np.int(np.floor(chapter_sizes.loc[chapter] / sample_rate)), sequence_use)
                        i_sample = chapter_start + np.int(np.floor(chapter_sizes.loc[chapter] / sample_rate)) - sequence_use - 1
                        canSteering = df_sample_submission.at[i_sample, 'canSteering']
                        canSpeed = df_sample_submission.at[i_sample, 'canSpeed']
                    else:
                        i_low = chapter_start + np.int(np.floor(img_i / sample_rate)) - sequence_use - 1
                        i_high = chapter_start + np.int(np.ceil(img_i / sample_rate)) - sequence_use - 1
                        #print(img_i, chapter_start, i_low, i_high)
                        x = [np.int(np.floor(img_i / sample_rate)) * sample_rate,np.int(np.ceil(img_i / sample_rate))* sample_rate]
                        y = [df_sample_submission.at[i_low, 'canSteering'], df_sample_submission.at[i_high, 'canSteering']]
                        f = interpolate.interp1d(x, y)
                        canSteering = f(img_i)
                        y = [df_sample_submission.at[i_low, 'canSpeed'], df_sample_submission.at[i_high, 'canSpeed']]
                        f = interpolate.interp1d(x, y)
                        canSpeed = f(img_i)
                    full_submission['canSteering'].append(canSteering)
                    full_submission['canSpeed'].append(canSpeed)
            chapter_start += np.int(np.floor(chapter_sizes.loc[chapter] / sample_rate)) - sequence_use


        df_full_submission = pd.DataFrame.from_dict(full_submission)
        df_full_submission.to_csv('./full_submission'+ version_name + '.csv', index=False)

#Create the train setup and then loop over it a set number of epochs
model = SomeDrivingModel()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Load the model if the model had to restart from a saved version
if os.path.isfile(path_name): #modified from https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3
  print("=> loading checkpoint '{}'".format(path_name))
  checkpoint = torch.load(path_name)
  start_epoch = checkpoint['epoch']
  start_batch_idx = checkpoint['batch_idx']
  model.load_state_dict(checkpoint['state_dict'])
  model.cuda()
  optimizer.load_state_dict(checkpoint['optimizer'])
  print("==> loaded checkpoint '{}'".format(path_name))
else:
  start_epoch = 0
  start_batch_idx = -1
  model.cuda()
  print("=> no checkpoint found at '{}'".format(path_name))
  print(datetime.datetime.now())

criterion = nn.MSELoss()

for epoch in range(start_epoch, max_epoch):
    model.train()
    running_loss = 0.0
    it = enumerate(train_loader)
    for batch_idx, (data, target) in it:
      if start_batch_idx > -1 and epoch == start_epoch and batch_idx % detail_size == 0 and batch_idx <= start_batch_idx:
          print('Checking: ', batch_idx)
          
      #Add values to the GPU
      if batch_idx > start_batch_idx or epoch > start_epoch or start_batch_idx >= min_start_batch_idx:
        for k1 in data:
          for k2 in data[k1]:
            data[k1][k2] = data[k1][k2].cuda()
        for k1 in target:
            target[k1] = target[k1].cuda()
            
        #Forward
        optimizer.zero_grad()
        prediction = model(data)
        
        # Optimize on steering and speed at the same time
        loss = criterion(prediction['canSpeed'], target['canSpeed']) + criterion(prediction['canSteering'], target['canSteering'])
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if batch_idx % detail_size == 0:  
            state = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }
            torch.save(state, path_name)
            print('[epoch: %d, batch:  %5d] loss: %.5f' %
                  (epoch, batch_idx, running_loss / detail_size), datetime.datetime.now())   
            running_loss = 0.0
    
    state = {
    'epoch': epoch + 1,
    'batch_idx': -1,
    'state_dict': model.state_dict(),
    'optimizer' : optimizer.state_dict(),
    }
    torch.save(state, path_name)
    if (epoch in output_states) or (epoch == max_epoch - 1):
        file = './submission '+ version_name +' - Epoch '+ str(epoch) + '.csv'
        create_submission(file)
        extrap(extrapolate, file)



#To run on the validation data
normalize_targets = config['target']['normalize']
target_mean = config['target']['mean']
target_std = config['target']['std']
model.cuda()
model.eval()
with torch.no_grad():
    if run_val:
        res_canSteering = []
        res_canSpeed = []
        for batch_idx, (data, target) in enumerate(validation_loader):
            for k1 in data:
                for k2 in data[k1]:
                    data[k1][k2] = data[k1][k2].cuda()
            prediction = model(data)

            if normalize_targets:
                res_canSpeed.extend((prediction['canSpeed'].cpu() - target['canSpeed'])*target_std['canSpeed'])
                res_canSteering.extend((prediction['canSteering'].cpu() - target['canSteering'])*target_std['canSteering'])
            else:
                res_canSpeed.extend((prediction['canSpeed'].cpu() - target['canSpeed']))
                res_canSteering.extend((prediction['canSteering'].cpu() - target['canSteering']))

            if batch_idx % detail_size == 0:  
                print(batch_idx, ': ', datetime.datetime.now())

        res_canSpeed2 = np.square(res_canSpeed)
        res_canSteering2 = np.square(res_canSteering)

        print('MSE Steering: ', res_canSteering2.mean(), 'MSE Speed:', res_canSpeed2.mean())
        print('MSE Combined:', res_canSpeed2.mean() + res_canSteering2.mean())