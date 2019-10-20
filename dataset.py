import os
from PIL import Image
import pandas as pd
import numpy as np
from random import shuffle
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch 


class SubsetSampler(Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class Drive360Loader(DataLoader):

    def __init__(self, config, phase):

        self.drive360 = Drive360(config, phase)
        batch_size = config['data_loader'][phase]['batch_size']
        sampler = SubsetSampler(self.drive360.indices)
        num_workers = config['data_loader'][phase]['num_workers']

        super().__init__(dataset=self.drive360,
                         batch_size=batch_size,
                         sampler=sampler,
                         num_workers=num_workers
                         )

class Drive360(object):
    ## takes a config json object that specifies training parameters and a
    ## phase (string) to specifiy either 'train', 'test', 'validation'
    def __init__(self, config, phase):
        self.config = config
        self.data_dir = config['data_loader']['data_dir']
        self.csv_name = config['data_loader'][phase]['csv_name']
        self.shuffle = config['data_loader'][phase]['shuffle']
        self.history_number = config['data_loader']['historic']['number']
        self.history_frequency = config['data_loader']['historic']['frequency']
        self.normalize_targets = config['target']['normalize']
        self.target_mean = {}
        target_mean = config['target']['mean']
        for k, v in target_mean.items():
            self.target_mean[k] = np.asarray(v, dtype=np.float32)
        self.target_std = {}
        target_std = config['target']['std']
        for k, v in target_std.items():
            self.target_std[k] = np.asarray(v, dtype=np.float32)

        self.front = self.config['front']
        self.use_data = self.config['data']
        self.right_left = config['multi_camera']['right_left']
        self.rear = config['multi_camera']['rear']

        #### reading in dataframe from csv #####
        self.dataframe = pd.read_csv(os.path.join(self.data_dir, self.csv_name),
                                     dtype={'cameraFront': object,
                                            'cameraRear': object,
                                            'cameraRight': object,
                                            'cameraLeft': object,
                                            'canSpeed': np.float32,
                                            'canSteering': np.float32
                                            })

        # Here we calculate the temporal offset for the starting indices of each chapter. As we cannot cross chapter
        # boundaries but would still like to obtain a temporal sequence of images, we cannot start at index 0 of each chapter
        # but rather at some index i such that the i-max_temporal_history = 0
        # To explain see the diagram below:
        #
        #             chapter 1    chapter 2     chapter 3
        #           |....-*****| |....-*****| |....-*****|
        # indices:   0123456789   0123456789   0123456789
        #
        # where . are ommitted indices and - is the index. This allows using [....] as temporal input.
        #
        # Thus the first sample will consist of images:     [....-]
        # Thus the second sample will consist of images:    [...-*]
        # Thus the third sample will consist of images:     [..-**]
        # Thus the fourth sample will consist of images:    [.-***]
        # Thus the fifth sample will consist of images:     [-****]
        # Thus the sixth sample will consist of images:     [*****]

        self.sequence_length = self.history_number*self.history_frequency
        max_temporal_history = self.sequence_length
        self.indices = self.dataframe.groupby('chapter').apply(lambda x: x.iloc[max_temporal_history:]).index.droplevel(level=0).tolist()

        #### phase specific manipulation #####
        if phase == 'train':
            self.dataframe['canSteering'] = np.clip(self.dataframe['canSteering'], a_max=360, a_min=-360)

            ##### If you want to use binning on angle #####
            ## START ##
            # self.dataframe['bin_canSteering'] = pd.cut(self.dataframe['canSteering'],
            #                                            bins=[-360, -20, 20, 360],
            #                                            labels=['left', 'straight', 'right'])
            # gp = self.dataframe.groupby('bin_canSteering')
            # min_group = min(gp.apply(lambda x: len(x)))
            # bin_indices = gp.apply(lambda x: x.sample(n=min_group)).index.droplevel(level=0).tolist()
            # self.indices = list(set(self.indices) & set(bin_indices))
            ## END ##

        elif phase == 'validation':
            self.dataframe['canSteering'] = np.clip(self.dataframe['canSteering'], a_max=360, a_min=-360)

        elif phase == 'test':
            # IMPORTANT: for the test phase indices will start 10s (100 samples) into each chapter
            # this is to allow challenge participants to experiment with different temporal settings of data input.
            # If challenge participants have a greater temporal length than 10s for each training sample, then they
            # must write a custom function here.
            
            if 'sample1' in self.data_dir.lower() or 'sample2' in self.data_dir.lower() or 'sample3' in self.data_dir.lower():
                pass
            else:
                self.indices = self.dataframe.groupby('chapter').apply(
                    lambda x: x.iloc[100:]).index.droplevel(
                    level=0).tolist()    
            
            if 'canSteering' not in self.dataframe.columns:
                self.dataframe['canSteering'] = [0.0 for _ in range(len(self.dataframe))]
            if 'canSpeed' not in self.dataframe.columns:
                self.dataframe['canSpeed'] = [0.0 for _ in range(len(self.dataframe))]


        if self.normalize_targets and not phase == 'test':
            self.dataframe['canSteering'] = (self.dataframe['canSteering'].values -
                                            self.target_mean['canSteering']) / self.target_std['canSteering']
            self.dataframe['canSpeed'] = (self.dataframe['canSpeed'].values -
                                            self.target_mean['canSpeed']) / self.target_std['canSpeed']

        if self.shuffle:
            shuffle(self.indices)



        print('Phase:', phase, '# of data:', len(self.indices))

        front_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=config['image']['norm']['mean'],
                                     std=config['image']['norm']['std'])
            ]),
            'validation': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=config['image']['norm']['mean'],
                                     std=config['image']['norm']['std'])
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=config['image']['norm']['mean'],
                                     std=config['image']['norm']['std'])
            ])}
        sides_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=config['image']['norm']['mean'],
                                     std=config['image']['norm']['std'])
            ]),
            'validation': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=config['image']['norm']['mean'],
                                     std=config['image']['norm']['std'])
            ]),
            'test': transforms.Compose([
                #transforms.Resize((320, 180)),
                transforms.ToTensor(),
                transforms.Normalize(mean=config['image']['norm']['mean'],
                                     std=config['image']['norm']['std'])
            ])}

        self.imageFront_transform = front_transforms[phase]
        self.imageSides_transform = sides_transforms[phase]


    def __getitem__(self, index):
        inputs = {}
        labels = {}
        end = index - self.sequence_length
        skip = int(-1 * self.history_frequency)
        rows = self.dataframe.iloc[index:end:skip].reset_index(drop=True, inplace=False)
        filter_cols = [col for col in rows if col.startswith('here') and col != 'here' and col != "hereSegmentOthersHeading"]

        if self.front:
            inputs['cameraFront'] = {}
            for row_idx, (_, row) in enumerate(rows.iterrows()):
                inputs['cameraFront'][row_idx] = (self.imageFront_transform(Image.open(self.data_dir + row['cameraFront'])))
        if self.use_data:
            inputs['hereData'] = {}
            for row_idx, (_, row) in enumerate(rows.iterrows()):
                tmp = row[filter_cols].fillna(value=0)
                inputs['hereData'][row_idx] = torch.FloatTensor(tmp.tolist())
        if self.right_left:
            inputs['cameraRight'] = {}
            inputs['cameraLeft'] = {}
            for row_idx, (_, row) in enumerate(rows.iterrows()):
                inputs['cameraRight'][row_idx] = (self.imageSides_transform(Image.open(self.data_dir + row['cameraRight'])))
                inputs['cameraLeft'][row_idx] = (self.imageSides_transform(Image.open(self.data_dir + row['cameraLeft'])))
        if self.rear:
            inputs['cameraRear'] = {}
            for row_idx, (_, row) in enumerate(rows.iterrows()):
                inputs['cameraRear'][row_idx] = (self.imageSides_transform(Image.open(self.data_dir + row['cameraRear'])))
        labels['canSteering'] = self.dataframe['canSteering'].iloc[index]
        labels['canSpeed'] = self.dataframe['canSpeed'].iloc[index]

        return inputs, labels

