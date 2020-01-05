"""THUMOS14 DataLoader"""

import os
import cv2
import random
import sys
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
import pandas as pd
sys.path.append('..')
from config import params
import math
import operator
from data.save_shot import read_window_shot

envs = os.environ


class UntrimmedVideoDataset(data.Dataset):
    def __init__(self, root, mode="train"):
        self.root = root
        self.mode = mode
        self.label = []
        self.videos = []

        self.toPIL = transforms.ToPILImage()
        self.transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.RandomCrop(112),
            transforms.ToTensor()
        ])


        self.data0 = self.get_0_shot_data()

        self.data = self.data0

        self.len0 = len(self.data0)

    def __getitem__(self, index):
        if self.mode == "train":

            clip, label = self.get_0_shot_3_label_frames(self.data[index][0], self.data[index][1])
            # if index < self.len1:
            #     x = random.random()
            #     if x <= 0.375:
            #         # 0,1,2
            #         clip, label = self.get_1_shot_3_label_frames(video_name, label_id)
            #     elif x > 0.375:
            #         # 3,4,5,6,7
            #         clip, label = self.get_1_shot_5_label_frames(video_name, label_id)
            # elif index >= self.len1 and index < self.len1+self.len0:
            #     # 9,10,11
            #     clip, label = self.get_0_shot_3_label_frames(video_name, label_id)
            # else:
            #     # 8
            #     clip, label = self.get_4_shot_1_label_frames(video_name, label_id)

            video_clips = self.crop(clip)
            return video_clips, label


    def __len__(self):
        len_all = self.len0
        print('len_all: ', len_all)
        return len_all

    # fname: video_validation_0000001/video_validation_0000001_23.mp4
    def loadcvvideo(self, fname, count_need=16):
        self.root = '/data2/data/video_data/Thumos14/'
        fname_ = os.path.join(self.root, 'validation_cut', fname)
        capture = cv2.VideoCapture(fname_)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if count_need == 0:
            count_need = frame_count
        start = np.random.randint(0, frame_count - count_need + 1)

        buffer = []
        count = 0
        retaining = True
        sample_count = 0


        while (sample_count < count_need and retaining):
            retaining, frame = capture.read()

            if retaining is False:
                count += 1

                break
            if count >= start:
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame = cv2.resize(frame, (171, 128))
                buffer.append(frame)
                sample_count = sample_count + 1
            count += 1

        capture.release()
        while len(buffer)<16 :
            buffer, retaining = self.loadcvvideo(fname, count_need=16)

        return buffer, retaining


    def load_2_videos(self, fname, count_need=16):
        self.root = '/data2/data/video_data/Thumos14/'
        fname_ = os.path.join(self.root, 'validation_cut', fname)
        capture = cv2.VideoCapture(fname_)
        capture2 = cv2.VideoCapture(fname_)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count_need == 0:
            count_need = frame_count

        start1 = random.randint(0, frame_count - 2 * count_need)
        start2 = random.randint(start1 + 16, frame_count - count_need)
        # print(fname)
        # print('start1:', start1, 'start2:', start2)

        buffer1, buffer2 = [], []
        count1, count2 = 0, 0
        retaining = True
        sample_count1, sample_count2 = 0, 0

        while (sample_count1 < count_need and retaining):
            retaining, frame1 = capture.read()

            if retaining is False:
                count1 += 1

                break
            if count1 >= start1:
                buffer1.append(frame1)
                sample_count1 = sample_count1 + 1
            count1 += 1

        while (sample_count2 < count_need and retaining):
            retaining, frame2 = capture2.read()

            if retaining is False:
                count2 += 1

                break
            if count2 >= start2:
                buffer2.append(frame2)
                sample_count2 = sample_count2 + 1
            count2 += 1

        capture.release()
        capture2.release()
        while len(buffer1)<16 or len(buffer2)<16:
            buffer1, buffer2, retaining = self.load_2_videos(fname_, count_need=16)
        return buffer1, buffer2, retaining


    def crop(self, frames):
        video_clips = []
        seed = random.random()
        for frame in frames:

            frame = self.toPIL(frame)
            frame = self.transforms(frame)

            video_clips.append(frame)

        return torch.stack(video_clips).permute(1,0,2,3)


    def get_video_name(self, video_id):
        # load video based on index(e.g. index=1, 10, 100, 1000)
        video_name = 'video_validation_' + str(video_id).zfill(7) + '.mp4'
        return video_name


    def get_0_shot_data(self):
        self.root = '/data1/fb/workspace/SSL_untrimmed_video_version/' \
                    'data/shots_txt/'
        path = os.path.join(self.root, 'zero_shot_3_label.txt')
        file = open(path, 'r', errors="ignore")
        data = []
        while True:
            line = file.readline()
            if not line:
                break
            video_name = line.split(' ')[0]
            video_seq = int(line.split(' ')[1])
            data.append((video_name, video_seq))
        file.close()

        return data


    def get_1_shot_data(self):     # read txt
        self.root = '/data1/fb/workspace/SSL_untrimmed_video_version/' \
                    'data/shots_txt/'
        path = os.path.join(self.root, 'one_shot_3_label.txt')
        file = open(path, 'r', errors="ignore")
        data = []
        while True:
            str_ = file.readline()
            if not str_:
                break
            str_video_name = str_.split(' ')[0]  # e.g. video_validation_0000003.mp4
            video_seq = int(str_.split(' ')[-1])
            temp = (str_video_name, video_seq)

            data.append(temp)
        file.close()
        data_final = []
        for i in range(0,len(data)-1):
            if data[i][1]==-1:
                continue
            if data[i][1]==1 and data[i+1][1]==1:
                continue
            data_final.append((data[i][0], data[i][1]))
        data_final = data_final[:400]

        return data_final


    def get_4_shot_data(self):
        self.root = '/data1/fb/workspace/SSL_untrimmed_video_version/' \
                    'data/shots_txt/'
        path = os.path.join(self.root, 'one_shot_3_label.txt')
        file = open(path, 'r', errors="ignore")
        data = []
        while True:
            str = file.readline()
            if not str:
                break
            str_video_name = str.split(' ')[0]
            str_shot_label = int(str.split(' ')[-1])
            data.append((str_video_name, str_shot_label))

        file.close()
        data_final = []
        for i in range(0, len(data)-1):
            if data[i][1]==1 and data[i+1][1]==1:
                continue
            video_name, video_seq = data[i][0], data[i][1]
            if video_seq%3==1 and data[i+1][1]!=-1:
                data_final.append((video_name, video_seq))

        # 每个label平均只含有2000条数据
        # data = random.sample(data, 2000)
        return data_final


    def get_top1_top2_data(self):
        self.root = '/data1/fb/workspace/SSL_untrimmed_video_version/' \
                    'data/shots_txt/'
        path = os.path.join(self.root, 'top1_top2_data.txt')
        file = open(path, 'r', errors='ignore')
        data = []
        while True:
            line = file.readline()
            if not line:
                break
            video_name = line.split(' ')[0]
            video_seq = line.split(' ')[1]
            data.append([video_name, video_seq])


        for i  in range(0, len(data)):
            if data[i][1] == 'None\n':
                f_name = data[i][0].split('.')[0]

                f_name = os.path.join('/data2/data/video_data/Thumos14/validation_cut/', f_name)
                video_num = len([lists for lists in os.listdir(f_name) if os.path.isfile(os.path.join(f_name, lists))])
                if video_num == 1:
                    data[i][1] = 1
                    continue
                if i%2 == 0:
                    data[i][1] = random.randint(1, video_num//2)
                elif i%2 == 1:
                    data[i][1] = random.randint(video_num//2+1, video_num)
            else:
                data[i][1] = int(data[i][1])

        for i in range(0,len(data)):
            other_top = np.random.randint(0, 2020)
            data[i].append(data[other_top][0])
            data[i].append(data[other_top][1])

        return data



    def random_flip(self, buffer):
        if np.random.random() <0.5:
            for i,frame in enumerate(buffer):

                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer



    def get_1_shot_5_label_frames(self, video_name, video_seq):   # A1A2B1B2  ->  A1B1A2B2
        video_name = video_name[:video_name.find('.mp4')]
        fname = video_name+'/'+video_name+'_'+str(video_seq)+'.mp4'
        A1, A2, retaining = self.load_2_videos(fname, count_need=16)
        next_fname = video_name+'/'+video_name+'_'+str(video_seq+1)+'.mp4'
        B1, B2, retrain_b = self.load_2_videos(next_fname, count_need=16)

        clip = []
        label = random.randint(0,4)
        if label==0:
            clip = A1+A2+B1
        elif label==1:
            clip = A1+B1+B2
        elif label==2:
            clip = A1+B1+A2
        elif label==3:
            clip = A1+B1+B1[::-1]
        elif label==4:
            clip = A1+B1+A2[::-1]
        return clip, label



    def get_4_shot_1_label_frames(self, video_name, video_seq):
        video_name = video_name[:video_name.find('.mp4')]
        fname_1 = video_name+'/'+video_name+'_'+str(video_seq)+'.mp4'
        fname_2 = video_name + '/' + video_name + '_' + str(video_seq+1) + '.mp4'
        fname_3 = video_name + '/' + video_name + '_' + str(video_seq+2) + '.mp4'
        A, retrainA = self.loadcvvideo(fname_1, count_need=16)
        B, retrainB = self.loadcvvideo(fname_2, count_need=16)
        C, retrainC = self.loadcvvideo(fname_3, count_need=16)
        clip = A+B+C
        return clip, 5



    def get_0_shot_3_label_frames(self, video_name, video_seq):
        video_name = video_name[:video_name.find('.mp4')]
        fname = video_name+'/'+video_name+'_'+str(video_seq)+'.mp4'
        A1, A2, retaining = self.load_2_videos(fname, count_need=16)
        label = random.randint(6,8)
        clip = []
        if label == 6:
            clip = A1+A2+A2[::-1]
        elif label== 7:
            clip = A1+A2+A1[::-1]
        return clip, label



    def get_top1_top2_frames(self, video_name1, video_seq1, video_name2, video_seq2):
        video_name1 = video_name1[:video_name1.find('.mp4')]
        video_name2 = video_name2[:video_name2.find('.mp4')]
        fname1 = video_name1+'/'+video_name1+'_'+str(video_seq1)+'.mp4'
        fname2 = video_name2+'/'+video_name2+'_'+str(video_seq2)+'.mp4'
        top1, retrain1 = self.loadcvvideo(fname1, count_need=16)
        top2, retrain2 = self.loadcvvideo(fname2, count_need=16)
        clip = top1+top1+top2

        return clip, 9


if __name__ == '__main__':
    params['root'] = '/data3/video_data/Thumos14/'
    train_data = UntrimmedVideoDataset(params['root'], mode="train")
    data = DataLoader(train_data, batch_size=8, num_workers=1, shuffle=True)

    for i, (clip, label) in enumerate(data):
        print('-'*40)
        print(i, ':')
        print('clip.size: ', clip.shape)
        print('label: ', label)
        if i == 5:
            break
