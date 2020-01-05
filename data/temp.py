import cv2
import os
import sys
sys.path.append('..')
from config import params
import subprocess

def save_img():
    video_path = r'/data2/data/video_data/UCF-101/video/'
    videos = os.listdir(video_path)
    # params['save_path'] = '/data6/video_data/Thumos14/validation_pic/'
    save_path = '/data2/data/video_data/UCF-101/video_img/'
    for video_name in videos:
        temp = os.path.join('/data2/data/video_data/UCF-101/video/', video_name)
        video_temp = os.listdir(temp)
        for video in video_temp:
            file_name = video.split('.')[0]
            folder_name = save_path + video_name + '/' + file_name
            os.makedirs(folder_name, mode=0o777, exist_ok=True)

            vc = cv2.VideoCapture(temp + '/' + video)
            frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
            print('frame_count； ',frame_count)
            c = 0
            rval = vc.isOpened()

            while rval:
                c = c + 1
                rval, frame = vc.read()
                pic_path = folder_name + '/'
                if rval:
                    cv2.imwrite(pic_path + file_name + '_' + str(c) + '.jpg', frame)
                    cv2.waitKey(1)
                else:
                    break
            vc.release()
            print('folder_name', folder_name)
            print('success')



def test_temp():
    video_path =  r'/data2/data/video_data/UCF-101/video/'
    file_name = video_path + 'Basketball/v_Basketball_g16_c02.avi'
    capture = cv2.VideoCapture(file_name)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print('frame_count； ', frame_count)
    folder_name = '/data2/data/video_data/UCF-101/video_i/Basketball/'

    os.makedirs(folder_name, mode=0o777, exist_ok=True)

    c = 0
    rval = capture.isOpened()

    while rval:
        c = c + 1
        rval, frame = capture.read()
        pic_path = folder_name + '/'
        if rval:
            cv2.imwrite(folder_name + str(c) + '.jpg', frame)
            cv2.waitKey(1)
        else:
            break
        print('c:',c)
    capture.release()




def get_data():
    path = os.path.join(params['root'], 'video_window_shot.txt')
    file = open(path, 'r', errors="ignore")
    data = []
    while True:
        str = file.readline()
        if not str:
            break
        # self.len = self.len + 1
        str_video_name = str[:str.find(' ')]  # e.g. video_validation_0000003.mp4
        str_shot_label = str[str.find('[') + 1: str.find(']')] + ','  # e.g.  2, 18, 98, 300,499, 1999,
        temp = []
        while str_shot_label.find(',') != -1:
            label = int(str_shot_label[:str_shot_label.find(',')])
            str_shot_label = str_shot_label[str_shot_label.find(',') + 1:]
            # print(str_video_name, label)
            temp.append(label)
        temp2 = (str_video_name, temp)
        data.append(temp2)
    print(len(data))
    return data


if __name__ == '__main__':
    root = '/data1/fb/workspace/SSL_untrimmed_video_version/data/shots_txt/'
    path = os.path.join(root, 'one_shot_3_label.txt')
    file = open(path, 'r', errors="ignore")
    data = []
    while True:
        str_ = file.readline()
        if not str_:
            break
        str_video_name = str_.split(' ')[0]  # e.g. video_validation_0000003.mp4
        str_shot_label = [int(str_.split(' ')[1]), int(str_.split(' ')[2]), int(str_.split(' ')[-1])]
        temp = (str_video_name, str_shot_label)
        data.append(temp)
    file.close()

    count = 1
    temp = []
    temp.append((data[0][0], data[0][1], 1))
    for i in range(1, len(data)):
        if data[i][0] == data[i - 1][0]:
            count = count + 1
            temp.append((data[i][0], data[i][1], count))
        else:
            count = 1
            temp.append((data[i][0], data[i][1], count))

    video_name_list = []
    video_root = '/data2/data/video_data/Thumos14/'


    for i in range(0, len(temp)):
        video_name_list.append(temp[i][0])

    x = 1
    for i in range(1,1011):
        video_name = 'video_validation_'+str(i).zfill(7)+'.mp4'
        if video_name not in video_name_list:
            print(video_name)
            x = x+1

            video_in = os.path.join(video_root, 'validation', video_name)
            capture = cv2.VideoCapture(video_in)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            frame = []
            for i in range(0, frame_count//300):
                frame.append(1+300*i)
                frame.append(299+1+300*i)

            print(frame)
            print(frame_count)
            for index in range(0, len(frame), 2):
                if index>=len(frame)-1:
                    break
                print('start:', frame[index])
                print('end:', frame[index+1])
                video_folder = video_name.split('.')[0]
                folder_path = os.path.join(video_root, 'validation_cut', video_folder)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path, mode=0o777, exist_ok=True)

                video_in = os.path.join(video_root,'validation',video_name)
                video_out = os.path.join(video_root,
                                         'validation_cut',
                                         video_folder,
                                         video_folder+'_'+str(index//2+1)+'.mp4')
                start = frame[index]/30
                end = frame[index+1]/30
                cmd = 'ffmpeg -y -i {} -t {} -ss {} -strict -2 {}'.format(video_in,
                                                                  end - start,
                                                                  start,
                                                                  video_out)
                subprocess.call(cmd, shell=True)


