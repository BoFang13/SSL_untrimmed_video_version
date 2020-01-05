import os
import cv2
import sys
sys.path.append('..')


root = '/data2/data/video_data/Thumos14/'

def get_video_name(video_seq):
    # load video based on index(e.g. index=1, 10, 100, 1000)
    video_name = 'video_validation_' + str(video_seq).zfill(7) + '.mp4'
    return video_name


def read_window_shot(video_seq_number):
    label = []
    video_name = get_video_name(video_seq_number)
    # print('video_name: ', video_name)
    video_name = video_name[:video_name.find('.')]  # remove  '.mp4'
    video_name = video_name + '_shot.txt'
    path = os.path.join('/data1/fb/workspace/SSL_untrimmed_video_version/data/shots/', video_name)
    # print('window_shot_path: ', path)

    file = open(path, 'r', errors="ignore")
    while True:
        str = file.readline()
        if not str:
            break
        label.append(int(str))

    label_final = []
    fname = os.path.join('/data2/data/video_data/Thumos14/validation/', get_video_name(video_seq_number))
    capture = cv2.VideoCapture(fname)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    label.insert(0,1)
    label.append(frame_count)
    for res in range(1, len(label)-1):
        if label[res]-label[res-1]>32 and label[res+1]-label[res]>32:
            label_final.append(label[res])

    return label_final, len(label_final)



def write_1_shot_3_label():
    path = '/data1/fb/workspace/SSL_untrimmed_video_version' \
           '/data/shots_txt/one_shot_3_label.txt'
    f = open(path, 'a')
    temp = 0
    for video_num in range(1, 1011):
        video_name = get_video_name(video_num)
        video_shot_label, number = read_window_shot(video_num)

        for i in range(1, len(video_shot_label)-1):
            f.writelines(video_name + ' '+str(video_shot_label[i-1])+
                         ' '+str(video_shot_label[i])+
                         ' '+str(video_shot_label[i+1])+ '\n')
    f.close()

#
# def write_zero_shot_2_label_():
#     path = os.path.join(root, 'one_shot_8_label.txt')
#     file = open(path, 'r', errors="ignore")
#     data = []
#     txt = '/data1/fb/workspace/SSL_untrimmed_video_version/' \
#           'data/shots_txt/zero_shot_3_label.txt'
#     f = open(txt, 'a')
#     while True:
#         line = file.readline()
#         if not line:
#             break
#         str_video_name = line[:line.find(' ')]  # e.g. video_validation_0000003.mp4
#         str_shot_label = int(line[line.find(' ') + 1:])
#         data.append(str_video_name)
#     # print(data)
#     for i in range(1, 1011):
#         video_name = get_video_name(i)
#         fname = os.path.join(root, 'validation', video_name)
#         capture = cv2.VideoCapture(fname)
#         frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # frame_count
#         print('video_name:',video_name,' frame_count:',frame_count)
#         for i in range(0, frame_count//300):
#             f.writelines(video_name + ' [' + str(1+300*i)+','+str(300*(i+1))+']\n')
#     f.close()



def write_four_shot_1_label():
    path = '/data1/fb/workspace/SSL_untrimmed_video_version' \
           '/data/shots_txt/four_shot_1_label.txt'
    f = open(path, 'a')
    for video_num in range(1,1011):

        video_name = get_video_name(video_num)
        video_shot_label, len_shots = read_window_shot(video_num)

        fname = os.path.join(root, 'validation', video_name)
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # frame_count

        for j in range(0, len(video_shot_label) // 4):
            list_ = []
            for x in range(4):
                list_.append(video_shot_label[x + 4 * j])

            if 4*(j+1) <= len(video_shot_label)-1:
                list_.append(video_shot_label[4*(j+1)])
            else:
                list_.append(frame_count)

            f.writelines(video_name + ' ' + str(list_) + '\n')

    f.close()


def write_top1_top2_data():
    path = '/data1/fb/workspace/SSL_untrimmed_video_version' \
           '/data/shots_txt/top12_shots.txt'
    file = open(path, 'a')

    for i in range(1, 1011):
        video_name = get_video_name(i)
        fname = os.path.join(root, 'validation', video_name)
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        shot_list, len_list = read_window_shot(i)
        shot_list = shot_list[0:-1]  # 删除末尾元素
        if len(shot_list)<=2:
            print(video_name)
            print(frame_count)
            top1_start = 1
            top1_end = frame_count//2
            top2_start = top1_end+1
            top2_end = frame_count
            file.writelines(video_name+' '+str(top1_start)+' '+str(top1_end)+'\n')
            file.writelines(video_name + ' ' + str(top2_start) + ' ' + str(top2_end) + '\n')
        else:
            top1_start, top1_end, top2_start, top2_end = find(frame_count, shot_list)
            file.writelines(video_name + ' ' + str(top1_start) + ' ' + str(top1_end) + '\n')
            file.writelines(video_name + ' ' + str(top2_start) + ' ' + str(top2_end) + '\n')



def find(frame_count, label_list):
    # if len(label_list)<=2:
    #     label_list.insert(0,1)
    #     label_list.append(frame_count)


    temp = []
    for i in range(1, len(label_list)):
        temp.append(label_list[i] - label_list[i - 1])

    top1_start = label_list[temp.index(max(temp))]
    top1_end = label_list[temp.index(max(temp)) + 1]

    temp[temp.index(max(temp))] = -1
    top2_start = label_list[temp.index(max(temp))]
    top2_end = label_list[temp.index(max(temp)) + 1]
    return top1_start, top1_end, top2_start, top2_end




def write_zero_shot_label():
    txt = '/data1/fb/workspace/SSL_untrimmed_video_version/' \
                   'data/shots_txt/zero_shot_3_label.txt'

    f = open(txt, 'a')
    for video_seq in range(1, 1011):
        video_name = get_video_name(video_seq)   # video_validation_0000001.mp4
        shot_list, len_shots = read_window_shot(video_seq)   # [34, 89, 300]
        fname = os.path.join(root, 'validation', video_name)
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # frame_count
        if shot_list==[]:
            for i in range(0, frame_count // 300):
                f.writelines(video_name + ' ' + str(i+1) + '\n')
            continue

        else:
            print('video_name:', video_name)
            print('shot_list:', shot_list)
            for shot in shot_list:
                out = decide(video_name, shot)
                print('shot:', shot, 'out:', out)
                if out != False  and out != None:
                    f.writelines(video_name + ' ' + str(out) + '\n')
    f.close()

def decide(video_name, shot):
    root = '/data1/fb/workspace/SSL_untrimmed_video_version/' \
                'data/shots_txt/'
    path = os.path.join(root, 'one_shot_3_label.txt')
    file = open(path, 'r', errors="ignore")
    data = []
    while True:
        str_ = file.readline()
        if not str_:
            break
        str_video_name = str_.split(' ')[0]  # e.g. video_validation_0000003.mp4
        shot_1 = int(str_.split(' ')[1])
        shot_2 = int(str_.split(' ')[2])
        video_seq = int(str_.split(' ')[-1])
        temp = [str_video_name,shot_1, shot_2, video_seq]

        data.append(temp)
    file.close()
    for i in range(1, len(data)):
        if data[i][3] == -1:
            data[i][3] = data[i-1][3]+1

    for j in range(0,len(data)):
        if data[j][0] == video_name and data[j][1] == shot:
            if data[j][2]-data[j][1] > 300:
                flag = data[j][3]
                return flag
            else:
                return False


def write_top1_top2_label():
    root = '/data1/fb/workspace/SSL_untrimmed_video_version/' \
           'data/shots_txt/'
    path = os.path.join(root, 'top12_shots.txt')
    file = open(path, 'r', errors="ignore")
    data = []
    while True:
        str_ = file.readline()
        if not str_:
            break
        str_video_name = str_.split(' ')[0]  # e.g. video_validation_0000003.mp4
        shot_1 = int(str_.split(' ')[1])
        print(str_video_name)
        print(shot_1)
        video_seq = get_video_seq(str_video_name, shot_1)
        temp = [str_video_name, video_seq]
        print(temp)
        data.append(temp)
    file.close()
    print(data)

    txt = '/data1/fb/workspace/SSL_untrimmed_video_version/' \
           'data/shots_txt/top1_top2_data.txt'
    file_ = open(txt,'a')
    for i in range(0,len(data)):
        file_.writelines(data[i][0]+' '+str(data[i][1])+'\n')
    file_.close()

def get_video_seq(name, shot):
    root = '/data1/fb/workspace/SSL_untrimmed_video_version/' \
           'data/shots_txt/'
    path = os.path.join(root, 'one_shot_3_label.txt')
    file = open(path, 'r', errors="ignore")
    data = []
    while True:
        str_ = file.readline()
        if not str_:
            break
        str_video_name = str_.split(' ')[0]  # e.g. video_validation_0000003.mp4
        shot_1 = int(str_.split(' ')[1])
        shot_2 = int(str_.split(' ')[2])
        video_seq = int(str_.split(' ')[-1])
        temp = [str_video_name, shot_1, shot_2, video_seq]

        data.append(temp)
    file.close()

    for i in range(1, len(data)):
        if data[i][3] == -1:
            data[i][3] = data[i-1][3]+1

    for i in range(0, len(data)):
        if data[i][0] == name and data[i][1] == shot:
            return data[i][3]




if __name__ == '__main__':
    write_top1_top2_label()