params = dict()

params['root'] = '/data2/data/video_data/Thumos14/'
params['window_shot_location'] = '/data2/fb/workspace/SSL_untrimmed_video/data/shots/'
params['val_data_root'] = '/data2/data/video_data/Thumos14/validation/'


params['data'] = 'Thumos14'

params['epoch_num'] = 300
params['learning_rate'] = 0.001
params['step'] = 10
params['momentum'] = 0.9
params['weight_decay'] = 0.0005
params['display'] = 10