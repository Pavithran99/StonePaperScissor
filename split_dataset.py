import os
import shutil
import random

path = '/home/pavithran/StonePaperScissor/dataset'
des_path = '/home/pavithran/StonePaperScissor/dataset_split'
c_name_l = os.listdir(path)
os.mkdir(os.path.join(des_path,'train'))
os.mkdir(os.path.join(des_path,'test'))
os.mkdir(os.path.join(des_path,'val'))
print('done')
for c_name in c_name_l:
    f_path_src = os.path.join(path,c_name)
    f_path_des_train = os.path.join(des_path,'train',c_name)
    f_path_des_test = os.path.join(des_path, 'test',c_name)
    f_path_des_val = os.path.join(des_path,'val',c_name)
    if not os.path.exists(f_path_des_train):
        os.mkdir(f_path_des_train)
    if not os.path.exists(f_path_des_test):
        os.mkdir(f_path_des_test)
    if not os.path.exists(f_path_des_val):
        os.mkdir(f_path_des_val)
    im_name_l = os.listdir(f_path_src)
    train = (len(im_name_l) / 100 ) * 70
    test = (len(im_name_l) / 100 ) * 20
    val = (len(im_name_l) / 100 ) * 10
    # print('Total images {} Train {} Test {} Validation {}'.format(len(im_name_l),train,test,val))
    random.shuffle(im_name_l)
    count1 = 0
    count2 = 0
    count3 = 0
    for im_name in im_name_l[:int(val)]:
        im_path_src = os.path.join(f_path_src,im_name)
        im_path_des = os.path.join(f_path_des_val,im_name)
        shutil.copy(im_path_src,im_path_des)
        count1 = count1 + 1
    for im_name in im_name_l[int(val):int(val) + int(test)]:
        im_path_src = os.path.join(f_path_src,im_name)
        im_path_des = os.path.join(f_path_des_test,im_name)
        shutil.copy(im_path_src,im_path_des)
        count2 = count2 + 1
    for im_name in im_name_l[int(val) + int(test):]:
        im_path_src = os.path.join(f_path_src,im_name)
        im_path_des = os.path.join(f_path_des_train,im_name)
        shutil.copy(im_path_src,im_path_des)
        count3 = count3 + 1
    print("{} Test: {} Train: {} Validation: {}".format(c_name,count2,count3,count1))
