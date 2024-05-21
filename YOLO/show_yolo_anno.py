#-*-coding:utf-8-*-
# date:2019-08
# Author: Eric.Lee
# function: show yolo datasets anno

import cv2
import os
import numpy as np

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # path=r'D:\Python\dataset\RHD\test\bbox.txt'
    path=r'D:\Python\dataset\Senz3d\data\bbox.txt'
    path_voc_names = './cfg/hand.names'

    with open(path_voc_names, 'r') as f:
        label_map = f.readlines()

    for i in range(len(label_map)):
        label_map[i] = label_map[i].strip()
        print(i,') ',label_map[i].strip())

    with open(path, 'r') as file:
        img_files = file.read().splitlines()
        img_files = list(filter(lambda x: len(x) > 0, img_files))

    # label_files = [
    #     x.replace('images', 'labels').replace("JPEGImages", 'labels').replace('.bmp', '.txt').replace('.jpg', '.txt').replace('.png', '.txt')
    #     for x in img_files]

    # print('img_files   : ',img_files[1])
    # print('label_files : ',label_files[1])

    for i in range(len(img_files)):
        # print(img_files[i])
        label, img_path, box_x, box_y, box_w, box_h = img_files[i].split()
        img_path = os.path.join(os.path.dirname(path), img_path)
        if not os.path.isfile(img_path):
            print('not found ', img_path)
            continue

        img = cv2.imread(img_path)
        w = img.shape[1]
        h = img.shape[0]

        # label_path = label_files[i]
        # print(i,label_path)
        # if os.path.isfile(label_path):
        #     with open(label_path, 'r') as file:
        #         lines = file.read().splitlines()
        #     x = np.array([x.split() for x in lines], dtype=np.float32)
        
        x = [(label, box_x, box_y, box_w, box_h)]
        for k in range(len(x)):
            anno = x[k]
            label = int(anno[0])
            x1 = int((float(anno[1])-float(anno[3])/2)*w)
            y1 = int((float(anno[2])-float(anno[4])/2)*h)

            x2 = int((float(anno[1])+float(anno[3])/2)*w)
            y2 = int((float(anno[2])+float(anno[4])/2)*h)

            cv2.rectangle(img, (x1,y1), (x2,y2), (255,30,30), 2)

            cv2.putText(img, ("%s" %(str(label_map[label]))), (x1,y1),\
            cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 55), 6)
            cv2.putText(img, ("%s" %(str(label_map[label]))), (x1,y1),\
            cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 55, 255), 2)
            # cv2.circle(img, (x1,y1), 4, (0,255,225), 6)

        cv2.namedWindow('image',0)
        cv2.imshow('image',img)
        while cv2.waitKey(1) != 13:
            continue
        if cv2.waitKey(1) == 20:
            break
    cv2.destroyAllWindows()
