import cv2
import csv
import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence

from math import ceil

def  create_list():
    path="/home/ubuntu/Stages/kaggle/airbus-ship-detection/train_ship_segmentations_v2.csv"
    label_list = []
    with open(path) as csvfile:
        data = csv.reader(csvfile, delimiter=",")
        for line in data:
            element = [line[0], line[1].split()]
            label_list.append(element)
    label_list.pop(0)  # simply to remove the first line which isn't data
    for elem in label_list:
        for i in range(len(elem[1])):
            elem[1][i]=int(elem[1][i])
    return label_list


def create_segmented_seg_img(values):
    R=[]
    G=[]
    B=[]
    index=0
    pos_in_seg_img=0
    while index<len(values):
        while pos_in_seg_img<values[index]:
            R.append(0)
            G.append(0)
            B.append(0)
            pos_in_seg_img+=1
        index+=1
        for i in range (values[index]):
            R.append(255)
            G.append(255)
            B.append(255)
            pos_in_seg_img+=1
        index+=1
    for j in range (pos_in_seg_img, 768*768):
        R.append(0)
        G.append(0)
        B.append(0)

    R=np.reshape(R, (768, 768)).T
    G=np.reshape(G, (768, 768)).T
    B=np.reshape(B, (768, 768)).T
    seg_img=np.dstack((R,G,B))
    return seg_img



class Sampler(Sequence):
    def __init__(self, batch_size, data_type):
        self.batch_size=batch_size
        self.data_type=data_type
        self.label_list=create_list()
        self.label_list=self.label_list[:60]
        self.indexes=list(range(len(self.label_list)))
        self.iterations = ceil(len(self.label_list) / self.batch_size)


    def get_item(self, idx):
        batch=[]
        outputs=[]
        frame_path='init'
        pool=self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        for position in pool:
            img=cv2.imread("/home/ubuntu/Stages/kaggle/airbus-ship-detection/train_v2/"+self.label_list[position][0])
            seg_img=create_segmented_seg_img(self.label_list[position][1])
            batch.append(img)
            outputs.append(seg_img)
            cv2.imwrite("/home/ubuntu/images/seg/{}".format(self.label_list[position][1]), seg_img)
            cv2.imwrite("/home/ubuntu/images/non_seg/{}".format(self.label_list[position][1]), img)
        batch=np.array(batch)
        return(batch, outputs)

    def __len__(self):
        return self.iterations
    
    def __hash__(self):
        return hash(repr(self))




class Dataloader():
    def __init__(self, batch_size, data_type):
        self.batch_size=batch_size
        self.data_type=data_type
        self.label_list=create_list()
        self.label_list=self.label_list[:60]
        self.indexes=list(range(len(self.label_list)))
        #self.iterations = ceil(len(self.label_list) / self.batch_size)


    def yielder(self):
        idx=0
        while True:
            batch=[]
            outputs=[]
            frame_path='init'
            pool=self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
            for position in pool:
                img=cv2.imread("/home/ubuntu/Stages/kaggle/airbus-ship-detection/train_v2/"+self.label_list[position][0])
                seg_img=create_segmented_seg_img(self.label_list[position][1])
                batch.append(img)
                outputs.append(seg_img)
                cv2.imwrite("/home/ubuntu/images/seg/{}".format(self.label_list[position][0]), seg_img)
                cv2.imwrite("/home/ubuntu/images/non_seg/{}".format(self.label_list[position][0]), img)
            batch=np.array(batch)
            outputs=np.array(outputs)

            yield(batch, outputs)
            idx+=1
