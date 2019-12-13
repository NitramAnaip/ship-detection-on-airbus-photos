import numpy as np
import csv


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