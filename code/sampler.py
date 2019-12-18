import cv2
import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence

from math import ceil
from utils import  create_segmented_seg_img




class Sampler(Sequence):
    def __init__(self, batch_size, data, data_type):
        self.batch_size=batch_size
        self.data_type=data_type
        self.label_list=data
        self.indexes=list(range(len(self.label_list)))
        self.iterations = ceil(len(self.label_list) / self.batch_size)

    def __getitem__(self, idx):
        batch=[]
        outputs=[]
        frame_path='init'
        pool=self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        for position in pool:
            img=cv2.imread("/home/ubuntu/martin/kaggle/data/train_v2/"+self.label_list[position][0])
            seg_img=create_segmented_seg_img(self.label_list[position][1])
            batch.append(img)
            outputs.append(seg_img)
            cv2.imwrite("/home/ubuntu/martin/kaggle/images/seg/{}".format(self.label_list[position][0]), seg_img)
            cv2.imwrite("/home/ubuntu/martin/kaggle/images/non_seg/{}".format(self.label_list[position][0]), img)
        batch=np.array(batch)
        outputs=np.array(outputs)
        return(batch, outputs)

    def __len__(self):
        return self.iterations
    
    def __hash__(self):
        return hash(repr(self))




def test_generator(test_list):
    """
    Generator specifically for the test list of images' paths
    """
    for i in len(test_list):
        img=cv2.imread(test_list[i])
        yield (img)



# class Dataloader():
#     def __init__(self, batch_size, data_type):
#         self.batch_size=batch_size
#         self.data_type=data_type
#         self.label_list=create_list()
#         self.label_list=self.label_list[:60]
#         self.indexes=list(range(len(self.label_list)))
#         #self.iterations = ceil(len(self.label_list) / self.batch_size)


#     def yielder(self):
#         idx=0
#         while True:
#             batch=[]
#             outputs=[]
#             frame_path='init'
#             pool=self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
#             for position in pool:
#                 img=cv2.imread("/home/ubuntu/Stages/kaggle/airbus-ship-detection/train_v2/"+self.label_list[position][0])
#                 seg_img=create_segmented_seg_img(self.label_list[position][1])
#                 batch.append(img)
#                 outputs.append(seg_img)
#                 cv2.imwrite("/home/ubuntu/images/seg/{}".format(self.label_list[position][0]), seg_img)
#                 cv2.imwrite("/home/ubuntu/images/non_seg/{}".format(self.label_list[position][0]), img)
#             batch=np.array(batch)
#             outputs=np.array(outputs)

#             yield(batch, outputs)
#             idx+=1
