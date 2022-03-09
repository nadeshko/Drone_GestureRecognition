import os
import sys
import cv2
import random
import numpy as np
import tensorflow as tf
#import dataset_visualization # run dataset visualization


from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# setting seed constant
seed = 25
np.random.seed(seed)
random.seed(seed)
#tf.random.set_seed(seed)

# image processing
h, w = 32, 64
DATASET_DIR = 'UAV-Gesture1'
CLASS_LIST = ['BOXING','CLAPPING','HITTING','JOGGING','KICKING',
              'RUNNING','STABBING','WALKING', 'WAVING']
SEQUENCE = 20

def get_frames(path):
    """
    Function extracts frames of video after it is resized and normalized
    in the form of a list.
    :param path: path of video
    :return: list of frames of every video
    """
    frame_list = []
    vid_reader = cv2.VideoCapture(path)
    frame_count = int(vid_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(int(frame_count/SEQUENCE), 1)
    for i in range(SEQUENCE):
        vid_reader.set(cv2.CAP_PROP_POS_FRAMES, i*frame_skip)
        success, frame = vid_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (h, w))
        normalized_frame = resized_frame / 255
        frame_list.append(normalized_frame)
    vid_reader.release()
    return frame_list

def make_dataset():

    feature   = []
    labels    = []
    vid_paths = []

    # go through all classes in CLASS_LIST
    for idx, class_name in enumerate(CLASS_LIST):
        print(f'Extracting Data from: {class_name}')
        files = os.listdir(os.path.join(DATASET_DIR, class_name))
        for name in files:
            path = os.path.join(DATASET_DIR, class_name, name)
            frames = get_frames(path)
            if len(frames) == SEQUENCE:
                feature.append(frames)
                labels.append(idx)
                vid_paths.append(path)

    features = np.asarray(feature)
    labels = np.array(labels)

    return features, labels, vid_paths

def to_categorical(labels):
    # to-categorical
    label_encoder = LabelEncoder()
    int_encoded = label_encoder.fit_transform(labels)
    one_hot_encoder = OneHotEncoder(sparse=False)
    int_encoded = int_encoded.reshape(len(int_encoded), 1)
    one_hot_encoded = one_hot_encoder.fit_transform(int_encoded)
    return one_hot_encoded

def LRCN_model():
    LRCN = Sequential()

    # model architecture
    LRCN.add()

def main():
    feature, labels, paths = make_dataset()
    one_hot_label = to_categorical(labels)
    feature_train, feature_test, label_train, label_test = train_test_split(
        feature, one_hot_label, test_size=0.3, shuffle=True, random_state=seed)
    
    #choose model
    LRCN_model()

if __name__ == '__main__':
    main()












