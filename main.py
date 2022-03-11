from sklearn.model_selection import train_test_split
from model import models
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
#import dataset_visualization # run dataset visualization

def get_frames(path):
    """
    Function extracts frames of video after it is resized and normalized
    in the form of a list.
    :param path: path of video
    :return: list of frames of every video
    """

    frame_list = []

    # set capture object and count frames
    vid_reader = cv2.VideoCapture(path)
    frame_count = int(vid_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(int(frame_count/SEQUENCE), 1)

    # extracting frames
    for i in range(SEQUENCE):
        # skip a couple of frames (don't need everything)
        vid_reader.set(cv2.CAP_PROP_POS_FRAMES, i*frame_skip)
        success, frame = vid_reader.read()
        # if no more frames, finish
        if not success:
            break
        resized_frame = cv2.resize(frame, (h, w))
        normalized_frame = resized_frame / 255
        frame_list.append(normalized_frame)
    vid_reader.release() # release object

    return frame_list

def make_dataset():
    """
    Function to prepare dataset for training
    :return: extracted features, labels and video paths from video frames
    """

    # initialize required lists, same amt of features and labels
    feature   = []
    labels    = []
    vid_paths = []

    # go through all classes in CLASS_LIST and get features and labels
    for idx, class_name in enumerate(CLASS_LIST):
        print(f'Extracting Data from: {class_name}') # show progress
        files = os.listdir(os.path.join(DATASET_DIR, class_name))
        for name in files:
            path = os.path.join(DATASET_DIR, class_name, name)
            # send path to get_frames from specific videos
            frames = get_frames(path)
            if len(frames) == SEQUENCE:
                feature.append(frames)
                labels.append(idx)
                vid_paths.append(path)

    # convert features and labels into arrays
    features = np.asarray(feature)
    labels = np.array(labels)

    return features, labels, vid_paths

def plot_graph(training_history, metric1, metric2, plot_name):
    """
    Function to plot and show graph of accuracy and loss
    :param training_history: record of training and validation (acc/loss)
    :param metric1: first metric
    :param metric2: second metric
    :param plot_name: name of plot
    """

    # Get metric values using metric names as identifiers.
    metric_1 = training_history.history[metric1]
    metric_2 = training_history.history[metric2]

    # Construct a range object which will be used as x-axis (horizontal plane) of the graph.
    epochs = range(len(metric_1))

    # Plotting
    plt.plot(epochs, metric_1, 'blue', label=metric1)
    plt.plot(epochs, metric_2, 'red', label=metric2)

    # Add title
    plt.title(str(plot_name))

    # Legends
    plt.legend()
    plt.show()

def main():
    """
    main function that trains and evaluates model to get accuracy and loss of model
    :return: ---
    """

    # initiate required utilities
    from utils import Utils
    utils = Utils()

    # prepare dataset
    feature, labels, paths = make_dataset()
    one_hot_label = utils.to_categorical(labels) # convert labels to one_hot format
    feature_train, feature_test, label_train, label_test = train_test_split(
        feature, one_hot_label, test_size=0.25, shuffle=True, random_state=seed)

    # initiate model class
    model = models(seed)
    # choose model
    LRCN_model = model.create_LRCN(SEQUENCE, h, w, CLASS_LIST)
    LRCN_history, test_loss, test_acc = model.compile_model(
        LRCN_model, epochs, feature_train, feature_test, label_train, label_test)

    # saving model
    file_name = f'LRCN_model__Loss_{test_loss}__Acc_{test_acc}'
    LRCN_model.save(file_name)

    # plotting loss and accuracy
    plot_graph(LRCN_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')
    plot_graph(LRCN_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')

if __name__ == '__main__':
    # setting seed constant
    seed = 25
    np.random.seed(seed)
    random.seed(seed)

    # determine video size, directory and classes
    h, w = 64, 128
    DATASET_DIR = 'UAV-Gesture'
    CLASS_LIST = ['BOXING', 'CLAPPING', 'HITTING', 'JOGGING-F', 'JOGGING-SIDE', 'KICKING',
                  'RUNNING-F', 'RUNNING-SIDE', 'STABBING', 'WALKING-F', 'WALKING-SIDE', 'WAVING']
    # frames to insert to model
    SEQUENCE = 20 # skipping every few frames

    # no. of iterations
    epochs = 25
    main() # running main












