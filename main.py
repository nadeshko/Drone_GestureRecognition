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

def plot_metric(training_history, metric1, metric2, plot_name):
    '''
    This function will plot the metrics passed to it in a graph.
    Args:
        model_training_history: A history object containing a record of training and validation
                                loss values and metrics values at successive epochs
        metric_name_1:          The name of the first metric that needs to be plotted in the graph.
        metric_name_2:          The name of the second metric that needs to be plotted in the graph.
        plot_name:              The title of the graph.
    '''

    # Get metric values using metric names as identifiers.
    metric_1 = training_history.history[metric1]
    metric_2 = training_history.history[metric2]

    # Construct a range object which will be used as x-axis (horizontal plane) of the graph.
    epochs = range(len(metric_1))

    # Plot the Graph.
    plt.plot(epochs, metric_1, 'blue', label=metric1)
    plt.plot(epochs, metric_2, 'red', label=metric2)

    # Add title to the plot.
    plt.title(str(plot_name))

    # Add legend to the plot.
    plt.legend()

def main():
    from utils import Utils
    utils = Utils()

    feature, labels, paths = make_dataset()
    one_hot_label = utils.to_categorical(labels)
    print(one_hot_label)
    feature_train, feature_test, label_train, label_test = train_test_split(
        feature, one_hot_label, test_size=0.3, shuffle=True, random_state=seed)
    
    # choose model
    model = models(seed)
    LRCN_model = model.create_LRCN(SEQUENCE, h, w, CLASS_LIST)
    LRCN_history, test_loss, test_acc = model.compile_model(
        LRCN_model, epochs, feature_train, feature_test, label_test, label_test)

    # Saving model
    file_name = f'LRCN_model__Loss_{test_loss}__Acc_{test_acc}'
    LRCN_model.save(file_name)

    plot_metric(LRCN_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')
    plot_metric(LRCN_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')

if __name__ == '__main__':
    # setting seed constant
    seed = 25
    np.random.seed(seed)
    random.seed(seed)

    # image processing
    h, w = 32, 64
    DATASET_DIR = 'UAV-Gesture1'
    CLASS_LIST = ['BOXING', 'CLAPPING', 'HITTING', 'JOGGING', 'KICKING',
                  'RUNNING', 'STABBING', 'WALKING', 'WAVING']
    SEQUENCE = 20

    epochs = 20
    main()












