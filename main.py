from sklearn.model_selection import train_test_split
from dataset import datasets
from model import models
import matplotlib.pyplot as plt
import numpy as np
import random
import os
#import dataset_visualization # run dataset visualization

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
    """

    # initiate required utilities
    from utils import Utils
    utils = Utils()

    # preparing dataset
    dataset = datasets(h, w, DATASET_DIR, CLASS_LIST, SEQUENCE)
    # create and save dataset if doesn't exist
    if not os.path.exists(f'image_array/{h}x{w}_features.npy'):
        dataset.make_dataset()
    # load dataset
    feature, labels = dataset.load_dataset()
    one_hot_label = utils.to_categorical(labels)  # convert labels to one_hot format

    # split datasets for training and testing
    feature_train, feature_test, label_train, label_test = train_test_split(
        feature, one_hot_label, test_size=0.25, shuffle=True, random_state=seed)

    # initiate model class
    model = models(seed)
    # choose model
    LRCN_model = model.create_LRCN(SEQUENCE, h, w, CLASS_LIST)
    LRCN_history, test_loss, test_acc = model.compile_model(
        LRCN_model, epochs, feature_train, feature_test, label_train, label_test)

    # saving model
    file_name = f'model results/LRCN_model__Loss_{test_loss}__Acc_{test_acc}'
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
    h, w = 240, 360
    DATASET_DIR = 'datasets/UAV-Gesture'
    CLASS_LIST = ['JOGGING-F', 'JOGGING-SIDE', 'RUNNING-F',
                  'RUNNING-SIDE', 'WALKING-F', 'WALKING-SIDE', 'WAVING']
    # frames to insert to model
    SEQUENCE = 15 # skipping every few frames

    # no. of iterations
    epochs = 50
    main() # running main

    """
    https: // github.com / faaip / OpenPose - Gesture - Recognition / blob / master / run_openpose.py
    https: // pythonawesome.com / a - gesture - recognition - system -
    with-openpose /"""












