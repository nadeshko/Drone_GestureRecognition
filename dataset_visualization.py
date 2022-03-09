import os
import cv2
import random
import matplotlib.pyplot as plt

def visualize(CLASSES, num_classes):
    plt.figure(figsize = (15,8)) # plot size
    for i, idx in enumerate (num_classes,1):
        class_Name = CLASSES[idx] # retrieve class names from idx
        video_lists = os.listdir(f'UAV-Gesture/{class_Name}')
        selected_vid = random.choice(video_lists) # select random video
        vid_reader = cv2.VideoCapture(f'UAV-Gesture/{class_Name}/{selected_vid}') # start vid capture
        _, frame_one = vid_reader.read() # reading 1st frame
        vid_reader.release() # stop vid capture

        # getting rgb images
        rgb = cv2.cvtColor(frame_one, cv2.COLOR_BGR2RGB)
        # plotting and display
        plt.subplot(4, 4, i)
        plt.text(50, 200, class_Name)
        plt.imshow(rgb)
        plt.axis('off')
    plt.show()

# get constant seed
seed = 25
random.seed(seed)
# getting dataset and visualizing
CLASSES = os.listdir('UAV-Gesture')
num_class = len(CLASSES)
# get all classes in random order
range = random.sample(range(num_class), 13)

visualize(CLASSES, range)