import os
import cv2
import random
import matplotlib.pyplot as plt

def visualize(CLASSES, num_classes):
    plt.figure(figsize = (5,4)) # plot size
    for i, idx in enumerate (num_classes,1):
        class_Name = CLASSES[idx] # retrieve class names from idx
        video_lists = os.listdir(f'OpenPose/{class_Name}')
        selected_vid = random.choice(video_lists) # select random video
        vid_reader = cv2.VideoCapture(f'OpenPose/{class_Name}/{selected_vid}') # start vid capture
        _, frame_one = vid_reader.read() # reading 1st frame
        vid_reader.release() # stop vid capture

        # getting rgb images
        rgb = cv2.cvtColor(frame_one, cv2.COLOR_BGR2RGB)
        # plotting and display
        plt.subplot(3, 3, i)
        plt.text(50, 200, class_Name)
        plt.imshow(rgb)
        plt.axis('off')
    plt.show()

# getting dataset and visualizing
CLASSES = os.listdir('OpenPose')
num_class = len(CLASSES)
# get all classes in random order
range = random.sample(range(num_class), 7)

visualize(CLASSES, range)