import os
import cv2
import random
import matplotlib.pyplot as plt

def visualize(CLASSES, num_classes):
    plt.figure(figsize = (15,8))
    for i, idx in enumerate (range,1):
        class_Name = CLASSES[idx]
        video_lists = os.listdir(f'UAV-Gesture/{class_Name}')
        selected_vid = random.choice(video_lists)
        vid_reader = cv2.VideoCapture(f'UAV-Gesture/{class_Name}/{selected_vid}')
        _, frame_one = vid_reader.read()
        vid_reader.release()

        rgb = cv2.cvtColor(frame_one, cv2.COLOR_BGR2RGB)

        plt.subplot(4, 4, i)
        plt.text(50, 200, class_Name)
        plt.imshow(rgb)
        plt.axis('off')
    plt.show()

# getting dataset and visualizing
CLASSES = os.listdir('UAV-Gesture')
num_class = len(CLASSES)
range = random.sample(range(num_class), 13)

visualize(CLASSES, range)

if __name__ == '__main__':
    # getting dataset and visualizing
    CLASSES = os.listdir('UAV-Gesture')
    num_class = len(CLASSES)
    range = random.sample(range(num_class), 13)

    visualize(CLASSES, range)