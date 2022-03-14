from tempfile import TemporaryFile
import numpy as np
import cv2
import os

class datasets():
    def __init__(self, h, w, DIR, CLASS_LIST, SEQUENCE):
        self.h = h
        self.w = w
        self.DIR = DIR
        self.SEQUENCE = SEQUENCE
        self.CLASS_LIST = CLASS_LIST

        # initialize required lists, same amt of features and labels
        self.feature = []
        self.labels = []
        self.vid_paths = []
        self.outfile = TemporaryFile()

    def get_frames(self, path):
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
        frame_skip = max(int(frame_count / self.SEQUENCE), 1)

        # extracting frames
        for i in range(self.SEQUENCE):
            # skip a couple of frames (don't need everything)
            vid_reader.set(cv2.CAP_PROP_POS_FRAMES, i * frame_skip)
            success, frame = vid_reader.read()
            # if no more frames, finish
            if not success:
                break
            resized_frame = cv2.resize(frame, (self.h, self.w))
            normalized_frame = resized_frame / 255
            frame_list.append(normalized_frame)
        vid_reader.release()  # release object

        return frame_list

    def save_dataset(self):
        if not os.path.exists('image_array'):
            os.makedirs('image_array')
            print("Created new directory 'image_array'")
        np.save(f'image_array/{self.h}x{self.w}_features', self.features)
        np.save(f'image_array/{self.h}x{self.w}_labels', self.labels)

    def load_dataset(self):
        features = np.load(f'image_array/{self.h}x{self.w}_features.npy')
        labels = np.load(f'image_array/{self.h}x{self.w}_labels.npy')
        return features, labels

    def make_dataset(self):
        """
        Function to prepare dataset for training
        :return: extracted features, labels and video paths from video frames
        """

        # go through all classes in CLASS_LIST and get features and labels
        for idx, class_name in enumerate(self.CLASS_LIST):
            print(f'Extracting Data from: {class_name}')  # show progress
            files = os.listdir(os.path.join(self.DIR, class_name))
            for name in files:
                path = os.path.join(self.DIR, class_name, name)
                # send path to get_frames from specific videos
                frames = self.get_frames(path)
                if len(frames) == self.SEQUENCE:
                    self.feature.append(frames)
                    self.labels.append(idx)
                    self.vid_paths.append(path)

        # convert features and labels into arrays
        self.features = np.asarray(self.feature)
        self.labels = np.array(self.labels)

        self.save_dataset()




