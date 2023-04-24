from tensorflow.python.keras.models import load_model
from PIL import Image
import numpy as np
import imageio
import random
import cv2
import os

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def get_frames(path):
    """
    Function extracts frames of video after it is resize and normalized
    in the form of a list.
    :param path: path of video
    :return: list of frames of every video
    """

    frame_list = []

    # set capture object and count frames
    vid_reader = cv2.VideoCapture(path)
    frame_count = int(vid_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(int(frame_count / seq), 1)

    # extracting frames
    for i in range(seq):
        # skip a couple of frames (don't need everything)
        vid_reader.set(cv2.CAP_PROP_POS_FRAMES, i * frame_skip)
        success, frame = vid_reader.read()
        # if no more frames, finish
        if not success:
            break
        frame = crop_center_square(frame)
        frame = cv2.resize(frame, (w, h))
        frame = frame / 255
        frame_list.append(frame)
    vid_reader.release()  # release object

    return frame_list

def to_gif(images):
    converted_img = images.astype(np.uint8)
    imageio.mimsave('animation.gif', converted_img, format = 'GIF-PIL',fps = 10)

if __name__ == '__main__':
    dataset_path = 'datasets'
    asd1 = 'P_3LRCN_model__Loss_0.7726011276245117__Acc_0.5833333134651184'
    asd = '3LRCN_model__Loss_4.3943634033203125__Acc_0.4166666567325592'
    model_path = 'results/3LRCN_model__Loss_4.3943634033203125__Acc_0.4166666567325592'
    vid1 = 'testing/clapping.mp4'
    vid = 'datasets1/waving/S7_wavingHands_HD.mp4'
    h, w = 224, 224
    seq = 50

    #class_names = os.listdir(dataset_path)
    class_names = ['clapping', 'kicking', 'stabbing', 'walking_f_b', 'walking_side', 'waving', 'hitting']
    num_class = len(class_names)

    random_class = random.choice(os.listdir(dataset_path))
    print(random_class)
    random_path = os.path.join(dataset_path, random_class)
    random_vid = random.choice(os.listdir(random_path))
    print(random_vid)
    vid_path = os.path.join(random_path, random_vid)

    vid_frames = np.asarray(get_frames(vid1))
    print(vid_frames.shape) # DEBUG
    vid_pred = np.expand_dims(vid_frames, 0)
    print(vid_pred.shape)  # DEBUG

    LRCN_model = load_model(model_path)
    prediction = LRCN_model.predict(vid_pred)
    predicted_idx = np.argmax(prediction)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(100*np.max(prediction), 2)

    print(predicted_class, predicted_idx, confidence)

    vid_frames = vid_frames * 255
    converted_frames = vid_frames.astype(np.uint8)
    imgs = [Image.fromarray(img) for img in converted_frames]
    imgs[0].save("array.gif", save_all=True, append_images = imgs[1:], duration = 50, loop=0)


