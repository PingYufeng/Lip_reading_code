#!/usr/bin/env python

import errno
import fnmatch
import os
import sys

import dlib
import numpy as np
import skvideo.io
import tensorflow as tf
from skimage import io, transform
from multiprocessing import Pool

FACE_PREDICTOR_PATH = './dataset/shape_predictor_68_face_landmarks.dat'

class Video(object):
    def __init__(self, vtype='mouth', face_predictor_path=None):
        if vtype == 'face' and face_predictor_path is None:
            raise AttributeError(
                'Face video need to be accompanied with face predictor')
        self.face_predictor_path = face_predictor_path
        self.vtype = vtype
        self.fail = False
        self.frame = 0
    
    def from_video(self, path):
        frames = self.get_video_frames(path)
        self.handle_type(frames)
        return self
    
    def handle_type(self, frames):
        if self.vtype == 'mouth':
            self.process_frames_mouth(frames)
        elif self.vtype == 'face':
            self.process_frames_face(frames)
        else:
            raise Exception('Video type not found')

    def process_frames_mouth(self, frames):
        pass
    
    def process_frames_face(self, frames):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.face_predictor_path)
        mouth_frames = self.get_frames_mouth(detector, predictor, frames)
        self.face = np.array(frames)
        self.mouth = np.array(mouth_frames)
        self.set_data(mouth_frames)
    
    def get_frames_mouth(self, detector, predictor, frames):
        MOUTH_WIDTH = 100
        MOUTH_HEIGHT = 50
        HORIZONTAL_PAD = 0.19
        normalize_ratio = None

        for frame in frames:
            self.frame += 1
            dets = detector(frame, 1) # 1 - unsample times
            shape = None
            for k, d in enumerate(dets):
                shape = predictor(frame, d)
                i = -1
            if shape is None:
                if self.fail == False:
                    print("detector doesn't detect face")
                    self.fail = True
                mouth_frames.append(frame)
                continue
            mouth_points = []
            for part in shape.parts():
                i += 1
                if i < 48: # Only take mouth region
                    continue
                mouth_points.append((part.x, part.y))
            np_mouth_points = np.array(mouth_points)

            mouth_centroid = np.mean(no_mouth_points[:, -2:], axis=0)

            if normalize_ratio is None:
                mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0-HORIZONTAL_PAD)
                mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0+HORIZONTAL_PAD)
                normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)
            
            new_img_shape = (int(frame.shape[0] * normalize_ratio),
                             int(frame.shape[1] * normalize_ratio))
            resized_img = transform.resize(frame, new_img_shape)

            mouth_centroid_norm = mouth_centroid * normalize_ratio
            mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
            mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
            mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
            mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)

            mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]

            if mouth_crop_image.shape[0] != MOUTH_HEIGHT or mouth_crop_image.shape[1] != MOUTH_HEIGHT:
                mouth_crop_image = transform.resize(
                    mouth_crop_image, (MOUTH_HEIGHT, MOUTH_WIDTH))
            mouth_frames.append(mouth_crop_image)

        return mouth_frames

            

    def get_video_frames(self, path):
        # read video frame by frame
        videogen = skvideo.io.vreader(path)
        frames = np.array([frame for frame in videogen])
        return frames
    
    def set_data(self, frames):
        data_frames = []
        for frame in frames:
            frame = frame.swapaxes(0, 1) # width x height x channel
            if len(frame.shape) < 3:
                frame = np.array([frame]).swapaxes(0, 2).swapaxes(0, 1)
            data_frames.append(frame)
        frames_n = len(data_frames)
        data_frames = np.array(data_frames) # T X W X H X C
        if tf.keras.backend.image_data_format() == 'channels_first':
            data_frames = np.rollaxis(data_frames, 3) # C X T X W X H
        self.data = data_frames
        self.length = frames_n
        




def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def extract_mouth(filepath):
    filepath_wo_ext = os.path.splitext(filepath)[0] # no extension
    target_dir = filepath_wo_ext.replace('video', 'mouth')
    if os.path.exists(target_dir):
        print('{} already processed'.format(filepath))
        return

    print('Processing {}'.format(filepath))
    video = Video(
        vtype='face', 
        face_predictor_path=FACE_PREDICTOR_PATH
    ).from_video(filepath)

    mkdir_p(target_dir)
    i = 1
    for frame in video.mouth:
        io.imsave(os.path.join(target_dir, '{0:03d}.jpg'.format(i)), frame)
        i += 1
    
    if video.fail and video.frame == 75:
        f = open('./fail.txt', 'a')
        f.write(str(filepath)[-13:]+'\n')
        f.close()
    if video.frame<75:
        f = open('./lack.txt', 'a')
        f.write(str(filepath)[-13:] + '\n')
        f.close()


def find_files(directory, pattern):
    filenames = []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                filenames.append(filename)
    return filenames


def main():
    SOUCE_PATH = sys.argv[1]
    SOURCE_EXTS = sys.argv[2]
    TARGET_PATH = sys.argv[3]
    filenames = find_files(SOUCE_PATH, SOURCE_EXTS)
    p = Pool(4)
    p.map(extract_mouth, filenames)
    for filename in filenames:
        extract_mouth(filename)


if __name__ == '__main__':
    main()


