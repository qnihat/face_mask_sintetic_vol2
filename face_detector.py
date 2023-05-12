"""
@author: JiXuan Xu, Jun Wang
@date: 20201019
@contact: jun21wangustc@gmail.com
"""
import os
import sys
sys.path.append('.')
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')
import yaml
import cv2
import numpy as np
import imutils
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler

from get_106_landmark import get_106_landmark_funk
from add_mask_one import dd_mask_to_face

with open('config/model_conf.yaml') as f:
    model_conf = yaml.load(f)

def detect_face(image):
    # common setting for all model, need not modify.
    model_path = 'models'
    # model setting, modified along with model
    scene = 'non-mask'
    model_category = 'face_detection'
    model_name = model_conf[scene][model_category]
    # load model
    faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
    model, cfg = faceDetModelLoader.load_model()
    faceDetModelHandler = FaceDetModelHandler(model, 'cuda:0', cfg)
    dets = faceDetModelHandler.inference_on_image(image)

    bboxs = dets
    line=''
    for box in bboxs:
        line = line+str(int(box[0])) + " " + str(int(box[1])) + " " + \
               str(int(box[2])) + " " + str(int(box[3])) + " " + \
               str(box[4]) + " \n"
    for box in bboxs:
        box = list(map(int, box))
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    return line


img_dir='face_img/'
for file in os.listdir(img_dir):
    file=img_dir+file
    image = cv2.imread(file, cv2.IMREAD_COLOR)

    detection_result=detect_face(image)
    print('face:',detection_result)

    land_face,landmarks=get_106_landmark_funk(image,detection_result)
    dd_mask_to_face(file,landmarks)

    #print(landmarks)
    #cv2.imshow('face',image)
    #cv2.waitKey(0)