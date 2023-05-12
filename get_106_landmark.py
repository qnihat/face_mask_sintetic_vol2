"""
@author: JiXuan Xu, Jun Wang
@date: 20201023
@contact: jun21wangustc@gmail.com
"""
import sys
sys.path.append('.')
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')

import yaml
#print(yaml.__version__)
import cv2
import numpy as np
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler

with open('config/model_conf.yaml') as f:
    model_conf = yaml.load(f)

def get_106_landmark_funk(full_img,face_coordinates):
    # common setting for all model, need not modify.
    model_path = 'models'

    # model setting, modified along with model
    scene = 'non-mask'
    model_category = 'face_alignment'
    model_name =  model_conf[scene][model_category]
    # load model
    faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)



    model, cfg = faceAlignModelLoader.load_model()


    faceAlignModelHandler = FaceAlignModelHandler(model, 'cuda:0', cfg)

    # read image
    #image_path = 'test_images/3.jpeg'
    #image_det_txt_path = 'test_images/1_detect_res.txt'
    #image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image=full_img
    #with open(image_det_txt_path, 'r') as f:
    #    lines = f.readlines()
    lines=face_coordinates
    for line in lines.split('\n'):
        if len(line):
            line = lines.strip().split()
            det = np.asarray(list(map(int, line[0:4])), dtype=np.int32)
            landmarks = faceAlignModelHandler.inference_on_image(image, det)

            #save_path_img = 'test1_' + 'landmark_res' + str(i) + '.jpg'
            #save_path_txt = 'test1_' + 'landmark_res' + str(i) + '.txt'
            image_show = image.copy()
            #with open(save_path_txt, "w") as fd:
            lmk_line=''
            for (x, y) in landmarks.astype(np.int32):
                cv2.circle(image_show, (x, y), 2, (255, 0, 0),-1)
                lmk_line = lmk_line+str(x) + ' ' + str(y) + ' '
            lmk_line=lmk_line+" \n"
    print('landmarks: ',lmk_line)
    return image_show,lmk_line
