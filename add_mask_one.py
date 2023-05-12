from face_masker import FaceMasker
import time

def dd_mask_to_face(image_path,face_lms_file):
    is_aug = False
    #image_path = 'test_images/3.jpeg'
    #face_lms_file = 'test1_landmark_res0.txt'
    template_name = '8.png'
    masked_face_path = 'masked_faces/'+str(time.time())+'_mask.jpg'
    face_lms_str = face_lms_file.strip().split(' ')
    face_lms = [float(num) for num in face_lms_str]
    face_masker = FaceMasker(is_aug)
    face_masker.add_mask_one(image_path, face_lms, template_name, masked_face_path)