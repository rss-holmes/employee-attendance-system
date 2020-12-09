import numpy as np
import cv2
import os
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from django.conf import settings

def Recognizer(temp_file_path):
    image_save_url = os.path.join(settings.MEDIA_ROOT, temp_file_path)
    employee_ids = check_capture(image_save_url)
    return employee_ids


def check_capture(image_save_url):
    KEY = '5a4eecbb57e845729c015730c6a79e75'
    ENDPOINT = 'https://realfaceapi.cognitiveservices.azure.com/'
    base_dir = os.getcwd()
    base_dir = '/home/ubuntu/attendance/'
    IMAGE_BASE_URL = os.path.join(base_dir, "{}/{}/{}".format('static', 'images', 'employee_images'))
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

    employee_ids = list()

    for file in os.listdir(IMAGE_BASE_URL):
        if file.endswith('jpg') or file.endswith('png'):
            image1_stream = open(os.path.join(IMAGE_BASE_URL, file), "rb")

            image1_detected_faces = face_client.face.detect_with_stream(image1_stream,
                                                                        detectionModel='detection_02')
            image1_detected_face_id = image1_detected_faces[0].face_id if len(image1_detected_faces) > 0 else None
            image2_stream = open(os.path.join(image_save_url), "rb")
            image2_detected_faces = face_client.face.detect_with_stream(image2_stream,
                                                                        detectionModel='detection_02')
            image2_detected_face_id = image2_detected_faces[0].face_id if len(image2_detected_faces) > 0 else None
            if not image1_detected_face_id or not image2_detected_face_id:
                return employee_ids
            verify_result = face_client.face.verify_face_to_face(image1_detected_face_id, image2_detected_face_id)
            if verify_result.is_identical:
                var = os.path.splitext(file)[0]
                employee_ids.append(var)

    return employee_ids

