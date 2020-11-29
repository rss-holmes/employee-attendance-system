import numpy as np
import cv2
import os
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials


def Recognizer():
    video = cv2.VideoCapture(0)

    while (True):
        check, frame = video.read()
        cv2.imshow('img1', frame)  # display the captured image
        if cv2.waitKey(1) & 0xFF == ord('y'):  # save on pressing 'y'
            # img_str = cv2.imencode('.jpg', frame)[1].tostring()
            image_save_url = os.path.join(os.getcwd(), "{}\{}\{}\c1.png".format('static', 'images', 'temp'))
            cv2.imwrite(image_save_url, frame)
            cv2.destroyAllWindows()
            break

    video.release()
    employee_ids = check_capture(image_save_url)
    return employee_ids


def check_capture(image_save_url):
    KEY = '4794e1e9ff6a41728e793d581b065d43'
    ENDPOINT = 'https://face-rec-instance.cognitiveservices.azure.com/'
    base_dir = os.getcwd()
    IMAGE_BASE_URL = os.path.join(base_dir, "{}\{}\{}".format('static', 'images', 'employee_images'))
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

# def Recognizer(details):
#
# 	video = cv2.VideoCapture(0)
#
# 	if not video.isOpened():
# 		video.open(0)
#
# 	known_face_encodings = []
# 	known_face_names = []
# 	base_dir = os.path.dirname(os.path.abspath(__file__))
# 	base_dir = os.getcwd()
# 	# image_dir = os.path.join(base_dir,"{}\{}\{}\{}".format('static','images','Student_Images',details['company']))
# 	image_dir = os.path.join(base_dir,"{}\{}\{}".format('static','images','employee_images'))
# 	print("Directory  where employee images are being checked - " + str(image_dir))
# 	names = []
#
# 	for root,dirs,files in os.walk(image_dir):
# 		for file in files:
# 			if file.endswith('jpg') or file.endswith('png'):
# 				path = os.path.join(root, file)
# 				img = face_recognition.load_image_file(path)
# 				label = file[:len(file)-4]
# 				img_encoding = face_recognition.face_encodings(img)[0]
# 				known_face_names.append(label)
# 				known_face_encodings.append(img_encoding)
#
# 	face_locations = []
# 	face_encodings = []
#
#
# 	while True:
#
# 		check, frame = video.read()
# 		small_frame = cv2.resize(frame, (0,0), fx=0.5, fy= 0.5)
# 		rgb_small_frame = small_frame[:,:,::-1]
#
# 		face_locations = face_recognition.face_locations(rgb_small_frame)
# 		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
# 		face_names = []
#
#
# 		for face_encoding in face_encodings:
#
# 			matches = face_recognition.compare_faces(known_face_encodings, np.array(face_encoding), tolerance = 0.6)
#
# 			face_distances = face_recognition.face_distance(known_face_encodings,face_encoding)
#
# 			try:
# 				matches = face_recognition.compare_faces(known_face_encodings, np.array(face_encoding), tolerance = 0.6)
#
# 				face_distances = face_recognition.face_distance(known_face_encodings,face_encoding)
# 				best_match_index = np.argmin(face_distances)
#
# 				if matches[best_match_index]:
# 					name = known_face_names[best_match_index]
# 					face_names.append(name)
# 					if name not in names:
# 						names.append(name)
# 			except:
# 				pass
#
# 		if len(face_names) == 0:
# 			for (top,right,bottom,left) in face_locations:
# 				top*=2
# 				right*=2
# 				bottom*=2
# 				left*=2
#
# 				cv2.rectangle(frame, (left,top),(right,bottom), (0,0,255), 2)
#
# 				# cv2.rectangle(frame, (left, bottom - 30), (right,bottom - 30), (0,255,0), -1)
# 				font = cv2.FONT_HERSHEY_DUPLEX
# 				cv2.putText(frame, 'Unknown', (left, top), font, 0.8, (255,255,255),1)
# 		else:
# 			for (top,right,bottom,left), name in zip(face_locations, face_names):
# 				top*=2
# 				right*=2
# 				bottom*=2
# 				left*=2
#
# 				cv2.rectangle(frame, (left,top),(right,bottom), (0,255,0), 2)
#
# 				# cv2.rectangle(frame, (left, bottom - 30), (right,bottom - 30), (0,255,0), -1)
# 				font = cv2.FONT_HERSHEY_DUPLEX
# 				cv2.putText(frame, name, (left, top), font, 0.8, (255,255,255),1)
#
# 		cv2.imshow("Face Recognition Panel",frame)
#
# 		if cv2.waitKey(1) == ord('s'):
# 			break
#
# 	video.release()
# 	cv2.destroyAllWindows()
# 	return names
