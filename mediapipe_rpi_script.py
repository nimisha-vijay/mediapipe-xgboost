import mediapipe as mp
import math
import numpy
import cv2
import os
from tqdm import tqdm 
import pickle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic


def get_annot(image_array):
		return_array = []
		
		pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
		for idx, image in enumerate(image_array):
#				 image = cv2.imread(file)
			temp_array = []
			image_size, image_size, _ = image.shape
			# Convert the BGR image to RGB before processing.
			results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

			if not results.pose_landmarks:
				continue
			
			nose_x = results.pose_landmarks.landmark[0].x * image_size
			nose_y = results.pose_landmarks.landmark[0].y * image_size
#				 middle_x = ((results.pose_landmarks.landmark[23].x + results.pose_landmarks.landmark[24].x) / 2) * image_size
#				 middle_y = ((results.pose_landmarks.landmark[23].y + results.pose_landmarks.landmark[24].y) / 2) * image_size
			
			for index in range(10, 23):		
					xxx = results.pose_landmarks.landmark[index].x * image_size
					yyy = results.pose_landmarks.landmark[index].y * image_size
					temp_array.append(math.sqrt((xxx - nose_x)**2 + (yyy - nose_y)**2))
					temp_array.append(math.atan2(yyy - nose_y, xxx - nose_x))
#						 temp_array.append(math.sqrt((xxx - middle_x)**2 + (yyy - middle_y)**2))	
			return_array.append(temp_array)
			
		return return_array


def getFrame(sec):
	vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
	hasFrames,image = vidcap.read()
	if hasFrames:
#				 image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			image = cv2.resize(image, (image_size, image_size))

	return [hasFrames, image]



video = "./takecarenimisha.mp4"
image_size = 512
frame_rate = 1/12
frames = []

vidcap = cv2.VideoCapture(video)
sec = 0
mini = []
success, frame = getFrame(sec)
while success:
		mini.append(frame)
		sec = sec + frame_rate
		sec = round(sec, 2)
		success, frame = getFrame(sec)

frames.append(numpy.array(get_annot(mini)))

max_length = max([len(video) for video in frames])
limit = 64

new_frames = []
for video in tqdm(frames):
		if video.shape[0] < limit:
				pad = numpy.zeros((limit, video.shape[1]))
				pad[:video.shape[0], :video.shape[1]] = video
				new_frames.append(pad)
		else:
				new_frames.append(video[:limit, :video.shape[1]])
		
new_frames = numpy.array(new_frames)
frame_size = new_frames.shape
new_frames = numpy.reshape(new_frames, (frame_size[0], frame_size[1]*frame_size[2]))

model = pickle.load(open("./mediapipe_xgboost.sav", 'rb'))

print(model.predict(new_frames))