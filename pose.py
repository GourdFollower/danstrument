import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 
import time

#model_path = "pose_landmarker_lite.task"
vid = cv2.VideoCapture(0) 

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Create a pose landmarker instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('pose landmarker result: {}'.format(result))

# options = PoseLandmarkerOptions(
#     #base_options=BaseOptions(model_asset_path=model_path),
#     running_mode=VisionRunningMode.LIVE_STREAM,
#     result_callback=print_result)
    
while(True):  
	ret, frame = vid.read() 
	timestamp = time.time()
	pose_results = pose.process(frame)
	mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
	
	# Convert the frame received from OpenCV to a MediaPipe’s Image object.
	mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
	# with PoseLandmarker.create_from_options(options) as landmarker:
    # # The landmarker is initialized. Use it here.
	# 	landmarker.detect_async(mp_image, timestamp)
	
	cv2.imshow('frame', frame) 

	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 