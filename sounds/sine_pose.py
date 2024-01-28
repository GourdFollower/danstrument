import mediapipe as mp
#from mediapipe.tasks import python
#from mediapipe.tasks.python import vision
import cv2
import time

from pysinewave import SineWave
from pydub import AudioSegment
from pydub.playback import play

import sys
sys.path.append('/opt/homebrew/Cellar/ffmpeg')

vid = cv2.VideoCapture(0)

VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

sinewave = SineWave(pitch=6, pitch_per_second=20)
drum = AudioSegment.from_wav("/Users/tkdavis/Coding/Hackathons/McHacks11/sounds/clicky-snare-drum-hit.wav")
drum = drum - 5

right_wrist_pos = []
left_wrist_pos = []
played = False

while True:
    sinewave.play()
    ret, frame = vid.read()
    timestamp = time.time()
    pose_results = pose.process(frame)
    mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    image_height, image_width, _ = frame.shape

    right_wrist = pose_results.pose_landmarks.landmark[16]
    right_wrist_pos.append({
                         'X': right_wrist.x * image_width,
                         'Y': right_wrist.y * image_height,
                         'Z': right_wrist.z,
                         'Visibility': right_wrist.visibility,
                         })

    left_wrist = pose_results.pose_landmarks.landmark[15]
    left_wrist_pos.append({
        'X': left_wrist.x,
        'Y': left_wrist.y,
        'Z': left_wrist.z,
        'Visibility': left_wrist.visibility,
    })

    """if len(right_wrist_pos) > 55:
        if right_wrist_pos[len(right_wrist_pos) - 1]['Y'] > 0.1 + right_wrist_pos[len(right_wrist_pos) - 30]['Y']:
            sinewave.set_pitch(-5)
        if right_wrist_pos[len(right_wrist_pos) - 1]['Y'] < 0.1 + right_wrist_pos[len(right_wrist_pos) - 30]['Y']:
            sinewave.set_pitch(12)"""

    if len(right_wrist_pos) > 55:
        if right_wrist_pos[len(right_wrist_pos) - 1]['Y'] > 700:
            sinewave.set_pitch(-5)
        if right_wrist_pos[len(right_wrist_pos) - 1]['Y'] < 200:
            sinewave.set_pitch(12)

    """if len(left_wrist_pos) % 7 == 0:
        if len(left_wrist_pos) > 55:
            if left_wrist_pos[len(left_wrist_pos) - 1]['Y'] > 0.5 + left_wrist_pos[len(left_wrist_pos) - 30]['Y']:
                play(drum)
            if left_wrist_pos[len(right_wrist_pos) - 1]['X'] < 0.1 + left_wrist_pos[len(left_wrist_pos) - 30]['X']:
                continue
                # sinewave.set_pitch(12)"""

    print(right_wrist.x * image_width, right_wrist.y * image_height)

    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object 
vid.release()
# Destroy all the windows 
cv2.destroyAllWindows()
