import mediapipe as mp
import cv2
import time

from pydub import AudioSegment
from pydub.playback import play

from threading import Thread

import sys
sys.path.append('/opt/homebrew/Cellar/ffmpeg')

vid = cv2.VideoCapture(0)

VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

C = AudioSegment.from_wav("/Users/tkdavis/Coding/Hackathons/McHacks11/sounds/piano/68441__pinkyfinger__piano-c.wav")
C = C[:700]

D = AudioSegment.from_wav("/Users/tkdavis/Coding/Hackathons/McHacks11/sounds/piano/68442__pinkyfinger__piano-d.wav")
D = D[:700]

E = AudioSegment.from_wav("/Users/tkdavis/Coding/Hackathons/McHacks11/sounds/piano/68443__pinkyfinger__piano-e.wav")
E = E[:700]

F = AudioSegment.from_wav("/Users/tkdavis/Coding/Hackathons/McHacks11/sounds/piano/68446__pinkyfinger__piano-f.wav")
F = F[:700]

G = AudioSegment.from_wav("/Users/tkdavis/Coding/Hackathons/McHacks11/sounds/piano/68448__pinkyfinger__piano-g.wav")
G = G[:700]

A = AudioSegment.from_wav("/Users/tkdavis/Coding/Hackathons/McHacks11/sounds/piano/68437__pinkyfinger__piano-a.wav")
A = G[:700]

B = AudioSegment.from_wav("/Users/tkdavis/Coding/Hackathons/McHacks11/sounds/piano/68438__pinkyfinger__piano-b.wav")
B = B[:700]

right_wrist_pos = []
left_wrist_pos = []

last_played = 'C'


def play_C():
    play(C)

def play_D():
    play(D)

def play_E():
    play(E)

def play_F():
    play(F)

def play_G():
    play(G)

def play_A():
    play(A)

def play_B():
    play(B)


while True:
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

    """if right_wrist_pos[len(right_wrist_pos) - 1]['X'] < 100:
        if last_played != 'C':
            thread = Thread(target=play_C)
            thread.start()
            last_played = 'C'

    if 300 < right_wrist_pos[len(right_wrist_pos) - 1]['X'] < 400:
        if last_played != 'F':
            thread = Thread(target=play_F)
            thread.start()
            last_played = 'F'

    if 500 < right_wrist_pos[len(right_wrist_pos) - 1]['X'] < 600:
        if last_played != 'G':
            thread = Thread(target=play_G)
            thread.start()
            last_played = 'G'

    if 700 < right_wrist_pos[len(right_wrist_pos) - 1]['X'] < 800:
        if last_played != 'A':
            thread = Thread(target=play_A)
            thread.start()
            last_played = 'A'"""

    if right_wrist_pos[len(right_wrist_pos) - 1]['X'] < 100:
        if last_played != 'C':
            thread = Thread(target=play_C)
            thread.start()
            last_played = 'C'

    if 300 < right_wrist_pos[len(right_wrist_pos) - 1]['X'] < 400:
        if last_played != 'D':
            thread = Thread(target=play_D)
            thread.start()
            last_played = 'D'

    if 500 < right_wrist_pos[len(right_wrist_pos) - 1]['X'] < 600:
        if last_played != 'E':
            thread = Thread(target=play_E)
            thread.start()
            last_played = 'E'

    if 600 < right_wrist_pos[len(right_wrist_pos) - 1]['Y'] < 700:
        if last_played != 'F':
            thread = Thread(target=play_F)
            thread.start()
            last_played = 'F'

    if 800 < right_wrist_pos[len(right_wrist_pos) - 1]['Y'] < 900:
        if last_played != 'G':
            thread = Thread(target=play_G)
            thread.start()
            last_played = 'G'

    if 1000 < right_wrist_pos[len(right_wrist_pos) - 1]['Y'] < 1100:
        if last_played != 'A':
            thread = Thread(target=play_A)
            thread.start()
            last_played = 'A'

    if 1000 < right_wrist_pos[len(right_wrist_pos) - 1]['Y'] > 1200:
        if last_played != 'B':
            thread = Thread(target=play_B)
            thread.start()
            last_played = 'B'

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
