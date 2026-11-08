import cv2
import mediapipe as mp
import time
from pathlib import Path

# Initialize MediaPipe Face Landmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = Path(__file__).resolve().parent / "models" / "face_landmarker.task"

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_path)),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

#initialize laptop webcam
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("ERROR: Webcam could not be opened.")
    exit()

start_time = time.monotonic()

#initialize facial landmark detector
with FaceLandmarker.create_from_options(options) as detector:

    #continuously capture frames from camera feed
    while True:

        #retrieve the frame
        success, frame = cam.read()

        if not success:
            print("ERROR: Frame could not be captured.")
            break

        #convert OpenCV BGR frame to RGB fro MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data = rgb_frame
        )

        timestamp_ms = int((time.monotonic() - start_time) * 1000)

        #process frame using MediaPipe
        result = detector.detect_for_video(image, timestamp_ms)

        #check if mediapipe recognizes a face
        if len(result.face_landmarks) > 0:

            #extract facial landmarks of the first detected face 
            face_landmarks = result.face_landmarks[0]
            
            #get the height and width of the webcam frames in pixel
            frame_height, frame_width = frame.shape[:2]

            # loop through each landmark detected by MediaPipe
            for landmark in face_landmarks:
                
                #landmark coordinates are normalized
                #convert to pixels based on frame dimensions
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                
                #draw a tiny green circle at each of the landmarks
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            #text in top left when face is detected
            cv2.putText(
                frame,
                "Face Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        #if len(result.face_landmarks) < 0, then theres no face, so disp that top left
        else:

            cv2.putText(
                frame,
                "No Face Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )


        #display the webcam frame
        cv2.imshow("Lookout - Driver Monitoring", frame)

        #exit when user presses q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

#release all resources
cam.release()
cv2.destroyAllWindows()