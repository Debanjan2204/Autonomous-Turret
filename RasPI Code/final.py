import cv2
import time
import numpy as np
import serial #pip install pyserial

# Set the random seed for reproducibility
np.random.seed(20)


ser = serial.Serial('/dev/serial0', 115200, timeout=1)
time.sleep(2)  # Wait for the serial connection to initialize

# Function to send firing point coordinates to ESP32

def send_coordinates(x, y):
    data = f"{x},{y}\n"  # Convert coordinates to string format
    # print(data,'\n')
    ser.write(data.encode())  # Send the data to ESP32

def read_from_esp32():
    if ser.in_waiting > 0:
        response = ser.readline().decode().strip()
        if(response== 0x7777777):
            print(f"Received from ESP32: ALLY DONT SHOOT({response})")
        else:
            print(f"Received from ESP32: ENEMY SHOOT({response})")



def main():
    # Set the video source: use 0 for webcam or specify a video file path
    # videoPath = "test_videos/street1.mp4"  # To access video from a file
    videoPath = 0  # For Webcam

    # Paths to the model configuration, weights, and class labels
    configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    # Create an instance of the Detector class and start video processing
    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()

class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        # Load the pre-trained model
        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        # Read class labels
        self.readClasses()

    def readClasses(self):
        # Read the class labels from the file
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()

        # Insert a background class at index 0
        self.classesList.insert(0, '__Background__')

        # Generate random colors for each class label
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 0))

    def onVideo(self):
        # Open the video capture
        cap = cv2.VideoCapture(self.videoPath)
        if not cap.isOpened():
            print("Error opening file...")
            return

        # Read the first frame
        success, image = cap.read()
        startTime = 0

        while success:
            currentTime = time.time()
            fps = 1 / (currentTime - startTime)
            startTime = currentTime

            # Perform object detection
            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold=0.4)
            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))

            # Apply Non-Maximum Suppression to filter out overlapping bounding boxes
            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)

            for i, bbox in enumerate(bboxs):
                classConfidence = confidences[i]
                classLabelID = int(classLabelIDs[i])
                classLabel = self.classesList[classLabelID]

                # Only process if the detected object is a person and confidence is above 0.69
                if classLabel == 'person' and classConfidence > 0.69:
                    x, y, w, h = bbox
                    center_x = x + w // 2
                    center_y = y + h // 2

                    # Draw bounding box and center point on the image
                    classColor = [int(c) for c in self.colorList[classLabelID]]
                    cv2.rectangle(image, (x, y), (x + w, y + h), classColor, thickness=2)
                    displayText = "{}:{:.2f}".format(classLabel, classConfidence)
                    cv2.putText(image, displayText, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.circle(image, (center_x, center_y), 5, (0, 255, 255), -1)

                    # Send coordinates to ESP32
                    send_coordinates(center_x, center_y)

            
            read_from_esp32()
            
            # Display the frame with detections
            cv2.imshow("Result", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            # Read the next frame
            success, image = cap.read()

        # Release the video capture and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()




if __name__ == '__main__':
    main()

