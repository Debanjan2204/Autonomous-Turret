import cv2
# import pigpio  #sudo apt-get install pigpio python-pigpio  # sudo pigpiod

# import RPi.GPIO as GPIO
import time
import serial #pip install pyserial


# GPIO pin setup

# LASER_PIN = 22
# BUZZER_PIN = 23


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
        print(f"Received from ESP32: {response}")



def main():
    
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read() 
            
            if not ret:
                print("Failed to grab frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                (x, y, w, h) = faces[0]

                # Center of the object
                center_x = x + w // 2
                center_y = y + h // 2
                
                # GPIO.output(LASER_PIN, GPIO.HIGH)
                # GPIO.output(BUZZER_PIN, GPIO.HIGH)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                     # Send the firing point coordinates to ESP32
                    send_coordinates(center_x, center_y)

            # else:
                # GPIO.output(LASER_PIN, GPIO.LOW)
                # GPIO.output(BUZZER_PIN, GPIO.LOW)
            read_from_esp32()
            cv2.imshow('Face Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        #pi.stop()
        # GPIO.cleanup()
        ser.close()

if __name__ == "__main__":
    main()
