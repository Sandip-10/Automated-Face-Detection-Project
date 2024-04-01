# Import necessary libraries
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Set the path to the dataset
data_path = 'E:\\WorksSpace\\Project\\facial_recognition_1\\Dateset\\'

# Get a list of all files in the dataset directory
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

# Initialize empty lists to store training data and labels
Training_Data, Labels = [], []

# Loop through each file in the dataset
for i, files in enumerate(onlyfiles):
    # Create the full path to the file
    image_path = data_path + onlyfiles[i]
    # Read the image from the file
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Append the image to the Training_Data list
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    # Append the label (i.e., the file index) to the Labels list
    Labels.append(i)

# Convert the Labels list to a numpy array
Labels = np.asarray(Labels, dtype=np.int32)

# Create a LBPH face recognizer model
model = cv2.face.LBPHFaceRecognizer_create()

# Train the model using the Training_Data and Labels arrays
model.train(np.asarray(Training_Data), np.asarray(Labels))

# Print a message indicating that the model training is complete
print("Dataset Model Training Complete!!!!!")

# Load the Haar Cascade Classifier for face detection
face_classifier = cv2.CascadeClassifier('E:\\WorksSpace\\Project\\facial_recognition_1\\haarcascade_frontalface_default.xml')

# Define a function for detecting faces in an image
def face_detector(img, size = 0.5):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    # If faces are detected, loop through each face
    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
        # Crop the detected face from the image
        roi = img[y:y+h, x:x+w]
        # Resize the cropped face to 200x200 pixels
        roi = cv2.resize(roi, (200,200))

    return img,roi

# Open the default camera using OpenCV
cap = cv2.VideoCapture(0)

# Begin an infinite loop to continuously capture and process frames from the camera
while True:

    # Capture a frame from the camera
    ret, frame = cap.read()

    # Detect faces in the frame
    image, face = face_detector(frame)

    # If a face is detected, predict the label using the trained model
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        # Calculate the confidence level for the prediction
        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))

        # Display the predicted name if the confidence level is above 82%
        if confidence > 82:
            cv2.putText(image, "Sandip Karmakar", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Face Cropper', image)

        # Display "Unknown" if the confidence level is below 82%
        else:
            cv2.putText(image, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)

    # If no face is detected, display "Face Not Found"
    except:
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass

    # Break the loop if the user presses the "Enter" key
    if cv2.waitKey(1)==13:
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()