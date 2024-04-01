# Import required libraries
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Set the path to the dataset containing facial images for training
data_path = 'E:\\WorksSpace\\Project\\facial_recognition_1\\Dateset\\'

# Get all file names in the directory and filter out the ones that are not image files
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

# Initialize empty lists for training data and labels
Training_Data, Labels = [], []

# Loop through each image file, read the image in grayscale, and append it to the Training_Data list
# Append the index of the image file to the Labels list
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

# Convert the Labels list to a numpy array of type int32
Labels = np.asarray(Labels, dtype=np.int32)

# Create a new LBPH face recognition model
model = cv2.face.LBPHFaceRecognizer_create()

# Train the model using the training data and labels
model.train(np.asarray(Training_Data), np.asarray(Labels))

# Print a message indicating that the dataset model training is completed
print("Dataset Model Training Completed ")