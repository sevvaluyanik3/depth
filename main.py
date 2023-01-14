import numpy as np
import cv2
import time

path_model= "models/"

# Read Network
model_name="model-f6b98070.onnx"; # MiDaS v2.1 Large
# model_name= "model-small.onnx"; # MiDaS v2.1 Small

# Load the DNN model
model = cv2.dnn.readNet(path_model + model_name)

if( model.empty() ):
    print("Could not load the neural net! - Check path")

#Set backend and target to CUDA to use GPU
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableBackend(cv2.dnn.DNN_TARGET_CUDA)

# Webcam
cap=cv2.VideoCapture(0)

while cap.isOpened():

    #Read in the image
    success, img=cap.read()

    imgHeight, imgWidth, channels = img.shape

    # start time to calculate FPS
    start=time.time()

     #Create Blob from Input Image
#MiDaS v2.1 Large (Scale : 1/255, Size: 384x384, Mean Subtraction: (123.67, 116.28, 103.53), Channels Order: RGB)
blob = cv2.dnn.blobFromImage(img, 1/255., (384,384), (123.675, 116.28, 103.53), True, False)

#MiDaS v2.1 Small (Scale : 1/255, Size: 256x256, Mean Subtraction: (123.67, 116.28, 103.53), Channels Order: RGB)

# Set input to the model
model.setInput(blob)

# Make forward pass in model
output = model.forward()

output = output[0,:,:]
output = cv2.resize(output, (imgWidth, imgHeight))

# Normalize the output
output = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# End Time
end = time.time()
# calculate the FPS for current
fps = 1/(end-start)
# Show FPS
cv2.putText(img, f"{fps:.2f} FPS", (20,30), cv2.FONT_HERSHEY_)

cv2.imshow('image', img)
cv2.imshow('Depth Map', output)

if cv2.waitKey(10) & 0xFF == ord('q'):
     break

cap.release()
cv2.destroyAllWindows()