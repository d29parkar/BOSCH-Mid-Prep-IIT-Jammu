import cv2
from PIL import Image
import numpy as np
from mtcnn import MTCNN
import pickle
import time
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array

detector = MTCNN()
#MTCNN Detector object

import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
#Tensorflow hub url for ESRGANs


#################### Function to help the face detection and cropping the detected face images out of the frame ##########################
def detect_face(img):    #
    
    mt_res = detector.detect_faces(img)   #Returns an array of detected faces in a frame.(In the form of bounding boxes) 
    return_res = []                       #mt_res is an array of dictionaries.
    face_pixel_array = []
    
    for face in mt_res:
        x, y, width, height = face['box']
        center = [x+(width/2), y+(height/2)]
        max_border = max(width, height)
        
        # center alignment1
        left = max(int(center[0]-(max_border/2)), 0)   #x_left coordinate
        right = max(int(center[0]+(max_border/2)), 0)  #x_right coordinate
        top = max(int(center[1]-(max_border/2)), 0)    #y_top coordinate
        bottom = max(int(center[1]+(max_border/2)), 0) #y_bottom coordinate


        
        # crop the face
        center_img_k = img[top:top+max_border, 
                           left:left+max_border, :]
        center_img = np.array(Image.fromarray(center_img_k).resize([224, 224]))   #Pixel values of the cropped faces
    
        
        # output to the cv2
        return_res.append([top, right, bottom, left])    #An array of the cropped face's bounding box top,left,right,bottom
        face_pixel_array.append(center_img)   #An array of the cropped face (in terms of pixels)
        
    return return_res, face_pixel_array


###################### Function to preprocess image so that it can be handled by ESRGANs model ###############################
def preprocess_image(face):
    '''Loads the image given make it ready for 
       the model
       Args:
         image_path: Path to the image file
    '''
    face = np.array(face)
    face = tf.convert_to_tensor(face)   #Converts numpy array to tensor
    #image = tf.image.decode_image(tf.io.read_file(face))
    
    if face.shape[-1] == 4:
        face = face[...,:-1]
    #size = (tf.convert_to_tensor(image.shape[:-1]) // 4) * 4
    size = (tf.convert_to_tensor(face.shape[:-1]) // 4) * 4
    face = tf.image.crop_to_bounding_box(face, 0, 0, size[0], size[1])
    face = tf.cast(face,tf.float32)
    return tf.expand_dims(face,0) 


#################### Function to plot images, currently not in use #######################
def plot_image(image,title=''):
   ''' 
   plots the Image tensors
   image: 3D image Tensor
   title: Title for plot
   '''
   image = np.asarray(image)
   image = tf.clip_by_value(image, 0, 255)
   image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
   plt.imshow(image)
   plt.axis('off')
   plt.title(title)

#Loading ESRGANs model from tensorflow hub
model = hub.load(SAVED_MODEL_PATH)    

################ VideoCapture, should be changed later to a parser input ########################
video_capture = cv2.VideoCapture('Test_Videos/Day_Video_2.mp4')

###################### Saving the video in the local device ##############################
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
size = (frame_width , frame_height)
result = cv2.VideoWriter('outputs/output_mtcnn.avi', cv2.VideoWriter_fourcc(*('MJPG')), 30 , size)

############# Initializing the skip frame count ##################
count = 0    


while video_capture.isOpened():
    # Grab a single frame of video
    ret, frame = video_capture.read()    
    start = time.time()

    if ret:
        # cv2.imwrite('frame{:d}.jpg'.format(count), frame)
        # count += 30 # i.e. at 30 fps, this advances one second
        # cap.set(cv2.CAP_PROP_POS_FRAMES, count)

        face_locations, faces_in_frame_array = detect_face(frame)   #face_location = List of bounding box coordinates of detected faces in a frame.
                                                                    #faces_in_frame_array = List of cropped face pixels of detected faces in a frame.

        for top, right, bottom, left in face_locations:             
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        #faces_in_frame_array = np.array(faces_in_frame_array)

        ################ESRGANs Implementation#####################
        if len(faces_in_frame_array) == 0:
            continue
        else:
            for face in faces_in_frame_array:    #"face" iterator of the face_in_frame_array over here represents cropped face image pixels
                #face = np.array(face)
                load_image = preprocess_image(face)    
                #plot_image(tf.squeeze(load_image),title='Original Photo')    
                #save_img('outputs/image/image.png', tf.make_ndarray(load_image),  scale=False)
                #tf.keras.utils.save_img("outputs\image\image.jpg", tf.make_ndarray(load_image))
                super_image = model(load_image)    #Making a super resolution face tensor   
                #tf.keras.utils.save_img("outputs\super_image\super_image.jpg", tf.make_ndarray(super_image))
                #save_img('outputs/super_image/image.png', tf.make_ndarray(super_image),  scale=False)
                #super_image = tf.squeeze(super_image)
                #plot_image(tf.squeeze(super_image),'Super Resolution')   


        # Display the resulting image (Can be removed later)
        cv2.imshow('Video', frame)


        #################### Writing frames to VideoWriter ############
        result.write(frame)

        ####################Skip Frame CODE (Should be removed later)###################
        count += 15 # i.e. at 15 fps, this advances one second
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, count)
        ######################################################
    else:
        video_capture.release()
        break

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime
    print("FPS: ", fps)



# Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release handle to the webcam
result.release()
video_capture.release()
cv2.destroyAllWindows()






# IMAGE_PATH = face



# def save_image(image,filename):
#    ''' 
#     Saves unscaled Tensor Images
#     image: 3D image Tensor
#     filename: Name of the file to be saved
#    '''
#    if not isinstance(image, Image.Image):
#        image = tf.clip_by_value(image, 0, 255)
#        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
#    image.save('%s.jpg' % filename)
#    print('Saved as %s.jpg' % filename) 


 #%matplotlib inline



#load_image = preprocess_image(IMAGE_PATH)

 # plot original image
# plot_image(tf.squeeze(load_image),title='Original Photo')

# # Start Performing resolution 
# start = time.time()
# super_image = model(load_image)
# super_image = tf.squeeze(super_image)
# print('Time taken to complete process: %f'%(time.time() - start))
#  #plot the output image 
# plot_image(tf.squeeze(super_image),'Super Resolution') 