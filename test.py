import time, math, argparse, cv2, sys, torch
import numpy as np
from mtcnn import MTCNN
from PIL import Image
import pickle
import time
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
import pandas as pd

import os
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from IPython.display import FileLink, FileLinks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Sample class contains: Constructors, Preprocessing functions, Models(Face Detection, ESRGANs, Levi and Hassner Model, Chotu Model)
class Sample:

    def __init__(self, args):

        self.args = args

        # classes for the age and gender category
        self.ageList = ['(0-3)', '(4-7)', '(8-13)', '(14-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.ages = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(21-24)", "(25-32)",
                     "(33-37)", "(38-43)", "(44-47)", "(48-53)", "(54-59)", "(60-100)"]
        self.genders = ["Male", "Female"]

        # loading face detector pretrained model
        # faceProto = "../models/face_detector/opencv_face_detector.pbtxt"
        # faceModel = "../models/face_detector/opencv_face_detector_uint8.pb"
        # self.faceNet = cv2.dnn.readNet(faceModel, faceProto)
        self.detector = MTCNN()



####################### Check These ###############################
        # age detector pretrained model  
        ageProto = "../models/age_detector/age_deploy.prototxt"
        ageModel = "../models/age_detector/age_net.caffemodel"
        self.ageNet = cv2.dnn.readNet(ageModel, ageProto)

####################### Check These ###############################
        # gender detector pretrained model
        genderProto = "../models/gender_detector/gender_deploy.prototxt"
        genderModel = "../models/gender_detector/gender_net.caffemodel"
        self.genderNet = cv2.dnn.readNet(genderModel, genderProto)

        # model mean values to subtract from facenet model
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

####################### Check These ###############################
        SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
        LOCAL_MODEL_PATH = "../models/ESRGANs/_"
        #"D:\IIT\Inter IIT Tech Meet\BOSCH Mid Prep\MP_BO_T1\MP_BO_T1_CODE\models\ESRGANs\_\saved_model.pb"
        #Loading ESRGANs model from tensorflow hub
        self.esrgan_model = hub.load(LOCAL_MODEL_PATH)

        age_path = 'saved_chotu_model (1).h5'
        self.age_model = tf.keras.models.load_model(age_path, compile=False)

    ###################### Function to preprocess image so that it can be handled by ESRGANs model ###############################
    def preprocess_image(self, face):
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

    @staticmethod

    def detect_face(detector, img):    #
        
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
    
    ######## Chotu model for age and gender #############
    def predict_age_gender(self,image):
        img = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
        img = cv2.resize(img , (48,48))
        img = img/255
        pred = self.age_model.predict(np.array([img]))
        age=int(np.round(pred[1][0]))
        sex=int(np.round(pred[0][0]))

        return age , sex

    def caffeInference(self):
        ################ VideoCapture, should be changed later to a parser input ########################
        video_capture = cv2.VideoCapture(self.args.input if self.args.input else 0)

        ###################### Saving the video in the local device ##############################
        frame_width = int(video_capture.get(3))
        frame_height = int(video_capture.get(4))
        size = (frame_width , frame_height)
        ####Extracting output filename from the terminal parser input #######
        path = "Final_outputs/" + self.args.input.split(chr(92))[-1]
        #print(self.args.input.split(chr(92))[-1])
        #print(path)
        path = path.split('.')[:-1]
        #print(path)
        path_1 = ""
        for i in path:
            path_1 = path_1 + i
        #print(path_1)


        result = cv2.VideoWriter(path_1 + ".avi", cv2.VideoWriter_fourcc(*('MJPG')), 10 , size)

        # ############# Initializing the skip frame count ##################
        count = 0  

        ##### Creating a dataframe to save csv file ########
        output_df = pd.DataFrame(columns=['frame_num', 'person_id', 'bb_xmin', 'bb_ymin', 'bb_height', 'bb_width', 'age_min', 'age_max',
             'age_actual', 'gender'])  

        while video_capture.isOpened():
            # Grab a single frame of video
            ret, frame = video_capture.read()    
            #start = time.time()
            #cv2.imshow("output1", frame)
            ##### 'j' records the frame count ######
            j=0
            # sex_list = ['male' , 'female']
            if ret:
                # cv2.imwrite('frame{:d}.jpg'.format(count), frame)
                # count += 30 # i.e. at 30 fps, this advances one second
                # cap.set(cv2.CAP_PROP_POS_FRAMES, count)

                face_locations, faces_in_frame_array = self.detect_face(self.detector , frame)   #face_location = List of bounding box coordinates of detected faces in a frame.
                                                                            #faces_in_frame_array = List of cropped face pixels of detected faces in a frame.

                label_list = []

                # for top, right, bottom, left in face_locations:             
                #     # Draw a box around the face
                #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                #faces_in_frame_array = np.array(faces_in_frame_array)

                for face in faces_in_frame_array:

                    load_image = self.preprocess_image(face)

                    super_image = self.esrgan_model(load_image)    #Making a super resolution face tensor

                    super_image_numpy = super_image.numpy() #coverting tensor to numpy array
                    super_image_numpy = super_image_numpy[0] #converting 4-D numpy array to 3-D ignoring the first index as it tells no. of faces only
                    # reshape the face image to 227x227 using the blob function and also swapping the RB planes
                    blob = cv2.dnn.blobFromImage(super_image_numpy, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)

                    self.genderNet.setInput(blob)  # pass the 227x227 reshaped face blob to the gender net for prediction
                    genderPreds = self.genderNet.forward()  # do the forward pass
                    gender = self.genders[genderPreds[0].argmax()]  # get the gender
                    gender_final= gender[0]

                    # print the gender along with its confidence score
                    #print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

                    # predict Age
                    self.ageNet.setInput(blob)  # pass the 227x227 face blob to the age net.
                    agePreds = self.ageNet.forward()  # do forward pass
                    age = self.ageList[agePreds[0].argmax()]  # get the age range

                    # print the age along with its confidence score
                    #print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))
                    #print(age)
                    age_2 , sex_2 = self.predict_age_gender(super_image_numpy)
                    
                    age_list = age.split("-")
                    #print(age_list)
                    #print((int(age_list[0][1:]) + int(age_list[1][:-1]))/2)
                    age = (int(age_list[0][1:])+int(age_list[1][:-1]))/2
                    #print(agePreds[0].max())
                    #print(type(agePreds[0].max()))
                    age_final = (age*agePreds[0].max())+((1-agePreds[0].max())*age_2)
                    age_final = round(age_final,0)

                    # get the label to display on the image frame
                    label = "{},{}".format(gender, age_final)
                    label_list.append(label)
                    output_df = output_df.append({'frame_num': count, 'person_id': j, 'bb_xmin': face_locations[j][3],
                                              'bb_ymin': face_locations[j][2],
                                              'bb_height': -(face_locations[j][0] - face_locations[j][2]),
                                              'bb_width': (face_locations[j][1] - face_locations[j][3]),
                                              'age_actual':age_final,'gender':gender_final},
                                               ignore_index=True)
                    j=j+1
                    # place the text on the image with squared bounding box
                    #cv2.putText(frame, label, (face_locations[top], face_locations[right] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),2,cv2.LINE_AA)
                    #print("age = " , age_final , " gender = ", gender)

                    if self.args.output != "":
                        filename = "output/predictions/" + str(args.output)
                        cv2.imwrite(filename, frame)
                    #cv2.imshow("Age Gender Demo", frame)

                i = 0
                for top, right, bottom, left in face_locations:             
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                    cv2.putText(frame, label_list[i], (right-14, top), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,cv2.LINE_AA)
                    i=i+1

                #print("time : {:.3f}".format(time.time() - t))

                ################ESRGANs Implementation#####################
                #if len(faces_in_frame_array) == 0:
                    #continue
                #else:
                    #for face in faces_in_frame_array:    #"face" iterator of the face_in_frame_array over here represents cropped face image pixels
                        #face = np.array(face)
                        #load_image = preprocess_image(face)    
                        #plot_image(tf.squeeze(load_image),title='Original Photo')    
                        #save_img('outputs/image/image.png', tf.make_ndarray(load_image),  scale=False)
                        #tf.keras.utils.save_img("outputs\image\image.jpg", tf.make_ndarray(load_image))
                        #super_image = model(load_image)    #Making a super resolution face tensor   
                        #tf.keras.utils.save_img("outputs\super_image\super_image.jpg", tf.make_ndarray(super_image))
                        #save_img('outputs/super_image/image.png', tf.make_ndarray(super_image),  scale=False)
                        #super_image = tf.squeeze(super_image)
                        #plot_image(tf.squeeze(super_image),'Super Resolution')

                        #super_image_numpy = super_image.numpy() #coverting tensor to numpy array
                        #super_image_numpy = super_image_numpy[0] #converting 4-D numpy array to 3-D ignoring the first index as it tells no. of faces only

                        #age , sex = predict_age_gender(super_image_numpy) 

                        #print("age = ", age , " sex = ", sex_list[sex])   


                # Display the resulting image (Can be removed later)
                #cv2.imshow('Video', frame)


                #################### Writing frames to VideoWriter ############
                #cv2.imshow("out", frame)
                
                count = count + 1
                ####################Skip Frame CODE (Should be removed later)###################
                #count += 1 # i.e. at 15 fps, this advances one second
                #video_capture.set(cv2.CAP_PROP_POS_FRAMES, count)
                ######################################################

                result.write(frame)

            else:
                video_capture.release()
                break

            # end = time.time()
            # totalTime = end - start

            # fps = 1 / totalTime
            # print("FPS: ", fps)
            # print (output_df)
            output_df.to_csv(path_1 + '.csv', index=False) 
         

parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('-i', '--input', type=str,
                    help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument('-o', '--output', type=str, default="",
                    help='Path to output the prediction in case of single image.')

args = parser.parse_args()
s = Sample(args)
s.caffeInference()


