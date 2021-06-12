import streamlit as st
import detection as dtc
import imutils
import datetime
import cv2
import os
import pandas as pd
import torch

# GUI header
st.set_page_config("Social Distancing Detector", None, "wide")
st.title('Automated Social Distancing Monitoring System :walking::walking:')
st.subheader('COS30018 Assignment Project Topic 3 - Social Distancing Monitoring System')

st.write("Do you want to use NVIDIA CUDA GPU? :desktop_computer:")

cuda = st.selectbox('', ('No', 'Yes'))

if cuda == "Yes":
    print(1)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'weights/best.pt')
    model.cuda('cuda:0')
elif cuda == "No":
    print(0)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'weights/best.pt')
    


st.write("")

st.write("Minimum confidence for the model to detect human")
MIN_CONF = st.slider('', 0.0, 1.0, 0.7)

st.subheader('Image detector :frame_with_picture:')
st.write('Upload your image to be analysed')
st.write('Note: Make sure the file is located in the images folder')
img = st.file_uploader('Image')


MIN_CONF = float(MIN_CONF)
DISTANCE = 40

if img is not None:
    path = img.name
    img = cv2.imread('images/' + path)
    img, violation = dtc.bird_detect_people_on_frame(img, MIN_CONF, DISTANCE, img.shape[1], img.shape[0], model)
    st.success("Image successfully analysed")
    st.image(img, caption='Image detection result')

st.subheader('Video detector :film_frames:')
st.write('Upload your video to be analysed')
st.write('Note: Make sure the file is located in the videos folder')
vid = st.file_uploader('Video')

if vid is not None:
    path = vid.name
    vid = "videos/" + path

    video = cv2.VideoCapture(vid)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    violations = dtc.bird_detect_people_on_video(vid, MIN_CONF, DISTANCE, model)

    path = 'bird_output.avi'
    compressed_path = path.split('.')[0]
    compressed_path = 'compressed_' + compressed_path + '.mp4'
    if os.path.exists(compressed_path):
        os.remove(compressed_path)
    # Convert video
    os.system(f"ffmpeg -i {path} -vcodec libx264 -nostdin {compressed_path} ")
    print(compressed_path)
    
    st.success("Video successfully analysed")
    st.video('compressed_bird_output.mp4', 'video/mp4', 0)

    # add graphs
    FPS = video.get(cv2.CAP_PROP_FPS)
    
    data = pd.DataFrame({'Violations':violations})
    st.subheader('Violations for each frame in the video :warning:')
    st.line_chart(data, height = 500)

                        
st.info("Design & Developed By MACE For COS30018 Intelligent Systems Assignment")                 
