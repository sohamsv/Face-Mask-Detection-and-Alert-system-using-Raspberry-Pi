# -*- coding:utf-8 -*-
import cv2
import time
import argparse
import asyncio
import os
import pyimgur

import requests
import json

import smtplib
from email.message import EmailMessage
import imghdr
from tkinter import *
from ttkthemes import ThemedTk
from tkinter import ttk


import numpy as np
from PIL import Image
#from keras.models import model_from_json
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.tensorflow_loader import load_tf_model, tf_inference
from dotenv import load_dotenv
load_dotenv()

sess, graph = load_tf_model('models/face_mask_detection.pb')
# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}
count_unmask = 0

slack_token = os.environ.get("SLACK_TOKEN")
slack_channel = os.environ.get("SLACK_CHANNEL")
count_threshold = int(os.environ.get("COUNT_THRESHOLD"))
confidence_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD"))



def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(160, 160),
              draw_result=True,
              show_result=True
              ):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size.
    :param draw_result: whether to daw bounding box to the image.
    :param show_result: whether to display the image.
    :return:
    '''
    
    
    # image = np.copy(image)
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)
    y_bboxes_output, y_cls_output = tf_inference(sess, graph, image_exp)

    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)
    #  threshold to capture
    global count_unmask

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)
        
        
        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
                count_unmask = 0
            else:
                color = (255, 0, 0)
                count_unmask += 1
                
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
            print(count_unmask)
            if class_id == 1 and count_unmask >= count_threshold:
               asyncio.run(write_image(image))
               #loop.run_in_executor(None, write_image, image)
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])
        

    if show_result:
        Image.fromarray(image).show()
    return output_info
    




def run_on_video(video_path, output_video_name, conf_thresh):

    cap = cv2.VideoCapture(video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # writer = cv2.VideoWriter(output_video_name, fourcc, int(fps), (int(width), int(height)))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened():
        raise ValueError("Video open failed.")
        return
    
    status = True
    idx = 0
    
    while status:
        start_stamp = time.time()
        status, img_raw = cap.read()
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        read_frame_stamp = time.time()
        if (status):
            inference(img_raw,
                      conf_thresh,
                      iou_thresh=0.5,
                      target_shape=(260, 260),
                      draw_result=True,
                      show_result=False)
            cv2.imshow('mask_detection', img_raw[:, :, ::-1])
            cv2.waitKey(1)
            inference_stamp = time.time()

            write_frame_stamp = time.time()
            idx += 1
            print("%d of %d" % (idx, total_frames))
            print("read_frame:%f, infer time:%f, write time:%f" % (read_frame_stamp - start_stamp,
                                                                   inference_stamp - read_frame_stamp,
                                                                   write_frame_stamp - inference_stamp))

        
    

    

async def write_image(image):
    global count_unmask
    
    millis = int(round(time.time() * 1000))
    filename = str(millis)+'_image.png'
    filename_path = 'unmask/'+filename
    
    cv2.imwrite(filename_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    #cv2.imwrite('violation.jpg',cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    count_unmask = 0
    

    #with open(filename_path, "rb") as img:
      #f = img.read()
      #b = bytearray(f)
      
    #result = post_file_to_slack('we found people which not using face mask', filename, b)
    e_count = 0
    if e_count == 0 :
        EMAIL_ADDRESS = ''
        EMAIL_PASSWORD = ''
        contacts = ['', '']
        msg = EmailMessage()
        msg['Subject'] = 'Person Found Violating Face Mask Policy'
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = ''
        msg.set_content('Face Mask Policy Violation')
        msg.add_alternative("""\
                <!DOCTYPE html>
                <html>
                    <body>
                        <h3 style="color:SlateGray;">A person was found violating the face mask policy in our premises!!!</h3>
                    </body>
                </html>
                """, subtype='html')
        with open(filename_path, 'rb') as f:
            file_data = f.read()
            file_type = imghdr.what(f.name)
            file_name = f.name
            msg.add_attachment(file_data, maintype='image', subtype=file_type, filename=file_name)
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                smtp.send_message(msg)
                e_count = 1
        e_count = 1
    
    #print(result)

def post_file_to_slack(text, file_name, file_bytes, file_type=None, title=None):
    return requests.post(
      'https://slack.com/api/files.upload', 
      {
        'token': slack_token,
        'filename': file_name,
        'channels': slack_channel,
        'filetype': file_type,
        'initial_comment': text,
        'title': title
      },
      files = { 'file': file_bytes }).json()

def btnclick():
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Face Mask Detection")
        parser.add_argument('--img-mode', type=int, default=1, help='set 1 to run on image, 0 to run on video.')
        parser.add_argument('--img-path', type=str, help='path to your image.')
        parser.add_argument('--video-path', type=str, default='0', help='path to your video, `0` means to use camera.')
        # parser.add_argument('--hdf5', type=str, help='keras hdf5 file')
        args = parser.parse_args()

        
        
        #if args.img_mode:
            #imgPath = args.img_path
            #img = cv2.imread(imgPath)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #inference(img, show_result=True, target_shape=(260, 260))
        #else:
        video_path = args.video_path
        if args.video_path == '0':
            video_path = 0
        run_on_video(video_path, '', conf_thresh=confidence_threshold)
        

def stopbtn():
    status = False


root  = ThemedTk()
# mainFrame = Main(root)
root.geometry("505x450+100+100")
#geometry(width * height + x-position on screen + y-position on screen)
root.title('Mask Detector')
root.configure(bg = 'white')


#Label Heading
labelHead = ttk.Label(root, text = 'Mask Detector',borderwidth=6 ,relief="sunken",
                            background = 'snow2', font = ("arial", 20) )
labelHead.config(anchor = CENTER)
labelHead.pack(pady = 15)

btnStart = ttk.Button(root, text = 'Start Detection', width = 15, command=btnclick)
btnStart.place(x = 200, y = 180)

#btnConfig = ttk.Button(root, text = 'Config.', width = 15, command=configMenu)
#btnConfig.place(x = 250, y = 80)
#btnStop = ttk.Button(root, text = 'Stop Detection', width = 12, command=stopbtn)
#btnStop.place(x = 240, y = 80)

#VideoFrame
#Create Lable for image (Video PlaceHolder)
labelDisplay = ttk.Label(root)
labelDisplay.place(x = 10, y = 120)

root.mainloop()