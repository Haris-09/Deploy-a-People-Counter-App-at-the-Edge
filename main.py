"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    counter = 0
    for box in result[0][0]:
        # Checking if the person is detected and confidence is greater than the threshold
        if box[2] >= args.prob_threshold and int(box[1]) == 1:
            # coordinates of the top left bounty box
            pt1 = (int(box[3] * width), int(box[4] * height))
            # coordinates of the bottom right bounty box
            pt2= (int(box[5] * width), int(box[6] * height))
            # Drawing rectangle on the person 
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
            counter += 1
    return frame, counter

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    
    #initialize parameters
    total_count = 0
    last_count = 0
    n_frames = 0
    total_persons = []
    detected = False

    ### TODO: Load the model through `infer_network` ###
    
    # Load the network model into the IE
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    
    ### TODO: Handle the input stream ###
    # Create a flag for single images
    image_flag = False
    # Check if the input is a webcam
    if args.input == 'CAM':
        args.input = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        image_flag = True
    
    # Get and open video capture
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag,frame = cap.read()
        if not flag:
            break
        #key_pressed = cv2.waitkey(60)
        n_frames += 1

        ### TODO: Pre-process the image as needed ###
        # Pre-process the frame
        p_frame = cv2.resize(frame,(net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        # Start timer
        start_time = time.time()
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(p_frame)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            #calculate time
            total_time = time.time()-start_time

            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
            # Get Inference Time
            infer_time = "Inference Time: {:.3f}ms".format(total_time * 1000)

            ### TODO: Extract any desired stats from the results ###
            out_frame, counter = draw_boxes(frame, result, args, width, height)
            total_persons.append(counter)
            
            # Get a writen text on the video
            cv2.putText(out_frame, "Counted Number: {} ".format(counter),
                        (20, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(out_frame, infer_time, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            if counter > last_count and detected == False:
                start = n_frames
                total_count = total_count + counter - last_count
                detected = True
                # Topic "person": keys of "total"
                client.publish("person", json.dumps({"total": total_count}))
            
            if counter == 0:                
                # Check if a person is detected in the current frame and no person was detected in the last 29 frames
                if (detected and all(p == 0 for p in total_persons[-29:])):
                    detected = False 
                    # Check if there was a person detected before the last 29 frames 
                    if(total_persons[-30] == 1):
                        # Substract the start_time and the last 30 frames from the current frame_num
                        end_frames = n_frames - start - 30
                        # Divide the total frames in which person is present by fps to get the duration in seconds
                        duration = int((end_frames)/fps)
                        
                        ### Topic "person/duration": key of "duration" ### 
                        client.publish("person/duration", json.dumps({"duration": duration}))
            
            # Topic "person": key of "count"
            client.publish("person", json.dumps({"count": counter}))
            last_count = counter

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(out_frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if image_flag:
            cv2.imwrite('output_image.jpg',out_frame)
        # Break if escape key pressed
        #if key_pressed == 27:
            #break

    # Release the capture and destroy any OpenCV windows
    cap.release()
    #cv2.destryAllWindows()
    client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
