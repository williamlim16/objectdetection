# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time

import cv2
from object_detector import Category, ObjectDetector
from object_detector import ObjectDetectorOptions
import utils
import requests
import time 

# LOAD DICTIONARY, 0 - organic. 1 - inorganic
def load_type(): 
  dictionary = {
    "person": 0,
    "bicycle": 1,
    "car": 1,
    "motorcycle": 1,
    "airplane": 1,
    "bus": 1,
    "train": 1,
    "truck": 1,
    "boat": 1,
    "traffic light": 1,
    "fire hydrant": 1,
    "stop sign": 1,
    "parking meter": 1,
    "bench": 1,
    "bird": 0,
    "cat": 0,
    "dog": 0,
    "horse": 0,
    "sheep": 0,
    "cow": 0,
    "elephant": 0,
    "bear": 0,
    "zebra": 0,
    "giraffe": 0,
    "backpack": 1,
    "umbrella": 1,
    "handbag": 1,
    "tie": 1,
    "suitcase": 1,
    "frisbee": 1,
    "skis":1,
    "snowboard": 1,
    "sports ball": 1,
    "kite": 1,
    "baseball bat": 1,
    "baseball glove": 1,
    "skateboard": 1,
    "surfboard": 1,
    "tennis racket": 1,
    "bottle": 1,
    "wine glass": 1,
    "cup": 1,
    "fork": 1,
    "knife": 1,
    "spoon": 1,
    "bowl": 1,
    "banana": 0,
    "apple": 0,
    "sandwich": 0,
    "orange": 0,
    "broccoli": 0,
    "carrot": 0,
    "hot dog": 0,
    "pizza": 0,
    "donut": 0,
    "cake": 0,
    "chair": 1,
    "couch": 1,
    "potted plant": 1,
    "bed": 1,
    "dining table": 1,
    "toilet": 1,
    "tv": 1,
    "laptop": 1,
    "mouse": 1, # mouse apa ini
    "remote": 1,
    "keyboard": 1,
    "cell phone": 1,
    "microwave": 1,
    "oven": 1, 
    "toaster": 1, 
    "sink": 1, 
    "refrigerator": 1, 
    "book": 1, 
    "clock": 1, 
    "vase": 1, 
    "scissors": 1, 
    "teddy bear": 1, 
    "hair drier": 1, 
    "toothbrush": 1
  }
  return dictionary 


# Global variable to prevent consecutive requests
midRequest = False

# Dictionary for organic and inorganic classification
# 0 = organic, 1 = inorganic 
type_dict = load_type()

def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  options = ObjectDetectorOptions(
      num_threads=num_threads,
      score_threshold=0.3,
      max_results=3,
      enable_edgetpu=enable_edgetpu)
  detector = ObjectDetector(model_path=model, options=options)

  # variable for storing label predictions and for counter before sending request
  consecutive_labels = {}

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    # Run object detection estimation using the model.
    detections = detector.detect(image)

    # Draw keypoints and edges on input image
    image = utils.visualize(image, detections)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Try to print detection from detections
    if len(detections) == 0: 
      consecutive_labels.clear() 
      midRequest = False
        
    for detection in detections:
      label = detection.categories[0].label
      consecutive_labels[label] = consecutive_labels.get(label, 0) + 1
      value = consecutive_labels[label]
      print("label [" + label + "] count: " + value)
      if value > 10 and midRequest == False: 
        consecutive_labels.clear() 
        midRequest = True
        doRequest(label)

      # category = detection.categories[0]
      # class_name = category.label 
      # probability = round(category.score, 2)
      # result_detection = class_name + " (" + str(probability) + ")"
      # print("label = " + result_detection)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', image)
    time.sleep(0.1) # sleep for 0.1 second to prevent overheat?


  cap.release()
  cv2.destroyAllWindows()

def doRequest(label: str): 
  print("sending label [" + label + "] to Heroku")
  url = 'https://trash-separator-api.herokuapp.com/node/sendLog'
  category = type_dict.get(label, 0)
  categoryStr = "organic"
  if category == 1:
    categoryStr = "inorganic" 
    
  body = {"trash_can_id": "999", "category": categoryStr, "type": label}

  req = requests.post(url, data=body)
  print(req.text)

  time.sleep(1) #sleep 1 second after sending log
  global midRequest
  midRequest = False

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()
