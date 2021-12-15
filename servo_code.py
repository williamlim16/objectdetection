import requests
from gpiozero import Servo
from time import sleep

# url = 'https://trash-separator-api.herokuapp.com/node/sendLog'
# test_object = {
#     'trash_can_id': 1,
#     'type': 'plastic',
#     'category': 'organic',
# }

# x = requests.post(url, test_object)
# print(x.text)

# servo = Servo(25,0,min_pulse_width=0.5, max_pulse_width=3.0, frame_width=20.0)

# try:
#         while True:
#             servo.mid()
#             print(servo)
#             sleep(0.5)
           
# except KeyboardInterrupt:
#     print("Program stopped")

import RPi.GPIO as GPIO
import time

servoPIN = 25
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)

p = GPIO.PWM(servoPIN, 50) # GPIO 17 for PWM with 50Hz
p.start(2.5) # Initialization
try:
  while True:
    p.ChangeDutyCycle(95)
    time.sleep(5)
    p.ChangeDutyCycle(100)
    time.sleep(5)
    p.ChangeDutyCycle(10)
    time.sleep(5)
   
 
except KeyboardInterrupt:
  p.stop()
  GPIO.cleanup()