"""Capture the sole enumerated local Logitech C920; camera remains read-only."""
import argparse
from pathlib import Path
import time
import cv2

parser=argparse.ArgumentParser()
parser.add_argument('--index',type=int,default=0)
parser.add_argument('--output',default='logs/robot_deployment/observer.jpg')
args=parser.parse_args()
camera=cv2.VideoCapture(args.index,cv2.CAP_MSMF)
if not camera.isOpened():
    camera.release()
    camera=cv2.VideoCapture(args.index,cv2.CAP_ANY)
try:
    if not camera.isOpened():raise RuntimeError('Logitech camera could not be opened')
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    deadline=time.monotonic()+5
    frame=None
    while time.monotonic()<deadline:
        success,current=camera.read()
        if success:frame=current
        time.sleep(.05)
    if frame is None:raise RuntimeError('Camera returned no image')
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(target),frame):raise RuntimeError('Image write failed')
    print(str(target.resolve()))
finally:
    camera.release()
