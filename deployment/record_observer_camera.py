"""Bounded Logitech observation recording for physical commissioning evidence."""
import argparse
import json
import os
from pathlib import Path
import time
import cv2

parser=argparse.ArgumentParser()
parser.add_argument('--duration',type=float,default=35.)
parser.add_argument('--output',default='logs/robot_deployment/commission_observer')
args=parser.parse_args()
target=Path(args.output)
target.parent.mkdir(parents=True,exist_ok=True)
ready_file=target.with_suffix('.ready.json')
camera=cv2.VideoCapture(0,cv2.CAP_MSMF)
if not camera.isOpened():
    camera.release()
    camera=cv2.VideoCapture(0,cv2.CAP_ANY)
if not camera.isOpened():raise RuntimeError('Logitech camera unavailable')
camera.set(cv2.CAP_PROP_FRAME_WIDTH,1280);camera.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
writer=None
frames=[]
started=time.monotonic();next_save=started
try:
    while time.monotonic()-started<args.duration:
        ok,frame=camera.read()
        if not ok:continue
        elapsed=time.monotonic()-started
        if writer is None:
            height,width=frame.shape[:2]
            writer=cv2.VideoWriter(str(target.with_suffix('.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),10.,(width,height))
            if not writer.isOpened():raise RuntimeError('Observer video writer failed to open')
            print('OBSERVER_READY',flush=True)
        ready_file.write_text(json.dumps({'pid':os.getpid(),'elapsed_s':elapsed}))
        if time.monotonic()>=next_save:
            filename=target.parent/(target.name+f'_{int(elapsed):03d}.jpg')
            cv2.imwrite(str(filename),frame)
            frames.append({'elapsed_s':elapsed,'file':str(filename)})
            next_save=time.monotonic()+2.
        cv2.putText(frame,f'{elapsed:.1f}s',(20,35),cv2.FONT_HERSHEY_SIMPLEX,1.,(0,255,0),2)
        writer.write(frame)
        time.sleep(.08)
finally:
    camera.release()
    ready_file.unlink(missing_ok=True)
    if writer is not None:writer.release()
    target.with_suffix('.json').write_text(json.dumps(frames,indent=2))
print(str(target.with_suffix('.mp4').resolve()),flush=True)
