"""Record a local webcam with fresh snapshots; never sends robot commands."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--seconds',type=float,default=60.)
    parser.add_argument('--fps',type=float,default=15.)
    parser.add_argument('--output',required=True)
    parser.add_argument('--stop-file',help='Optional file whose appearance ends recording')
    args=parser.parse_args()
    if not 0 < args.seconds <= 600 or not 0 < args.fps <= 60:
        parser.error('Use a positive duration <=600 s and FPS <=60')
    import cv2
    import imageio_ffmpeg
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    ready=output/'ready.json'
    cap=cv2.VideoCapture(args.camera,cv2.CAP_DSHOW)
    writer=None;frames=0;started=time.monotonic();capture_epoch=None;last_snapshot=float('-inf')
    try:
        if not cap.isOpened():raise RuntimeError('Webcam could not be opened')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        ok,frame=cap.read()
        if not ok:raise RuntimeError('Webcam did not return a frame')
        height,width=frame.shape[:2]
        writer=imageio_ffmpeg.write_frames(str(output/'observer.mp4'),(width,height),
            fps=args.fps,pix_fmt_in='bgr24',pix_fmt_out='yuv420p',codec='libx264',
            macro_block_size=2,ffmpeg_timeout=5,output_params=['-movflags','+faststart'])
        writer.send(None)
        started=time.monotonic()
        capture_epoch=time.time()
        while time.monotonic()-started<args.seconds:
            tick=time.monotonic()
            if args.stop_file and Path(args.stop_file).exists():break
            if frames:
                ok,frame=cap.read()
                if not ok:raise RuntimeError('Webcam frame stream lost')
            writer.send(frame.tobytes());frames+=1
            if tick-last_snapshot>=.5:
                image=output/'latest.tmp.jpg'
                if not cv2.imwrite(str(image),frame):raise RuntimeError('Snapshot write failed')
                image.replace(output/'latest.jpg')
                stamp={'pid':__import__('os').getpid(),'epoch':time.time(),'frame':frames,'camera':args.camera,
                       'width':width,'height':height,'note':'Fresh capture only; framing must be inspected before motion.'}
                tmp=output/'ready.tmp';tmp.write_text(json.dumps(stamp));tmp.replace(ready)
                last_snapshot=tick
            time.sleep(max(0.,1/args.fps-(time.monotonic()-tick)))
    finally:
        if ready.exists():ready.unlink()
        cap.release()
        if writer is not None:writer.close()
        (output/'summary.json').write_text(json.dumps({'frames':frames,'elapsed_s':time.monotonic()-started,
            'camera':args.camera,'capture_started_epoch':capture_epoch,'encoded_fps':args.fps,
            'robot_commands_sent':False},indent=2))
    print(json.dumps({'output':str(output),'frames':frames}),flush=True)


if __name__=='__main__':main()
