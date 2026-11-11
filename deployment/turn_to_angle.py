"""Supervised IMU-relative turn; defaults to preview, with hard stop limits."""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path('/home/pi/docker/tmp/mentorpi_nav')
sys.path.insert(0,str(ROOT/'runtime'))
from drivetrain import DriveConfig
from rrc_serial import SerialBoard
from turn_control import turn_decision

def stop():
    board=SerialBoard()
    try:
        for _ in range(5):
            board.set_motor_speed([[i,0.] for i in range(1,5)]);time.sleep(.02)
    finally:board.port.close()

def worker(degrees):
    drive=DriveConfig(**json.loads((ROOT/'config/drivetrain.json').read_text()))
    board=SerialBoard();board.enable_reception()
    result={'target_degrees':degrees,'trace':[],'passed':False}
    yaw=0.
    try:
        board.set_motor_speed([[i,0.] for i in range(1,5)])
        bias_samples=[];battery=None;last_imu=float('-inf')
        start=time.monotonic()
        while time.monotonic()-start<8:
            imu=board.get_imu();volts=board.get_battery();now=time.monotonic()
            if imu is not None:bias_samples.append(imu[5]);last_imu=now
            if volts is not None:battery=volts
            if now-start>=1 and len(bias_samples)>=20 and battery is not None and now-last_imu<.15:break
            time.sleep(.005)
        if len(bias_samples)<20 or battery is None or time.monotonic()-last_imu>.25:
            raise RuntimeError('Fresh stationary IMU/battery not available')
        if statistics.pstdev(bias_samples)>1.:
            raise RuntimeError('IMU baseline unstable; keep robot still before test')
        bias=statistics.mean(bias_samples)
        result.update(gyro_bias_degrees_per_second=bias,battery_before_mv=battery)
        target=math.radians(degrees);commanded=0.;rate=0.
        body=(0.,)*3;wheels=(0.,)*4
        started=previous=time.monotonic()
        print('TURN_STARTED: target '+str(degrees)+' degrees; maximum 4 seconds',flush=True)
        while True:
            now=time.monotonic();dt=now-previous;previous=now
            imu=board.get_imu();volts=board.get_battery()
            if imu is not None:last_imu=now;rate=math.radians(imu[5]-bias)
            if volts is not None:battery=volts
            yaw+=rate*dt
            status,speed=turn_decision(target,yaw,now-started,commanded,now-last_imu,battery)
            if status!='turning':
                result.update(stop_reason=status,passed=status=='target_reached')
                break
            body=drive.advance_body(body,(0.,0.,speed),min(dt,.05))
            wheels=drive.advance(wheels,drive.wheel_speeds(body),min(dt,.05))
            if max(map(abs,wheels))>.6:
                scale=.6/max(map(abs,wheels));wheels=tuple(v*scale for v in wheels)
            board.set_motor_speed([[i+1,v] for i,v in enumerate(wheels)])
            commanded+=abs(drive.body_velocity(wheels)[2])*dt
            result['trace'].append({'elapsed_s':now-started,'imu_degrees':math.degrees(yaw),'wheel_rps':wheels})
            time.sleep(.02)
        result.update(motor_active_seconds=time.monotonic()-started,
                      commanded_angle_degrees=math.degrees(commanded),battery_after_mv=battery)
    except Exception as exc:
        result.update(stop_reason='error',error=str(exc))
    finally:
        try:
            for _ in range(5):board.set_motor_speed([[i,0.] for i in range(1,5)]);time.sleep(.02)
        finally:board.port.close()
        result['estimated_yaw_degrees']=math.degrees(yaw)
        result['angle_source']='Integrated stationary-bias-corrected IMU, not external angle measurement'
        (ROOT/'diagnostics/turn_to_angle.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k!='trace'},indent=2),flush=True)
    return 0 if result['passed'] else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--degrees',type=float,default=45.)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--floor-ready',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not math.isfinite(args.degrees) or not 1<=abs(args.degrees)<=45:parser.error('Angle magnitude must be 1..45 degrees')
    if not args.execute:
        print(json.dumps({'target_degrees':args.degrees,'max_motor_seconds':4,'max_wheel_rps':.6,'commanded_angle_cap_degrees':90,'nonzero_commands_sent':0}))
        return 0
    if not args.floor_ready:raise RuntimeError('Clear floor and slack tether required: --floor-ready')
    if args.worker:return worker(args.degrees)
    import fcntl
    lock=(ROOT/'hardware.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for service in ('mentorpi-navigation.service','start_node.service','wifi.service'):
        if subprocess.run(['systemctl','is-active','--quiet',service]).returncode==0:
            raise RuntimeError('Stop conflicting service before turn: '+service)
    child=subprocess.Popen([sys.executable,__file__,'--degrees',str(args.degrees),'--execute','--floor-ready','--worker'])
    try:return child.wait(timeout=16)
    finally:
        if child.poll() is None:child.kill();child.wait()
        stop();print('PARENT_STOP: explicit motor-zero packets sent',flush=True)

if __name__=='__main__':raise SystemExit(main())
