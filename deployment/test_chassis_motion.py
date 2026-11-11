"""Bounded ground commissioning through shared geometry and independent serial.

Default previews commands only. Explicit --execute --floor-ready runs ONE axis.
Parent deadline and stop-only cleanup cover worker termination.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/home/pi/docker/tmp/mentorpi_nav')
sys.path.insert(0,str(ROOT/'runtime'))
from drivetrain import DriveConfig
from rrc_serial import SerialBoard

VELOCITIES={'forward':(.05,0.,0.),'reverse':(-.05,0.,0.),
            'left':(0.,.05,0.),'right':(0.,-.05,0.),
            'yaw':(0.,0.,.15),'clockwise':(0.,0.,-.15)}

def stop_only():
    board=SerialBoard()
    try:
        for _ in range(5):
            board.set_motor_speed([[i,0.] for i in range(1,5)])
            time.sleep(.02)
    finally:
        board.port.close()

def worker(axis,turn_rate,duration):
    drive=DriveConfig(**json.loads((ROOT/'config/drivetrain.json').read_text()))
    board=SerialBoard()
    board.enable_reception()
    velocity=VELOCITIES[axis]
    if axis in ('yaw','clockwise'):
        velocity=(0.,0.,turn_rate if axis=='yaw' else -turn_rate)
    result={'axis':axis,'requested_body_velocity':velocity,
            'duration_s':duration,'wheel_targets_rps':drive.wheel_speeds(velocity),
            'trace':[],'physical_response':'pending supervisor observation'}
    try:
        board.set_motor_speed([[i,0.] for i in range(1,5)])
        imu_count=0
        biases=[]
        battery=None
        deadline=time.monotonic()+8
        last_imu=float('-inf')
        while time.monotonic()<deadline:
            imu=board.get_imu()
            volts=board.get_battery()
            if imu is not None:
                biases.append(imu[5]);imu_count+=1;last_imu=time.monotonic()
            if volts is not None:battery=volts
            if imu_count>=10 and battery is not None and time.monotonic()-last_imu<.15:
                break
            time.sleep(.005)
        if imu_count<5 or time.monotonic()-last_imu>.25 or battery is None or battery<6400:
            raise RuntimeError('Board telemetry not ready: '+str((imu_count,battery)))
        result['battery_before_mv']=battery
        result['ready_imu_samples']=imu_count
        bias=sum(biases)/len(biases)
        body,wheels=(0.,)*3,(0.,)*4
        started=previous=time.monotonic()
        integrated_yaw=0.
        last_rate=0.
        imu_pulse_samples=0
        while time.monotonic()-started<duration:
            now=time.monotonic();dt=min(.05,now-previous);previous=now
            imu=board.get_imu()
            volts=board.get_battery()
            if imu is not None:
                imu_pulse_samples+=1
                last_imu=now
                last_rate=math.radians(imu[5]-bias)
            integrated_yaw+=last_rate*dt
            if volts is not None:battery=volts
            if now-last_imu>.25 or battery<6400:
                raise RuntimeError('Board telemetry stale or battery below threshold')
            body=drive.advance_body(body,velocity,dt)
            wheels=drive.advance(wheels,drive.wheel_speeds(body),dt)
            board.set_motor_speed([[i+1,v] for i,v in enumerate(wheels)])
            result['trace'].append({'elapsed_s':now-started,'wheel_rps':wheels,'inferred_body_velocity':drive.body_velocity(wheels)})
            time.sleep(.02)
        result.update(command_completed=True,battery_after_mv=battery,
                      pulse_imu_samples=imu_pulse_samples,
                      approximate_bias_corrected_imu_yaw_rad=integrated_yaw)
    except Exception as exc:
        result.update(command_completed=False,error=str(exc))
    finally:
        try:
            for _ in range(5):
                board.set_motor_speed([[i,0.] for i in range(1,5)])
                time.sleep(.02)
        finally:
            board.port.close()
        (ROOT/'diagnostics'/f'chassis_{axis}.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k!='trace'},indent=2),flush=True)
    return 0 if result.get('command_completed') else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--axis',choices=VELOCITIES,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--floor-ready',action='store_true')
    parser.add_argument('--turn-rate',type=float,default=.15,help='Yaw speed magnitude in rad/s, at most 0.5')
    parser.add_argument('--duration',type=float,default=.8,help='Pulse duration, at most 0.8 seconds')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not math.isfinite(args.turn_rate) or not .1<=args.turn_rate<=.5 or not math.isfinite(args.duration) or not .1<=args.duration<=.8:
        parser.error('Turn rate must be 0.1..0.5 rad/s; duration 0.1..0.8 seconds')
    if args.axis in ('yaw','clockwise') and args.turn_rate*args.duration>.200001:
        parser.error('Turn pulse must request at most 0.2 radians (11.5 degrees)')
    drive=DriveConfig(**json.loads((ROOT/'config/drivetrain.json').read_text()))
    if not args.execute:
        velocity=VELOCITIES[args.axis]
        if args.axis in ('yaw','clockwise'):velocity=(0.,0.,args.turn_rate if args.axis=='yaw' else -args.turn_rate)
        print(json.dumps({'axis':args.axis,'body_velocity':velocity,
                          'wheel_rps_channels_1_to_4':drive.wheel_speeds(velocity),
                          'duration_s':args.duration,'nonzero_commands_sent':0},indent=2))
        return 0
    if not args.floor_ready:
        raise RuntimeError('Clear floor, slack tether and supervisor required: --floor-ready')
    if args.worker:return worker(args.axis,args.turn_rate,args.duration)
    import fcntl
    lock=(ROOT/'hardware.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for service in ('start_node.service','mentorpi-navigation.service','wifi.service'):
        if subprocess.run(['systemctl','is-active','--quiet',service]).returncode==0:
            raise RuntimeError('Conflicting service: '+service)
    child=subprocess.Popen([sys.executable,__file__,'--axis',args.axis,'--execute','--floor-ready','--worker',
                            '--turn-rate',str(args.turn_rate),'--duration',str(args.duration)])
    try:
        return child.wait(timeout=12)
    finally:
        if child.poll() is None:
            child.kill();child.wait()
        stop_only()
        print('PARENT_STOP: zero motor packets sent',flush=True)

if __name__=='__main__':
    raise SystemExit(main())
