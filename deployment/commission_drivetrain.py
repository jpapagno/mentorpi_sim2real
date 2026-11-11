"""Interactive operator-run drivetrain checks; no motion until explicit prompts."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/home/pi/docker/tmp/mentorpi_nav')
sys.path.insert(0,str(ROOT/'runtime'))
from rrc_serial import SerialBoard

DESCRIPTIONS={'forward':'forward','reverse':'backward','left':'sideways LEFT',
              'right':'sideways RIGHT','yaw':'COUNTERCLOCKWISE (viewed from above)',
              'clockwise':'CLOCKWISE (viewed from above)'}

def prepare():
    print('Stopping competing services; NetworkManager and Ethernet remain running.',flush=True)
    for service in ('mentorpi-navigation.service','start_node.service','wifi.service'):
        subprocess.run(['systemctl','stop',service],check=True)
    code='''
import os,signal,time
from pathlib import Path
pids=[]
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=path.read_bytes().split(b'\\0')
  factory=b'launch' in args and b'bringup' in args and b'bringup.launch.py' in args
  owned=b'/home/ubuntu/shared/mentorpi_nav/runtime/bringup.launch.py' in args
  if factory or owned:
   pid=int(path.parent.name);os.kill(pid,signal.SIGINT);pids.append(pid)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
deadline=time.monotonic()+8
while time.monotonic()<deadline and any(Path('/proc/'+str(pid)).exists() for pid in pids):time.sleep(.1)
print('Stopped known launch processes:',pids,flush=True)
'''
    subprocess.run(['docker','exec','MentorPi','python3','-c',code],check=True)
    # Refuses to open if any remaining process still owns the board.
    board=SerialBoard()
    try:
        for _ in range(3):
            board.set_motor_speed([[i,0.] for i in range(1,5)])
            time.sleep(.02)
    finally:board.port.close()
    print('Serial port verified free; stop packets sent. Services stay stopped after testing.')

def print_report(report):
    print('\n=== COPY RESULTS BELOW ===')
    print('Wheel mapping: 1=front left(-forward), 2=rear left(-forward), 3=rear right(+forward), 4=front right(+forward)')
    if not report['tests']:print('No chassis tests recorded.')
    for entry in report['tests']:
        raw=entry['telemetry']
        print(f"{entry['axis']}: {entry['observation']}; measured={entry['measurement']} {entry['measurement_unit']}")
        print(f"  command completed={raw.get('command_completed')}; battery={raw.get('battery_before_mv')}->{raw.get('battery_after_mv')} mV; IMU yaw={raw.get('approximate_bias_corrected_imu_yaw_rad')} rad")
        if raw.get('error'):print('  ERROR:',raw['error'])
    print('=== END RESULTS ===')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--axis',choices=DESCRIPTIONS,help='Test a single direction')
    parser.add_argument('--all-directions',action='store_true',help='Test forward, reverse, left, right, CCW and CW')
    parser.add_argument('--report',action='store_true',help='Print saved results without opening the motor port')
    parser.add_argument('--turn-rate',type=float,default=.15,help='Turn rate magnitude in rad/s; maximum 0.5')
    parser.add_argument('--duration',type=float,default=.8,help='Pulse duration; maximum 0.8 seconds')
    args=parser.parse_args()
    if not math.isfinite(args.turn_rate) or not .1<=args.turn_rate<=.5 or not math.isfinite(args.duration) or not .1<=args.duration<=.8:
        parser.error('Turn rate 0.1..0.5 rad/s and duration 0.1..0.8 seconds required')
    if (args.axis in ('yaw','clockwise') or not args.axis) and args.turn_rate*args.duration>.200001:
        parser.error('Turn pulses must request at most 0.2 rad; shorten --duration')
    report_file=ROOT/'diagnostics/chassis_operator_report.json'
    report=json.loads(report_file.read_text()) if report_file.exists() else {'tests':[],
        'robot_clock_note':'Robot wall clock may be wrong; recorded_utc is ordering metadata, not verified real date.',
        'measurements':'Supervisor observations; IMU yaw is a short bias-corrected estimate, not position measurement.'}
    if args.report:
        print_report(report)
        return
    if os.geteuid()!=0:
        raise SystemExit('Run this tester with sudo python3.')
    print(f'Each test lasts {args.duration} seconds, then stops. Travel command <=4 cm; turn request {args.turn_rate*args.duration*180/math.pi:.1f} degrees.')
    print('Robot LEFT/RIGHT means its own left/right when facing forward. Directions of rotation are viewed from ABOVE.')
    print('Place it on a clear, flat FLOOR with slack cables. Watch it; keep hands away from moving wheels.')
    if input('Type FLOOR to confirm, or anything else to cancel: ').strip()!='FLOOR':
        print('Cancelled; no motor commands sent.')
        return
    prepare()
    report['drive_config']=json.loads((ROOT/'config/drivetrain.json').read_text())
    axes=[args.axis] if args.axis else list(DESCRIPTIONS) if args.all_directions else ['forward','left','yaw']
    for axis in axes:
        while True:
            print(f'\nNext: {DESCRIPTIONS[axis]}.',flush=True)
            action=input('Enter=run pulse, s=skip, q=quit: ').strip().lower()
            if action=='q':
                print_report(report)
                return
            if action=='s':break
            if action:continue
            raw_file=ROOT/'diagnostics'/f'chassis_{axis}.json'
            raw_file.unlink(missing_ok=True)
            process=subprocess.run([sys.executable,str(ROOT/'test_chassis_motion.py'),
                '--axis',axis,'--execute','--floor-ready','--turn-rate',str(args.turn_rate),
                '--duration',str(args.duration)],capture_output=True,text=True)
            print(process.stdout)
            if process.stderr:print(process.stderr)
            raw=json.loads(raw_file.read_text()) if raw_file.exists() else {'error':'No diagnostic file created'}
            if process.returncode or not raw.get('command_completed'):
                print('Test failed; no further pulses will be run. Paste the error output into chat.')
                report['tests'].append({'axis':axis,'observation':'test failed','measurement':None,
                    'measurement_unit':'degrees' if axis in ('yaw','clockwise') else 'cm','telemetry':raw})
                report_file.write_text(json.dumps(report,indent=2))
                return
            observation=input('What happened? (example: moved left, no motion, mostly rotated): ').strip()
            unit='degrees' if axis in ('yaw','clockwise') else 'cm'
            while True:
                entered=input(f'Optional measured {unit} along requested direction (negative if opposite), or Enter=unknown: ').strip()
                try:
                    measured=None if not entered else float(entered)
                    if measured is not None and not math.isfinite(measured):raise ValueError()
                    break
                except ValueError:print('Enter a finite number or leave blank.')
            entry={'axis':axis,'observation':observation,'measurement':measured,'measurement_unit':unit,
                   'recorded_utc':datetime.now(timezone.utc).isoformat(),'telemetry':raw}
            report['tests'].append(entry)
            report_file.parent.mkdir(parents=True,exist_ok=True)
            report_file.write_text(json.dumps(report,indent=2))
            print('Saved. Robot has been stopped.')
            if input('r=repeat same test; Enter=next test: ').strip().lower()!='r':break
    print_report(report)
    print('Full report:',report_file)

if __name__=='__main__':
    try:main()
    except (KeyboardInterrupt,EOFError):
        print('\nTester closed. Each pulse worker has its own deadline and stop teardown.')
        raise SystemExit(130)
