"""Repeatable bounded motion protocol and offline repeatability analysis.

Host supervisor runs on the Pi outside ROS; existing calibrate_motion owns ROS
and guarded commands. No serial access, firmware changes or implicit motion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time


def protocol(translation_only=False):
    if translation_only:
        return [dict(name=f'translation_{int(speed*100):02d}',
                     directions=['forward','forward','backward','backward','left','left','right','right'],
                     linear=speed,yaw=.2,hold=1.5) for speed in (.1,.08,.06)]
    # Balanced blocks limit net travel; duplicate directions expose history.
    return [
        dict(name='longitudinal', directions=['forward','forward','backward','backward'], linear=.1, yaw=.3, hold=1),
        dict(name='lateral', directions=['left','left','right','right'], linear=.1, yaw=.3, hold=1),
        dict(name='yaw_left_first', directions=['ccw','ccw','cw','cw'], linear=.1, yaw=.3, hold=1),
        dict(name='yaw_right_first', directions=['cw','cw','ccw','ccw'], linear=.1, yaw=.3, hold=1),
        dict(name='low_command', directions=['forward','backward','left','right','ccw','cw'], linear=.06, yaw=.2, hold=1),
        dict(name='idle_reversal', directions=['ccw','cw'], linear=.1, yaw=.3, hold=5),
    ]


def summarize(root):
    groups={}; failures=[]
    manifest_path=Path(root)/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    expected=manifest.get('repeats')
    for filename in sorted(Path(root).glob('run_*/block_*.json')):
        record=json.loads(filename.read_text())
        if not record.get('passed'): failures.append({'file':str(filename),'error':record.get('error')})
        for index,pulse in enumerate(record.get('pulses',[])):
            key=f"{filename.stem}/{index}/{pulse['direction']}"
            axis=2 if pulse['direction'] in ('cw','ccw') else int(pulse['direction'] in ('left','right'))
            metrics={f'body_{i}':v for i,v in enumerate(pulse['body_delta'])}
            metrics.update({f'laser_{i}':v for i,v in enumerate(pulse['laser_body_delta'])})
            metrics.update({f'settling_{i}':v for i,v in enumerate(pulse['settling_body_delta'])})
            metrics.update(gyro_angle=pulse['gyro_angle_radians'], applied_integral=pulse['integrated_applied_velocity'][axis], axis_gain=pulse['axis_gain'])
            metrics['command_duration_s']=pulse['command_duration_s'] if 'command_duration_s' in pulse else None
            metrics['unintended_translation_m']=sum(v*v for i,v in enumerate(pulse['body_delta'][:2]) if i!=axis)**.5
            start=pulse.get('start_time'); end=pulse.get('command_end_time')
            if start is not None and end is not None:
                trace=[r for r in record.get('trace',[]) if start<=r['time']<end+1]
                moving=[r for r in trace if any(abs(v)>1e-6 for v in r['guard'].get('applied',[]))]
                metrics['first_applied_delay_s']=moving[0]['time']-start if moving else None
                hold=[r for r in trace if r['time']>=end+.3]
                metrics['post_stop_max_wheel_command']=max((abs(v) for r in hold for v in r['guard'].get('wheel_command_rps',[])),default=None)
            groups.setdefault(key,[]).append(metrics)
    result={'version':'motion_repeatability_v1','failures':failures,'groups':{}}
    result['expected_repeats']=expected
    result['missing_blocks']=[]
    if expected:
        for repeat in range(expected):
            for index,block in enumerate(manifest['protocol']):
                filename=Path(root)/f'run_{repeat+1:02d}'/f'block_{index:02d}_{block["name"]}.json'
                if not filename.exists():result['missing_blocks'].append(str(filename))
    for key,rows in groups.items():
        stats={}
        for metric in rows[0]:
            values=[row[metric] for row in rows if row[metric] is not None]
            if values:
                mean=statistics.mean(values); sd=statistics.stdev(values) if len(values)>1 else None
                stats[metric]={'n':len(values),'mean':mean,'std':sd,'min':min(values),'max':max(values),
                               'cv':sd/abs(mean) if sd is not None and abs(mean)>1e-5 else None}
        result['groups'][key]=stats
    result['interpretation']='Compare identical ordered slots across runs. Missing/aborted trials are failures, not zero motion. No automatic calibration acceptance; angles radians, distances metres. No power reset between runs.'
    return result


def execute(root, runtime, repeats,translation_only=False):
    root=Path(root); root.mkdir(parents=True,exist_ok=False)
    plan=protocol(translation_only); encoded=json.dumps(plan,sort_keys=True).encode()
    manifest={'version':'bounded_motion_suite_v1','protocol':plan,'sha256':hashlib.sha256(encoded).hexdigest(),
              'repeats':repeats,'pulse_seconds':1,'clock':'Pi monotonic in result traces; Pi wall clock may be wrong',
              'reset':'service restart between runs; no motor-board power reset',
              'runtime':str(runtime),'translation_only':translation_only}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    # Host and container share this deployment directory; preserve exact probe.
    deployed=Path('/home/pi/docker/tmp/mentorpi_nav')/Path(runtime).relative_to('/home/ubuntu/shared/mentorpi_nav')
    source=deployed.read_bytes()
    (root/'calibrate_motion_source.py').write_bytes(source)
    (root/'suite_source.py').write_bytes(Path(__file__).read_bytes())
    manifest['probe_source_sha256']=hashlib.sha256(source).hexdigest()
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    service=['sudo','-n','systemctl']; recorder=None
    try:
        for repeat in range(repeats):
            run=root/f'run_{repeat+1:02d}';run.mkdir()
            container_run=Path('/home/ubuntu/shared/mentorpi_nav')/run.relative_to('/home/pi/docker/tmp/mentorpi_nav')
            subprocess.run(service+['start','mentorpi-navigation.service'],check=True,timeout=20)
            topics='/scan /imu /odom_rf2o /mentorpi/measured_odom /mentorpi/guard_status /mentorpi/applied_cmd_vel /mentorpi/motion_enable'
            bag=f'source /home/ubuntu/.zshrc >/dev/null; exec ros2 bag record -o {container_run}/bag {topics}'
            base=['docker','exec','-u','ubuntu','-w','/home/ubuntu','MentorPi','/bin/zsh','-c']
            with (run/'bag.log').open('w') as baglog:
                recorder=subprocess.Popen(base+[bag],stdout=baglog,stderr=subprocess.STDOUT)
                try:
                    time.sleep(2)
                    if recorder.poll() is not None:raise RuntimeError('Recorder failed')
                    for index,block in enumerate(plan):
                        output=run/f'block_{index:02d}_{block["name"]}.json'
                        cmd=f'source /home/ubuntu/.zshrc >/dev/null; exec python3 {runtime} --execute --check-clearance --allow-stationary --pulse-seconds 1 --repetitions 1 --linear-speed {block["linear"]} --yaw-speed {block["yaw"]} --settle-seconds {block["hold"]} --directions '+ ' '.join(block['directions'])+f' --output {container_run/output.name}'
                        print(json.dumps({'event':'block_start','run':repeat+1,'block':block['name']}),flush=True)
                        with output.with_suffix('.log').open('w') as log:
                            subprocess.run(base+[cmd],stdout=log,stderr=subprocess.STDOUT,timeout=55,check=True)
                        print(json.dumps({'event':'block_done','run':repeat+1,'block':block['name']}),flush=True)
                finally:
                    subprocess.run(service+['stop','mentorpi-navigation.service'],timeout=20,check=False)
                    subprocess.run(['docker','exec','MentorPi','pkill','-INT','-f',f'ros2 bag record -o {container_run}/bag'],timeout=10,check=False)
                    try:recorder.wait(timeout=15)
                    except subprocess.TimeoutExpired:recorder.terminate()
                    recorder=None
    finally:
        subprocess.run(service+['stop','mentorpi-navigation.service'],timeout=20,check=False)
        (root/'summary.json').write_text(json.dumps(summarize(root),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--summarize',action='store_true')
    parser.add_argument('--translation-only',action='store_true',help='No yaw commands; three speeds with repeated/reversed translation')
    parser.add_argument('--repeats',type=int,choices=[2,3],default=3)
    parser.add_argument('--runtime',default='/home/ubuntu/shared/mentorpi_nav/runtime/calibrate_motion.py')
    args=parser.parse_args()
    if args.summarize:
        print(json.dumps(summarize(args.output),indent=2));return
    if not args.execute:
        print(json.dumps({'protocol':protocol(args.translation_only),'repeats':args.repeats,'motion':False},indent=2));return
    # Paths enter container shell commands; accept only simple absolute paths.
    for value in (args.output,args.runtime):
        if not value.startswith('/') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./-' for c in value):
            parser.error('Execution paths must be simple absolute Linux paths')
    execute(args.output,args.runtime,args.repeats,args.translation_only)


if __name__=='__main__':main()
