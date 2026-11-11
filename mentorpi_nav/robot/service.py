"""Host-side service preflight and orderly ROS launch shutdown."""
import argparse
from pathlib import Path
import subprocess
from .rrc_serial import DEVICE

def preflight():
    device=Path(DEVICE)
    if not device.exists():raise SystemExit('Motor board is absent')
    result=subprocess.run(['/usr/bin/fuser',str(device.resolve())],capture_output=True,text=True)
    if result.returncode==0:raise SystemExit('Motor serial port already in use: '+result.stdout.strip())
    if result.returncode!=1:raise SystemExit('Cannot inspect motor serial ownership: '+result.stderr)

def stop():
    code="""
import os,signal,time
from pathlib import Path
pids=[]
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=path.read_bytes().split(bytes([0]))
  if b'launch' in args and b'deployment/navigation.launch.py' in args:
   pids.append(int(path.parent.name));os.kill(pids[-1],signal.SIGINT)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
deadline=time.monotonic()+8
while time.monotonic()<deadline and any(Path('/proc/'+str(pid)).exists() for pid in pids):time.sleep(.1)
"""
    subprocess.run(['docker','exec','MentorPi','python3','-c',code],check=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','stop'])
    args=parser.parse_args()
    (preflight if args.action=='preflight' else stop)()

if __name__=='__main__':main()
