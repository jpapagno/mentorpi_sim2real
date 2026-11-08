"""Host-side stop for only the project's complete ROS launch process."""
import subprocess
code="""
import os,signal,time
from pathlib import Path
pids=[]
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=path.read_bytes().split(b'\\0')
  if b'/home/ubuntu/shared/mentorpi_nav/runtime/bringup.launch.py' in args:
   pids.append(int(path.parent.name));os.kill(pids[-1],signal.SIGINT)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
deadline=time.monotonic()+8
while time.monotonic()<deadline and any(Path('/proc/'+str(pid)).exists() for pid in pids):time.sleep(.1)
print('Stopped project launch PIDs',pids)
"""
subprocess.run(['docker','exec','MentorPi','python3','-c',code],check=True)
