import subprocess
stop_code="""
import os,signal
from pathlib import Path
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=path.read_bytes().split(b'\\0')
  if len(args)>1 and args[1]==b'/home/ubuntu/shared/mentorpi_nav/runtime/hardware_bridge.py':
   os.kill(int(path.parent.name),signal.SIGTERM)
   print('Stopped owned hardware bridge',path.parent.name)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
"""
subprocess.run(['docker','exec','MentorPi','python3','-c',stop_code],check=True)
command="source /home/ubuntu/.zshrc >/dev/null; exec python3 /home/ubuntu/shared/mentorpi_nav/runtime/hardware_bridge.py > /home/ubuntu/shared/mentorpi_nav/logs/hardware_bridge.log 2>&1"
subprocess.run(['docker','exec','-d','-u','ubuntu','-w','/home/ubuntu','MentorPi','/bin/zsh','-c',command],check=True)
