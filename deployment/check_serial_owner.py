"""Host-side preflight catches serial users outside the ROS container."""
from pathlib import Path
import subprocess

device=Path('/dev/serial/by-id/usb-1a86_USB_Single_Serial_5917011712-if00')
if not device.exists():
    raise SystemExit('Recorded motor board is absent')
result=subprocess.run(['/usr/bin/fuser',str(device.resolve())],capture_output=True,text=True)
if result.returncode == 0:
    raise SystemExit('Motor serial port already in use by PIDs: '+result.stdout.strip())
if result.returncode != 1:
    raise SystemExit('Unable to verify serial ownership: '+result.stderr)
print('Motor serial port has no existing owner',flush=True)
