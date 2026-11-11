"""Select live local-grid/odometry navigation; does not enable motion."""
import json
from pathlib import Path
import shutil
import subprocess
import uuid

root=Path('/home/pi/docker/tmp/mentorpi_nav')
subprocess.run(['systemctl','stop','mentorpi-navigation.service'],check=True)
mode=root/'config/localization_mode.json'
backup=root/'diagnostics'/('navigation_mode_before_local_'+uuid.uuid4().hex[:8]+'.json')
if mode.exists():shutil.copy2(mode,backup)
mode.write_text(json.dumps({'mode':'local'}))
subprocess.run(['systemctl','start','mentorpi-navigation.service'],check=True)
print('Live local navigation selected. No saved map or AMCL is started. Motors default DISARMED.',flush=True)
