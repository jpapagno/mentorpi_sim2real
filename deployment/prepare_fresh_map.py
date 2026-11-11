"""Back up current mapping configuration, then start a fresh disarmed SLAM map."""
import json
from pathlib import Path
import shutil
import subprocess
import uuid

root=Path('/home/pi/docker/tmp/mentorpi_nav')
subprocess.run(['sudo','-n','systemctl','stop','mentorpi-navigation.service'],check=True)
backup=root/'diagnostics'/('map_before_refresh_'+uuid.uuid4().hex[:8])
backup.mkdir()
for relative in ('config/localization_mode.json','config/localization.yaml','config/initial_pose.json',
                 'maps/test_area.yaml','maps/test_area.pgm'):
    source=root/relative
    if source.exists():shutil.copy2(source,backup/source.name)
(root/'config/localization_mode.json').write_text(json.dumps({'mode':'mapping'}))
subprocess.run(['sudo','-n','systemctl','start','mentorpi-navigation.service'],check=True)
print(json.dumps({'backup':str(backup),'mode':'mapping','motion':'default disarmed','old_map_preserved':True}),flush=True)
