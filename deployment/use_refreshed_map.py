"""Switch disarmed live mapping to the newly saved map/localizer."""
import json
from pathlib import Path
import subprocess

root=Path('/home/pi/docker/tmp/mentorpi_nav')
for name in ('test_area_20261110.yaml','test_area_20261110.pgm'):
    if not (root/'maps'/name).exists():raise RuntimeError('Fresh map missing: '+name)
if 'test_area_20261110.yaml' not in (root/'config/localization.yaml').read_text():
    raise RuntimeError('Localizer does not reference the refreshed map')
subprocess.run(['systemctl','stop','mentorpi-navigation.service'],check=True)
(root/'config/localization_mode.json').write_text(json.dumps({'mode':'amcl'}))
subprocess.run(['systemctl','start','mentorpi-navigation.service'],check=True)
print('Refreshed map selected; navigation starts DISARMED',flush=True)
