"""Build a local human-readable report from read-only robot captures."""
import json
from pathlib import Path
import numpy as np
import cv2

root = Path('logs/robot_deployment')
inventory = json.loads((root/'topic_snapshot_current.json').read_text())
sensors = json.loads((root/'sensors_current.json').read_text())
scan = np.load(root/'scan_current.npz')
r = scan['ranges']
a = float(scan['angle_min']) + np.arange(len(r))*float(scan['angle_increment'])
valid = np.isfinite(r) & (r >= .02) & (r <= 25)
canvas = np.full((800, 800, 3), 248, dtype=np.uint8)
scale = 105
for radius in (1, 2, 3):
    cv2.circle(canvas, (400,400), radius*scale, (210,210,210), 1)
    cv2.putText(canvas, f'{radius} m', (405,400-radius*scale+18), 0,.5,(100,100,100),1)
cv2.line(canvas,(400,40),(400,760),(220,220,220),1)
cv2.line(canvas,(40,400),(760,400),(220,220,220),1)
for x,y in zip(r[valid]*np.cos(a[valid]),r[valid]*np.sin(a[valid])):
    px,py=int(400-y*scale),int(400-x*scale)
    cv2.circle(canvas,(px,py),2,(200,90,25),-1)
cv2.arrowedLine(canvas,(400,400),(400,375),(30,30,210),3)
cv2.putText(canvas,'Live LiDAR: sensor frame (+x up, +y left)',(40,30),0,.65,(30,30,30),1)
cv2.imwrite(str(root/'lidar_current.png'),canvas)
lines = ['# Stationary robot diagnostics — November 10, 2026', '',
    'Factory start_node.service active; custom navigation service stopped. No movement commands published by these captures.', '',
    'Robot clock reports February 2025; use collection context above for real date. Topic rates are observed subscriber arrival rates during short captures, not guaranteed hardware rates.', '',
    '## Sensor measurements', '', '| Topic | Hz | Messages | Details |', '|---|---:|---:|---|']
for topic, info in sensors['topics'].items():
    details = {k:v for k,v in info.items() if k not in ('hz','count','pose_covariance')}
    lines.append(f'| `{topic}` | {info["hz"]:.2f} | {info["count"]} | {json.dumps(details)} |')
lines += ['', '## Findings', '',
    '- RGB and depth images received at 640 × 480. RGB appears dark; depth has 42.2% valid pixels in this frame. Invalid depth is not free space.',
    '- LiDAR: 466/500 valid returns, nearest 0.668 m, median 1.692 m, farthest 2.720 m.',
    '- /odom_rf2o has no publisher; stock EKF reports an ERROR diagnostic (No events recorded). /odom exists but position covariance is about 6 m². Odometry output alone is not proof of actual wheel motion.',
    '- Battery telemetry in topic capture: 6.927 V. This does not establish a battery fault.',
    '- No /cmd_vel, /controller/cmd_vel, or motor-set messages arrived during stationary observation; motor-set has the expected /ros_robot_controller subscriber.',
    '- Idle command/event topics normally have no messages. Transient-local latched topics such as /tf_static and /robot_description can have publishers but no samples in this volatile capture.',
    '- Earlier custom and stock short motor pulses produced software command reports but user observed no wheel motion. Physical cause remains unresolved.', '',
    '## Every discovered topic', '', '| Topic | Message type | Publishers | Subscribers | Messages | Observed Hz |', '|---|---|---|---|---:|---:|']
for topic,info in sorted(inventory['topics'].items()):
    pubs=', '.join(p['node'] for p in info['publishers']) or 'none'
    subs=', '.join(info['subscribers']) or 'none'
    lines.append(f'| `{topic}` | {", ".join(info["types"])} | {pubs} | {subs} | {info["messages"]} | {info["observed_hz"]:.2f} |')
lines += ['', '## Compact topic samples', '', 'Large arrays are truncated; images and point clouds contain metadata instead of raw byte dumps.']
for topic,info in sorted(inventory['topics'].items()):
    lines += ['', f'### {topic}', '', '```json',json.dumps(info.get('sample',{'note':'No sample received during observation; see publisher/subscriber inventory.'}),indent=2),'```']
(root/'DIAGNOSTICS_CURRENT.md').write_text('\n'.join(lines),encoding='utf-8')
print(f'{len(inventory["topics"])} topics documented; report and LiDAR plot generated.')
