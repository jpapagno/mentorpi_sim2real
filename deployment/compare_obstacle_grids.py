"""Plot actual before/after policy tensors and quantify new forward obstacles."""
import json
from pathlib import Path
import cv2
import numpy as np

root=Path('logs/robot_deployment/obstacle_test')
before=np.load(root/'baseline_fixed_policy_grid.npy')>0
after=np.load(root/'obstacle_fixed_policy_grid.npy')>0
new=after&~before
axis=(49.5-np.arange(100))*.05
x,y=np.meshgrid(axis,axis,indexing='ij')
front=(x>.15)&(x<1.)&(abs(y)<.35)
metrics={'baseline_forward_blocked_cells':int(before[front].sum()),
         'obstacle_forward_blocked_cells':int(after[front].sum()),
         'new_forward_blocked_cells':int(new[front].sum()),
         'all_new_blocked_cells':int(new.sum()),'cell_resolution_m':.05,
         'interpretation':'Blocked includes occupied and unknown; live LiDAR/depth overlays add obstacles.'}
panels=[]
for title,grid in [('Clear baseline',before),('Obstacle placed',after),('New blocked cells',after)]:
    pixels=np.repeat(np.where(grid,0,255).astype(np.uint8)[:,:,None],3,axis=2)
    if title=='New blocked cells':pixels[new]=(0,130,255)
    pixels=cv2.resize(pixels[25:75,25:75],(400,400),interpolation=cv2.INTER_NEAREST)
    cv2.circle(pixels,(200,200),10,(0,0,220),2)
    cv2.arrowedLine(pixels,(200,200),(200,178),(0,0,220),2)
    panel=np.full((470,420,3),245,np.uint8)
    panel[45:445,10:410]=pixels
    cv2.putText(panel,title,(12,28),0,.7,(30,30,30),2)
    cv2.putText(panel,'2.5m square; forward UP',(12,464),0,.5,(30,30,30),1)
    panels.append(panel)
cv2.imwrite(str(root/'policy_grid_comparison.png'),np.concatenate(panels,axis=1))
(root/'policy_grid_comparison.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
