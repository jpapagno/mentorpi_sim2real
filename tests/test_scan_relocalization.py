import math
import importlib.util
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deployment/runtime'))
from scan_relocalization import fit_scan_pose

@unittest.skipUnless(importlib.util.find_spec('cv2'),'OpenCV is available in the robot/SSH environment')
class ScanRelocalizationTests(unittest.TestCase):
    def test_recovers_manual_rotation_and_translation_from_asymmetric_map(self):
        values=np.zeros((100,120),np.int16)
        values[12:80,15]=100;values[80,15:100]=100
        values[30:81,100]=100;values[30,65:100]=100
        rows,cols=np.nonzero(values>=50)
        world=np.column_stack(((cols+.5)*.05-2.5,(rows+.5)*.05-2.5))
        pose=np.array([.1,-.05,.2]);delta=world-pose[:2]
        c,s=math.cos(pose[2]),math.sin(pose[2])
        points=np.column_stack((c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1]))
        result=fit_scan_pose(values,.05,[-2.5,-2.5,0],points,[0,0,0])
        self.assertLess(np.linalg.norm(np.array(result['pose'][:2])-pose[:2]),.04)
        self.assertLess(abs(result['pose'][2]-pose[2]),.03)
        self.assertGreater(result['after']['matched_fraction'],.98)
