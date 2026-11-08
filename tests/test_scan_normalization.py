import math
import unittest
import numpy as np
from deployment.runtime.scan_math import fixed_circle_scan


class ScanNormalizationTests(unittest.TestCase):
    def test_constant_shape_and_obstacle_preservation(self):
        for count in [499,500,501]:
            scan=np.full(count,2.,dtype=np.float32)
            scan[count//2]=.4
            result=fixed_circle_scan(scan,-math.pi,2*math.pi/count,.1,12.)
            self.assertEqual(result.shape,(720,))
            self.assertAlmostEqual(float(result.min()),.4,places=6)
            self.assertTrue(np.any(np.flatnonzero(result<1.)>350))

    def test_invalid_ranges_never_become_close_obstacles(self):
        result=fixed_circle_scan([float('nan')]*500,-math.pi,2*math.pi/500,.1,12.)
        self.assertTrue(np.isinf(result).all())
        with self.assertRaises(ValueError):fixed_circle_scan([1.]*100,0.,.01,.1,12.)
