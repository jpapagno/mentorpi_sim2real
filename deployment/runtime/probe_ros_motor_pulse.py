"""Single supervised ROS motor-path probe, independent of map localization."""
import json
import time
import rclpy
from std_srvs.srv import SetBool
from commission_drive import CommissionDrive

rclpy.init()
node = CommissionDrive()
result = {}
try:
    if not node.ownership.wait_for_service(timeout_sec=3):
        raise RuntimeError('Commissioning ownership service unavailable')
    request = SetBool.Request()
    request.data = True
    future = node.ownership.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=3)
    if future.result() is None or not future.result().success:
        raise RuntimeError('Commissioning ownership denied')
    node.phase(2, (0., 0., 0.), False)
    if not node.status or not node.applied:
        raise RuntimeError('Motor guard feedback unavailable')
    if any(reason not in ('disarmed', 'command_stale') for reason in node.status.get('blocked_by', [])):
        raise RuntimeError('Motor guard not ready: ' + str(node.status))
    start = time.monotonic()
    node.phase(.5, (.1, 0., 0.), True)
    end = time.monotonic()
    node.phase(1, (0., 0., 0.), False)
    result = {'requested_vx': .1, 'duration_s': .5,
              'applied_samples': [row[1:] for row in node.applied if start <= row[0] <= end],
              'filtered_nonzero_samples': sum(any(abs(v) > 1e-6 for v in row[1:]) for row in node.filtered if start <= row[0] <= end),
              'pulse_guard_reasons': sorted({reason for stamp, status in node.status_history if start <= stamp <= end for reason in status.get('blocked_by', [])}),
              'final_guard': node.status}
except Exception as exc:
    result['error'] = str(exc)
finally:
    node.phase(.5, (0., 0., 0.), False)
    if node.ownership.service_is_ready():
        request = SetBool.Request()
        request.data = False
        future = node.ownership.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=3)
    print(json.dumps(result, indent=2), flush=True)
    node.destroy_node()
    rclpy.shutdown()
