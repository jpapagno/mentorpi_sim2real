"""Pure limits for a supervised relative-angle motor diagnostic."""
import math

def turn_decision(target, yaw, elapsed, commanded_angle, imu_age, battery_mv):
    if not all(math.isfinite(v) for v in (target,yaw,elapsed,commanded_angle,imu_age,battery_mv)):
        return 'fault:nonfinite',0.
    sign=1 if target>0 else -1
    progress=sign*yaw
    if imu_age>.25:return 'fault:imu_stale',0.
    if battery_mv<6400:return 'fault:low_battery',0.
    if progress < -math.radians(5):return 'fault:wrong_direction',0.
    if progress > abs(target)+math.radians(10):return 'fault:overshoot',0.
    if abs(target)-progress <= math.radians(2):return 'target_reached',0.
    if elapsed>=4:return 'fault:time_limit',0.
    if commanded_angle>=math.radians(90):return 'fault:commanded_angle_limit',0.
    # Slow for the final 8 degrees; both levels are bounded below the wheel cap.
    return 'turning',sign*(.5 if abs(target)-progress<math.radians(8) else .85)
