"""LiDAR, measured odometry, motor bridge and disarmed policy ownership."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess,IncludeLaunchDescription,RegisterEventHandler,EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from .paths import ROOT,CONFIG

def generate_launch_description():
    peripherals=Path(get_package_share_directory('peripherals'))
    description=Path(get_package_share_directory('mentorpi_description'))
    def module(name):
        return ExecuteProcess(cmd=['python3','-m','mentorpi_nav.robot.'+name],cwd=str(ROOT),output='screen',
            additional_env={'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
    hardware=module('hardware_bridge');ownership=module('ownership')
    actions=[
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(description/'launch/robot_description.launch.py')),
            launch_arguments={'use_gui':'false','use_rviz':'false','use_sim_time':'false','namespace':'','use_namespace':'false'}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(peripherals/'launch/imu_filter.launch.py'))),
        hardware,
        Node(package='ldlidar_stl_ros2',executable='ldlidar_stl_ros2_node',name='mentorpi_lidar',
            parameters=[str(CONFIG/'lidar.yaml')],output='screen',respawn=True,respawn_delay=1.),
        module('scan_normalizer'),module('sensor_supervisor'),
        Node(package='rf2o_laser_odometry',executable='rf2o_laser_odometry_node',parameters=[str(CONFIG/'rf2o.yaml')],
            arguments=['--ros-args','--log-level','WARN'],output='screen'),
        module('measured_odometry'),
        Node(package='robot_localization',executable='ekf_node',name='ekf_filter_node',parameters=[str(CONFIG/'ekf.yaml')],
            remappings=[('odometry/filtered','odom')],output='screen'),
        module('occupancy_publisher'),ownership]
    for process in (hardware,ownership):
        actions.append(RegisterEventHandler(OnProcessExit(target_action=process,
            on_exit=[EmitEvent(event=Shutdown(reason='critical navigation process exited'))])))
    return LaunchDescription(actions)
