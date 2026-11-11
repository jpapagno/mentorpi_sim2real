"""Complete persistent stack; motors always start disarmed."""
from pathlib import Path
import json
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess,IncludeLaunchDescription,RegisterEventHandler,EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    root=Path('/home/ubuntu/shared/mentorpi_nav')
    peripherals=Path(get_package_share_directory('peripherals'))
    description=Path(get_package_share_directory('mentorpi_description'))
    runtime=root/'runtime';config=root/'config'

    def python(name):
        return ExecuteProcess(cmd=['python3',str(runtime/name)],output='screen',
            additional_env={'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})

    hardware=python('hardware_bridge.py')
    mode_file=config/'localization_mode.json'
    mode=json.loads(mode_file.read_text()).get('mode','amcl') if mode_file.exists() else 'amcl'
    if mode not in ('amcl','mapping','local'):raise ValueError('Unknown localization mode: '+str(mode))
    controller_process=python('local_goal_supervisor.py' if mode=='local' else 'autonomous_trials.py')
    if mode=='local':
        localization=[];managed=[];planners=[]
    else:
        planners=[Node(package='nav2_planner',executable='planner_server',name='planner_server',parameters=[str(config/'planner.yaml')],output='screen')]
    if mode=='mapping':
        localization=[Node(package='slam_toolbox',executable='async_slam_toolbox_node',
            name='slam_toolbox',parameters=[str(config/'slam.yaml')],output='screen')]
        managed=['planner_server']
    elif mode=='amcl':
        localization=[
            Node(package='nav2_map_server',executable='map_server',name='map_server',parameters=[str(config/'localization.yaml')],output='screen'),
            Node(package='nav2_amcl',executable='amcl',name='amcl',parameters=[str(config/'localization.yaml')],output='screen'),
            python('initialize_localization.py')]
        managed=['map_server','amcl','planner_server']
    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(description/'launch/robot_description.launch.py')),
            launch_arguments={'use_gui':'false','use_rviz':'false','use_sim_time':'false','namespace':'','use_namespace':'false'}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(peripherals/'launch/imu_filter.launch.py'))),
        hardware,
        Node(package='ldlidar_stl_ros2',executable='ldlidar_stl_ros2_node',name='mentorpi_lidar',
            parameters=[str(config/'lidar.yaml')],output='screen',respawn=True,respawn_delay=1.),
        python('scan_normalizer.py'),
        python('sensor_supervisor.py'),
        Node(package='rf2o_laser_odometry',executable='rf2o_laser_odometry_node',parameters=[str(config/'rf2o.yaml')],
            arguments=['--ros-args','--log-level','WARN'],output='screen'),
        python('measured_odometry.py'),
        Node(package='robot_localization',executable='ekf_node',name='ekf_filter_node',
            parameters=[str(config/'ekf.yaml')],remappings=[('odometry/filtered','odom')],output='screen'),
        *localization,
        *planners,
        python('perception_node.py'),
        *([Node(package='nav2_lifecycle_manager',executable='lifecycle_manager',name='mentorpi_lifecycle',
            parameters=[{'autostart':True,'node_names':managed}],output='screen')] if managed else []),
        controller_process,
        RegisterEventHandler(OnProcessExit(target_action=hardware,on_exit=[EmitEvent(event=Shutdown(reason='hardware bridge exited'))])),
        RegisterEventHandler(OnProcessExit(target_action=controller_process,on_exit=[EmitEvent(event=Shutdown(reason='controller exited'))])),
    ])
