"""Depth camera without unnecessary RGB-colored point-cloud construction."""
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(namespace='ascamera',package='ascamera',executable='ascamera_node',
             respawn=True,output='screen',parameters=[{
                 'usb_bus_no':-1,'usb_path':'null',
                 'confiPath':'/home/ubuntu/third_party_ros2/third_party_ws/src/ascamera/configurationfiles',
                 'color_pcl':False,'pub_tfTree':True,
                 'depth_width':640,'depth_height':480,
                 'rgb_width':640,'rgb_height':480,'fps':15}]),
        Node(package='tf2_ros',executable='static_transform_publisher',
             arguments=['0','0','0','-1.57','0','-1.57','depth_cam','ascamera_camera_link_0']),
        Node(package='tf2_ros',executable='static_transform_publisher',
             arguments=['0','0','0','-1.57','0','-1.57','depth_cam','ascamera_color_0'])])
