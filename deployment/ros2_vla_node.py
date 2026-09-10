#!/usr/bin/env python3
"""
ROS 2 Deployment Node for Intel Physical AI Challenge
------------------------------------------------------
This node demonstrates how the optimized OpenVINO VLA policy can be deployed 
onto real physical SO-101 robots via ROS 2 (Humble/Iron).

It subscribes to dual physical camera feeds and a language command topic, 
and publishes JointTrajectory commands to both arms.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
import openvino as ov
import numpy as np

class PhysicalVINONode(Node):
    def __init__(self):
        super().__init__('physical_vla_node')
        
        self.bridge = CvBridge()
        
        # Load Intel OpenVINO Model (INT8 PTQ)
        self.get_logger().info("Loading INT8 OpenVINO Policy for Edge inference...")
        core = ov.Core()
        model_path = "inference/ir_model/vla_policy_int8.xml"
        self.policy = core.compile_model(core.read_model(model_path), "AUTO")
        
        # ROS 2 Subscribers
        self.sub_cam = self.create_subscription(Image, '/robot/camera_main/image_raw', self.cam_callback, 10)
        self.sub_cmd = self.create_subscription(String, '/robot/language_command', self.cmd_callback, 10)
        
        # ROS 2 Publishers
        self.pub_left_arm = self.create_publisher(JointTrajectory, '/so101_left/joint_trajectory_controller/command', 10)
        self.pub_right_arm = self.create_publisher(JointTrajectory, '/so101_right/joint_trajectory_controller/command', 10)
        
        self.current_image = None
        self.current_instruction = None
        
        # 30Hz Inference Loop
        self.timer = self.create_timer(1.0 / 30.0, self.inference_loop)
        
    def cam_callback(self, msg):
        # Convert ROS Image to OpenCV/Numpy array
        self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        
    def cmd_callback(self, msg):
        self.current_instruction = msg.data
        self.get_logger().info(f"Received new command: {self.current_instruction}")
        
    def inference_loop(self):
        if self.current_image is None or self.current_instruction is None:
            return
            
        # Format for OpenVINO (1, 3, 480, 640)
        img_input = np.transpose(self.current_image, (2, 0, 1))
        img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
        
        # Dummy Language Embedding (In real deploy, run tokenizer here)
        lang_emb = np.random.randn(1, 768).astype(np.float32)
        
        # Execute VLA Policy on Intel Core Ultra Hardware
        actions = self.policy([img_input, lang_emb])[0][0] # (16,)
        
        left_arm_action = actions[:8]
        right_arm_action = actions[8:]
        
        self.publish_trajectory(self.pub_left_arm, left_arm_action)
        self.publish_trajectory(self.pub_right_arm, right_arm_action)

    def publish_trajectory(self, publisher, action_vector):
        traj = JointTrajectory()
        point = JointTrajectoryPoint()
        point.positions = action_vector.tolist()
        traj.points.append(point)
        publisher.publish(traj)

def main(args=None):
    rclpy.init(args=args)
    node = PhysicalVINONode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
