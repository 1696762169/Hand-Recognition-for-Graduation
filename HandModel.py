import os
import numpy as np
import site
from enum import Enum
import time
import threading
import math

from glass_engine import *
from glass_engine.GlassEngineConfig import GlassEngineConfig
from glass_engine.Geometries import *
from glass_engine.Lights import *
from glass_engine.Renderers import ForwardRenderer
from glass_engine.Screens import QtScreen
from glass import sampler2D, ShaderProgram
import glm

from Utils import Utils

class DepthRenderer(ForwardRenderer):
    def __init__(self, depth_min: float = 0.0, depth_max: float = 1.0, **kwargs):
        ForwardRenderer.__init__(self)
        self.depth_min = depth_min
        self.depth_max = depth_max

    @property
    def forward_program(self):
        if "forward" in self.programs:
            return self.programs["forward"]

        program = ShaderProgram()
        GlassEngineConfig.define_for_program(program)
        shader_folder = os.path.join(site.getsitepackages()[1], r"glass_engine\glsl\Pipelines\forward_rendering")
        program.compile(
            shader_folder + "\\forward_rendering.vs"
        )
        program.compile(
            shader_folder + "\\forward_rendering.gs"
        )
        program.compile(
            os.path.join(Utils.get_root_path(), "Shader/depth_rendering.fs")
        )

        # 设置深度 范围
        program["depth_min"] = self.depth_min
        program["depth_max"] = self.depth_max
        self.programs["forward"] = program

        return program

    def render(self):
        self._should_update = False
        sampler2D._should_update = False
        self.classify_meshes()
        self.draw_opaque()
        return self._should_update or sampler2D._should_update

class HandColor(Enum):
    YELLOW = glm.vec3(1, 1, 0)
    RED = glm.vec3(1, 0, 0)
    GREEN = glm.vec3(0, 1, 0)
    BLUE = glm.vec3(0, 0, 1)

class HandModel(object):
    def __init__(self):
        self.scene = Scene()
        light = DirLight()
        light.pitch = -45
        light.yaw = 45
        self.scene.add(light)

        self.__init_model()

        self.camera = Camera()
        self.camera.fov = 30
        self.camera.position = glm.vec3(0, 0, 0)
        self.screen: QtScreen = self.camera.screen
        self.screen.resize(320, 320)    # 调整屏幕大小
        self.scene.add(self.camera)

    def show(self, show_color=False):
        self.screen.renderer = ForwardRenderer() if show_color else DepthRenderer(4, 6)
        self.screen.show()

    def render(self) -> np.ndarray:
        self.screen.renderer = DepthRenderer(4, 6)
        # 因为不明原因 需要仿照show方法来创建OpenGL上下文 才能正常运行
        self.screen.__class__.__base__.show(self.screen)
        self.screen.close()   # 创建窗口后直接退出
        self.screen.capture()
        return self.camera.take_photo(None, (0, 0, 320, 320))

    def __init_model(self):
        self.hand = SceneNode()

        # 创建手掌
        palm_height = 1.5    # 手掌高度
        palm_width = 2.0     # 手掌宽度
        palm_thickness = 0.5    # 手掌厚度与手掌宽度的比例  
        self.palm = Cylinder(radius=palm_width / 2, height=palm_height, color=HandColor.YELLOW.value)
        self.hand.add_child(self.palm)
        self.palm.scale.y = palm_thickness
        st1 = SphericalCapTop(height=palm_height / 5, color=HandColor.RED.value)
        self.palm.add_child(st1)
        st1.position.z = palm_height
        st2 = SphericalCapTop(height=palm_height / 5, color=HandColor.RED.value)
        self.palm.add_child(st2)
        st2.pitch = 180

        # 创建四根手指
        finger_length = 0.5  # 手指长度
        finger_diameter = 0.35   # 手指直径
        finger_length_scale = [0.9, 1.0, 0.9, 0.7]  # 手指长度比例
        finger_diameter_scale = [0.95, 1.0, 0.95, 0.85]  # 手指粗细比例
        self.fingers = []

        for i in range(4):
            finger = Sphere(radius=finger_diameter / 2, color=HandColor.GREEN.value)
            pre_joint = finger
            for j in range(3):
                knuckle = Cylinder(radius=finger_diameter / 2, height=finger_length, color=HandColor.BLUE.value)
                pre_joint.add_child(knuckle)
                knuckle.position.z = finger_diameter / 8
                joint = Sphere(radius=finger_diameter / 2, color=HandColor.GREEN.value)
                knuckle.add_child(joint)
                joint.position.z = finger_length + finger_diameter / 8

                pre_joint = joint

            # 计算手指位置
            finger.position.x = (palm_width - finger_diameter) * (i / 3 - 0.5)
            finger.position.z = palm_height * 1.1
            finger.scale.z = finger_length_scale[i]
            finger.scale.x = finger_diameter_scale[i]
            finger.scale.y = finger_diameter_scale[i]

            self.hand.add_child(finger)
            self.fingers.append(finger)

        # 创建大拇指
        thumb_length = finger_length * 0.8   # 大拇指长度
        thumb_diameter = finger_diameter * 1.2   # 大拇指直径
        self.thumb = SceneNode()
        self.hand.add_child(self.thumb)
        muscle_length = palm_height * 0.8   # 大拇指肌肉长度
        muscle_offset = palm_width / 2 * 1.2    # 大拇指肌肉横向偏移
        muscle_diameter = palm_thickness * palm_width * 0.7  # 大拇指肌肉直径
        self.thumb_muscle = Sphere(radius=muscle_diameter / 2, color=HandColor.RED.value)
        self.thumb.add_child(self.thumb_muscle)
        self.thumb_muscle.scale.z = muscle_length / muscle_diameter
        self.thumb_muscle.position.x = -muscle_offset
        self.thumb_muscle.position.z = palm_height * 0.4
        self.thumb_muscle.roll = -20

        self.thumb_joint = Sphere(radius=thumb_diameter / 2, color=HandColor.GREEN.value)
        self.thumb_muscle.add_child(self.thumb_joint)
        self.thumb_joint.position.z = self.thumb_muscle.radius * 0.9
        self.thumb_joint.scale.z = muscle_diameter / muscle_length
        pre_joint = self.thumb_joint
        for i in range(2):
            knuckle = Cylinder(radius=thumb_diameter / 2, height=thumb_length, color=HandColor.BLUE.value)
            pre_joint.add_child(knuckle)
            joint = Sphere(radius=thumb_diameter / 2, color=HandColor.GREEN.value)
            knuckle.add_child(joint)
            joint.position.z = thumb_length + thumb_diameter / 8

            pre_joint = joint

        self.thumb.yaw = 0
        self.thumb_muscle.position.x = -math.sqrt(
            math.pow(muscle_offset * math.cos(math.radians(self.thumb.yaw)), 2) + 
            math.pow(muscle_offset * palm_thickness * math.sin(math.radians(self.thumb.yaw)), 2))

        self.hand.position = glm.vec3(0, 5, 0)

        self.scene.add(self.hand)

    def set_pose(self, pose: np.ndarray):
        """
        设置手部姿态参数
        :param pose: 姿态参数，shape=(27,)
        0-2: 整体位置
        3-5: 整体姿态旋转角度
        6-26: 预留
        """
        self.set_position(pose[0], pose[1], pose[2])
        self.set_rotation(pose[3], pose[4], pose[5])

    def set_position(self, x: float, y: float, z: float):
        self.hand.position = glm.vec3(x, y, z)
    def set_rotation(self, pitch: float, yaw: float, roll: float):
        self.hand.pitch = pitch
        self.hand.yaw = yaw
        self.hand.roll = roll
    # def test(self):
    #     thread = threading.Thread(target=self.__test)
    #     thread.daemon = True
    #     thread.start()

    # def __test(self):
    #     while True:
    #         self.thumb.yaw += 1
    #         self.screen.show()
    #         time.sleep(0.03)
        
    @staticmethod
    def get_bounds():
        """
        获取手部姿态参数范围
        """
        return [
            (-5, 5), (-5, 5), (-5, 5),                  # 整体位置
            (-180, 180), (-180, 180), (-180, 180),      # 整体姿态旋转角度
            # 预留
            (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), 
            (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), 
            (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), 
            (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), 
            (0, 1),
        ]