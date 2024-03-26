import os
import numpy as np
import site

from glass_engine import *
from glass_engine.GlassEngineConfig import GlassEngineConfig
from glass_engine.Geometries import *
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

        program["DirLights"].bind(self.scene.dir_lights)
        program["PointLights"].bind(self.scene.point_lights)
        program["SpotLights"].bind(self.scene.spot_lights)

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


class HandModel(object):
    def __init__(self):
        self.scene = Scene()

        self.sphere = Cylinder()
        self.sphere.position = glm.vec3(0, 5, 0)
        self.scene.add(self.sphere)

        self.camera = Camera()
        self.camera.position = glm.vec3(0, 0, 0)
        self.screen: QtScreen = self.camera.screen
        self.screen.renderer = DepthRenderer(0, 10)
        self.scene.add(self.camera)

    def show(self):
        self.screen.show()