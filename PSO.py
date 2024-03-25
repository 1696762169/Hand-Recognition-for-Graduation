import torch
from torch import Tensor as tensor
from numpy import ndarray as array
from typing import Callable

from Utils import Utils

class PSOSolver(object):
    def __init__(self,
                particles_num: int,
                dimensions_num: int,
                dimensions_bounds: list[tuple],
                function: Callable[[tensor], tensor],
                *,
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                max_iter: int = 50,
                c1: float = 2.8,
                c2: float = 1.3,
                w: float | Callable[[int], float] = 0.9,
                bound_function: Callable[[tensor], tensor] = None):
        """
        parameters:
        particles_num: 粒子数量
        dimensions_num: 参数空间维度数量
        dimensions_range: 参数空间范围
        function: 目标函数 输入p*d的tensor 输出p个目标函数值
        device: 设备类型
        max_iter: 最大迭代次数
        c1: 个体加速因子
        c2: 群体加速因子
        w: 速度权重 可设置为常数或函数 函数输入迭代次数 输出速度权重
        bound_function: 限制参数范围的函数 输入p*d的tensor 输出p*d个限制函数值
        """
        self.p_num = particles_num
        self.d_num = dimensions_num
        self.bounds = torch.tensor(dimensions_bounds, device=device).float().t().to(device)
        self.function = function

        self.device = device
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.w = w if callable(w) else lambda _: w
        self.bound_function = bound_function if bound_function is not None else self._default_bound_function

    def solve(self, *,
              init_position: tensor | array = None,
              init_velocity: tensor | array = None) -> tensor:
        """
        执行粒子群算法
        parameters:
        init_position: 粒子初始位置
        init_velocity: 粒子初始速度
        return: 全局最佳位置对应的参数值
        """

        # 初始化粒子位置和速度
        if init_position is None:
            self.position = torch.rand(self.p_num, self.d_num, device=self.device)
            self.position = self.position * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        else:
            self.position = self._to_tensor(init_position)
        if init_velocity is None:
            self.velocity = torch.zeros(self.p_num, self.d_num, device=self.device)
        else:
            self.velocity = self._to_tensor(init_velocity)

        self.p_best = -torch.inf * torch.ones(self.p_num, device=self.device)    # 每个粒子最佳函数值
        self.p_best_position = self.position    # 每个粒子最佳位置对应的参数值
        self.g_best = -torch.inf    # 全局最佳位置
        self.g_best_position = None    # 全局最佳位置对应的参数值

        # 迭代
        for i in range(self.max_iter):
            # 计算目标值
            fitness = self.function(self.position)  # p个目标函数值
            self.p_best = torch.where(fitness > self.p_best, fitness, self.p_best)  # 更新每个粒子的最佳函数值
            self.p_best_position = torch.where(fitness > self.p_best, self.position.T, self.p_best_position.T).T  # 更新每个粒子的最佳位置对应的参数值
            g_best_index = torch.argmax(self.p_best)  # 全局最佳位置对应的粒子索引
            self.g_best = self.p_best[g_best_index]   # 更新全局最佳位置
            self.g_best_position = self.position[g_best_index]  # 全局最佳位置对应的参数值

            # 更新粒子位置和速度
            r1 = torch.rand(self.p_num, self.d_num, device=self.device)
            r2 = torch.rand(self.p_num, self.d_num, device=self.device)
            w = self.w(i)
            self.velocity = w * (self.velocity + self.c1 * r1 * (self.p_best_position - self.position) + self.c2 * r2 * (self.g_best_position - self.position))
            self.position = self.position + self.velocity
            # 限制粒子位置在参数空间范围内
            self.position = self.bound_function(self.position)

            # 打印日志
            Utils.log(f"Iteration {i+1}/{self.max_iter}, Global best fitness: {self.g_best:.4f}")

        return self.g_best_position


    def _to_tensor(self, x: tensor | array) -> tensor:
        if isinstance(x, tensor):
            return x.to(self.device)
        else:
            return torch.from_numpy(x).to(self.device)
        
    def _default_bound_function(self, x: tensor) -> tensor:
        x = torch.max(x, self.bounds[0])
        x = torch.min(x, self.bounds[1])
        return x
