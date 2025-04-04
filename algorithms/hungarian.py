import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from algorithms.base import TaskAllocationAlgorithm

class HungarianAlgorithm(TaskAllocationAlgorithm):
    """
    匈牙利算法实现（中心化最优分配）。
    """
    def __init__(self):
        super().__init__()

    def assign_tasks(self, tasks, agents, current_time=0):
        start_time = time.time()
        # 筛选当前可用的代理和任务（空闲且未失效的代理，已发布且未分配的任务）
        free_agents = [agent for agent in agents if (not agent.busy and not agent.failed)]
        available_tasks = [task for task in tasks if (not task.assigned and not task.completed 
                                                     and task.release_time <= current_time)]
        nA = len(free_agents)
        nT = len(available_tasks)
        if nA == 0 or nT == 0:
            # 无可分配的代理或任务
            self.last_computation_time = time.time() - start_time
            self.last_num_iterations = 1
            self.last_communication_cost = 0
            return []
        # 构建成本矩阵（大小：nA x nT）
        cost_matrix = np.full((nA, nT), fill_value=1e9)
        for i, agent in enumerate(free_agents):
            for j, task in enumerate(available_tasks):
                # 检查类型匹配
                if task.required_type is None or agent.type == task.required_type:
                    # 使用行驶时间作为成本
                    travel_time = agent.travel_time_to(task.location)
                    cost_matrix[i, j] = travel_time
        # 求解指派问题的最小成本分配
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        assignments = []
        for r, c in zip(row_idx, col_idx):
            if cost_matrix[r, c] < 1e9:
                agent = free_agents[r]
                task = available_tasks[c]
                assignments.append((agent, task))
        # 记录性能指标
        self.last_computation_time = time.time() - start_time
        self.last_num_iterations = 1
        self.last_communication_cost = 0
        return assignments
