from abc import ABC, abstractmethod

class TaskAllocationAlgorithm(ABC):
    """
    任务分配算法的基类，定义接口和通用属性。
    """
    def __init__(self):
        # 统计指标初始化
        self.last_num_iterations = 0      # 上次分配的迭代次数/轮数
        self.last_communication_cost = 0  # 上次分配的通信开销（消息数）
        self.last_computation_time = 0.0  # 上次分配的计算耗时

    @abstractmethod
    def assign_tasks(self, tasks, agents, current_time=0):
        """
        分配任务的核心接口，返回任务分配结果列表。
        :param tasks: 当前待分配的任务列表（Task 对象）
        :param agents: 当前可用的代理列表（Agent 对象）
        :param current_time: 当前仿真时间（用于动态任务场景）
        :return: 分配结果列表，元素为 (agent, task) 二元组
        """
        pass
