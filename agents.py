import numpy as np

class Agent:
    """
    代理类，包含代理的属性。
    """
    def __init__(self, agent_id, location=(0,0), speed=1.0, agent_type=None):
        """
        初始化代理。
        :param agent_id: 代理ID
        :param location: 初始位置 (x, y)
        :param speed: 移动速度 (距离单位每时间单位)
        :param agent_type: 代理类型或能力（用于匹配任务需要的类型）
        """
        self.id = agent_id
        self.location = np.array(location, dtype=float)
        self.speed = speed
        self.type = agent_type
        self.busy = False   # 是否正在执行任务
        self.failed = False # 是否失效（用于模拟故障）
    def travel_time_to(self, task_location):
        """
        计算到某任务位置的行驶时间。
        """
        dist = np.linalg.norm(self.location - task_location)
        if self.speed <= 0:
            return float('inf')
        return dist / self.speed
    def __repr__(self):
        return f"Agent(id={self.id}, loc={tuple(self.location)}, type={self.type})"
