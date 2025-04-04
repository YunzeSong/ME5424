import numpy as np

class Task:
    """
    任务类，包含任务的属性。
    """
    def __init__(self, task_id, location=(0,0), reward=0, release_time=0, duration=0, required_type=None):
        """
        初始化任务。
        :param task_id: 任务ID
        :param location: 任务位置 (x, y)
        :param reward: 任务价值（可用于收益或优先级）
        :param release_time: 任务发布（出现）时间
        :param duration: 任务执行所需时间
        :param required_type: 任务需要的代理类型或能力
        """
        self.id = task_id
        self.location = np.array(location, dtype=float)
        self.reward = reward
        self.release_time = release_time
        self.duration = duration
        self.required_type = required_type
        # 标志任务是否已被分配或完成
        self.assigned = False
        self.completed = False

    def __repr__(self):
        return f"Task(id={self.id}, loc={tuple(self.location)}, reward={self.reward}, rel={self.release_time})"

def generate_tasks(num_tasks, area_size=100, max_release_time=0, max_duration=0, task_types=None, seed=None):
    """
    随机生成任务列表。
    :param num_tasks: 任务数量
    :param area_size: 任务位置的区域大小（0到area_size的方形区域）
    :param max_release_time: 最大任务发布时间（如果>0则任务可能延迟出现）
    :param max_duration: 最大任务持续时间（如果>0则任务可能需要执行一定时间）
    :param task_types: 任务类型列表（若提供，则随机赋给任务一个类型）
    :param seed: 随机种子
    :return: 任务对象列表
    """
    if seed is not None:
        np.random.seed(seed)
    tasks = []
    for i in range(num_tasks):
        x = np.random.rand() * area_size
        y = np.random.rand() * area_size
        release_time = 0
        if max_release_time > 0:
            release_time = np.random.rand() * max_release_time  # 随机发布时间
        duration = 0
        if max_duration > 0:
            duration = np.random.rand() * max_duration  # 随机执行时长
        reward = np.random.rand() * 100  # 任务价值（0~100随机）
        t_type = None
        if task_types:
            t_type = np.random.choice(task_types)
        task = Task(task_id=i, location=(x, y), reward=reward, release_time=release_time,
                    duration=duration, required_type=t_type)
        tasks.append(task)
    # 根据发布时间排序任务列表
    tasks.sort(key=lambda t: t.release_time)
    return tasks
