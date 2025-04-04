import numpy as np
import time
# Gym和稳定基线库仅在可用时导入
try:
    import gym
    from gym import spaces
except ImportError:
    gym = None
try:
    from stable_baselines3 import PPO, DQN
except ImportError:
    PPO = None
    DQN = None
from algorithms.base import TaskAllocationAlgorithm

# 定义自定义 Gym 环境（仅在 gym 可用时）
if gym:
    class TaskAllocationEnv(gym.Env):
        """
        多代理任务分配的自定义环境，用于强化学习训练。
        状态: 每个代理到每个任务的距离 (展平为一维数组)；未分配用 -1 表示。
        动作: 一个整数，编码 (代理索引 * 任务数 + 任务索引) 表示给某代理分配某任务。
        奖励: 使用负距离作为即时奖励，完成所有分配后结束。
        """
        def __init__(self, n_agents, n_tasks, area_size=100):
            super(TaskAllocationEnv, self).__init__()
            self.n_agents = n_agents
            self.n_tasks = n_tasks
            self.area_size = area_size
            # 观察空间: 距离矩阵展平后的向量，每项范围 [0, max_dist] 或 -1 表示无效
            obs_size = n_agents * n_tasks
            max_dist = float(area_size) * np.sqrt(2)
            self.observation_space = spaces.Box(low=-1.0, high=max_dist, shape=(obs_size,), dtype=np.float32)
            # 动作空间: 离散值，对应代理-任务组合
            self.action_space = spaces.Discrete(n_agents * n_tasks)
            # 环境内部状态
            self.tasks = []
            self.agents = []
            self.assigned_agents = set()
            self.assigned_tasks = set()

        def reset(self):
            # 随机初始化代理和任务的位置
            self.tasks = [(np.random.rand()*self.area_size, np.random.rand()*self.area_size) 
                          for _ in range(self.n_tasks)]
            self.agents = [(np.random.rand()*self.area_size, np.random.rand()*self.area_size) 
                           for _ in range(self.n_agents)]
            self.assigned_agents.clear()
            self.assigned_tasks.clear()
            return self._get_observation()

        def _get_observation(self):
            obs = np.full((self.n_agents, self.n_tasks), -1.0, dtype=np.float32)
            for i in range(self.n_agents):
                for j in range(self.n_tasks):
                    if i not in self.assigned_agents and j not in self.assigned_tasks:
                        dx = self.agents[i][0] - self.tasks[j][0]
                        dy = self.agents[i][1] - self.tasks[j][1]
                        obs[i, j] = np.hypot(dx, dy)  # 计算欧氏距离
            return obs.flatten()

        def step(self, action):
            # 解码动作为 (代理, 任务)
            agent_index = action // self.n_tasks
            task_index = action % self.n_tasks
            reward = 0.0
            done = False
            # 判断动作是否有效（代理或任务是否已被分配）
            if agent_index in self.assigned_agents or task_index in self.assigned_tasks:
                reward = -10.0  # 无效动作惩罚
            else:
                # 奖励使用负距离（距离越短奖励越高）
                dx = self.agents[agent_index][0] - self.tasks[task_index][0]
                dy = self.agents[agent_index][1] - self.tasks[task_index][1]
                dist = np.hypot(dx, dy)
                reward = -dist
                # 标记该代理和任务为已分配
                self.assigned_agents.add(agent_index)
                self.assigned_tasks.add(task_index)
            # 终止条件：所有代理或所有任务都已分配完
            if len(self.assigned_agents) == self.n_agents or len(self.assigned_tasks) == self.n_tasks:
                done = True
            obs = self._get_observation()
            return obs, reward, done, {}
else:
    # 如果 gym 不可用，则定义一个占位环境类
    class TaskAllocationEnv:
        def __init__(self, n_agents, n_tasks, area_size=100):
            pass
        def reset(self):
            return None
        def step(self, action):
            return None, 0, True, {}

class RLAlgorithm(TaskAllocationAlgorithm):
    """
    强化学习任务分配方法。
    支持集中训练、分散执行。若提供预训练模型路径则加载模型，否则采用简单策略。
    """
    def __init__(self, model_path=None):
        super().__init__()
        self.model = None
        # 尝试加载预训练模型（需要stable_baselines3支持）
        if model_path and PPO:
            try:
                self.model = PPO.load(model_path)
            except Exception as e:
                self.model = None
        # 如无模型或未安装RL库，则在assign_tasks中使用随机策略

    def assign_tasks(self, tasks, agents, current_time=0):
        start_time = time.time()
        free_agents = [agent for agent in agents if (not agent.busy and not agent.failed)]
        available_tasks = [task for task in tasks if (not task.assigned and not task.completed 
                                                     and task.release_time <= current_time)]
        if not free_agents or not available_tasks:
            self.last_computation_time = time.time() - start_time
            self.last_num_iterations = 0
            self.last_communication_cost = 0
            return []
        assignments = []
        if self.model is not None:
            # 使用预训练RL模型进行决策
            env = TaskAllocationEnv(len(free_agents), len(available_tasks))
            obs = env.reset()
            done = False
            # 迭代调用模型直到分配完成
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
            # 根据 env 分配结果映射回真实代理和任务（简化为顺序对应）
            for ai, ti in zip(env.assigned_agents, env.assigned_tasks):
                agent = free_agents[ai]
                task = available_tasks[ti]
                assignments.append((agent, task))
        else:
            # 无模型时的占位策略：随机为每个代理选择一个任务（不重复）
            np.random.shuffle(available_tasks)
            for agent, task in zip(free_agents, available_tasks):
                assignments.append((agent, task))
        # 记录性能指标（这里迭代次数取分配的数量，通信视为0）
        self.last_computation_time = time.time() - start_time
        self.last_num_iterations = len(assignments)
        self.last_communication_cost = 0
        return assignments
