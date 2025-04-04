import numpy as np
import time
from algorithms.base import TaskAllocationAlgorithm

class AuctionAlgorithm(TaskAllocationAlgorithm):
    """
    拍卖算法实现（Bertsekas 拍卖法，可视为中心化迭代竞价过程）。
    增加了最大迭代次数限制，防止因无法收敛而出现死循环。
    """
    def __init__(self, epsilon=1e-3, max_iterations=100):
        super().__init__()
        self.epsilon = epsilon  # 微小增量，保证收敛
        self.max_iterations = max_iterations

    def assign_tasks(self, tasks, agents, current_time=0):
        start_time = time.time()
        free_agents = [agent for agent in agents if (not agent.busy and not getattr(agent, 'failed', False))]
        available_tasks = [task for task in tasks if (not task.assigned and not task.completed 
                                                     and task.release_time <= current_time)]
        nA = len(free_agents)
        nT = len(available_tasks)
        if nA == 0 or nT == 0:
            self.last_computation_time = time.time() - start_time
            self.last_num_iterations = 0
            self.last_communication_cost = 0
            return []
        # 计算价值矩阵（用负距离作为价值）
        value = np.full((nA, nT), -np.inf)
        for i, agent in enumerate(free_agents):
            for j, task in enumerate(available_tasks):
                if task.required_type is None or agent.type == task.required_type:
                    dist = np.linalg.norm(agent.location - task.location)
                    value[i, j] = -dist

        prices = np.zeros(nT)
        agent_to_task = {agent.id: None for agent in free_agents}
        task_to_agent = {task.id: None for task in available_tasks}
        unassigned_agents = [agent for agent in free_agents]
        iter_count = 0
        message_count = 0
        while unassigned_agents and iter_count < self.max_iterations:
            iter_count += 1
            agent = unassigned_agents.pop(0)
            i = free_agents.index(agent)
            utilities = value[i, :] - prices
            if np.all(np.isneginf(utilities)):
                continue
            j_best = int(np.nanargmax(utilities))
            best_utility = utilities[j_best]
            utilities[j_best] = -np.inf
            second_best = np.nanmax(utilities) if np.any(utilities != -np.inf) else -np.inf
            if second_best == -np.inf:
                increment = self.epsilon
            else:
                increment = best_utility - second_best + self.epsilon
            prices[j_best] += increment
            task = available_tasks[j_best]
            prev_agent_id = task_to_agent.get(task.id)
            if prev_agent_id is not None:
                prev_agent = next((a for a in free_agents if a.id == prev_agent_id), None)
                if prev_agent is not None:
                    unassigned_agents.append(prev_agent)
                    agent_to_task[prev_agent.id] = None
            agent_to_task[agent.id] = task.id
            task_to_agent[task.id] = agent.id
            message_count += 1
        if iter_count >= self.max_iterations:
            print("Warning: AuctionAlgorithm reached max iterations without full convergence.")
        assignments = []
        for agent_id, task_id in agent_to_task.items():
            if task_id is not None:
                agent_obj = next((a for a in free_agents if a.id == agent_id), None)
                task_obj = next((t for t in available_tasks if t.id == task_id), None)
                if agent_obj is not None and task_obj is not None:
                    assignments.append((agent_obj, task_obj))
        self.last_computation_time = time.time() - start_time
        self.last_num_iterations = iter_count
        self.last_communication_cost = message_count
        return assignments
