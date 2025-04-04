import numpy as np
import time
import statistics
from algorithms.base import TaskAllocationAlgorithm

class CBBAAlgorithm(TaskAllocationAlgorithm):
    """
    一致性捆绑算法 (CBBA) 的简化实现（分布式任务分配）。
    在迭代过程中增加最大迭代次数限制，确保在状态不收敛时也能退出。
    """
    def __init__(self, max_iterations=100):
        super().__init__()
        self.max_iterations = max_iterations

    def assign_tasks(self, tasks, agents, current_time=0):
        start_time = time.time()
        free_agents = [agent for agent in agents if (not agent.busy and not getattr(agent, 'failed', False))]
        available_tasks = [task for task in tasks if (not task.assigned and not task.completed and task.release_time <= current_time)]
        if not free_agents or not available_tasks:
            self.last_computation_time = time.time() - start_time
            self.last_num_iterations = 0
            self.last_communication_cost = 0
            return []
        # 计算每个代理对每个任务的效用，使用负距离作为效用值
        value = { (agent.id, task.id): -np.linalg.norm(agent.location - task.location) 
                  if (task.required_type is None or agent.type == task.required_type) else -np.inf
                  for agent in free_agents for task in available_tasks }
        # 为每个代理生成任务偏好列表（按效用从高到低排序）
        agent_preferences = {
            agent.id: sorted([t.id for t in available_tasks], key=lambda tid: value[(agent.id, tid)], reverse=True)
            for agent in free_agents
        }
        # 当前分配状态
        agent_current_task = {agent.id: None for agent in free_agents}
        task_current_agent = {task.id: None for task in available_tasks}
        unassigned_queue = [agent.id for agent in free_agents]
        message_count = 0
        iterations = 0
        while unassigned_queue and iterations < self.max_iterations:
            iterations += 1
            agent_id = unassigned_queue.pop(0)
            if agent_current_task.get(agent_id) is not None or not agent_preferences.get(agent_id):
                continue
            pref_list = agent_preferences[agent_id]
            assigned = False
            while pref_list:
                task_id = pref_list[0]
                message_count += 1
                if task_current_agent[task_id] is None:
                    agent_current_task[agent_id] = task_id
                    task_current_agent[task_id] = agent_id
                    assigned = True
                    break
                else:
                    current_holder = task_current_agent[task_id]
                    if value[(agent_id, task_id)] > value[(current_holder, task_id)]:
                        task_current_agent[task_id] = agent_id
                        agent_current_task[agent_id] = task_id
                        agent_current_task[current_holder] = None
                        if current_holder in agent_preferences and task_id in agent_preferences[current_holder]:
                            agent_preferences[current_holder].remove(task_id)
                        unassigned_queue.append(current_holder)
                        assigned = True
                        break
                    else:
                        pref_list.pop(0)
                        continue
            if not assigned:
                continue
        if iterations >= self.max_iterations:
            print("Warning: CBBAAlgorithm reached max iterations without full convergence.")
        assignments = []
        for task_id, agent_id in task_current_agent.items():
            if agent_id is not None:
                agent_obj = next((a for a in free_agents if a.id == agent_id), None)
                task_obj = next((t for t in available_tasks if t.id == task_id), None)
                if agent_obj and task_obj:
                    assignments.append((agent_obj, task_obj))
        self.last_computation_time = time.time() - start_time
        self.last_num_iterations = iterations
        self.last_communication_cost = message_count
        return assignments
