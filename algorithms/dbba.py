import numpy as np
import time
from algorithms.base import TaskAllocationAlgorithm
from algorithms.auction import AuctionAlgorithm

class DBBAAlgorithm(TaskAllocationAlgorithm):
    """
    分布式竞价/搜索算法 (DBBA) 的代表实现。
    采用：先利用拍卖算法得到初始分配，再进行局部交换优化。
    在局部搜索过程中增加最大迭代次数限制，以防止死循环。
    """
    def __init__(self, max_iterations=100):
        super().__init__()
        self.max_iterations = max_iterations
        self.initial_allocator = AuctionAlgorithm()

    def assign_tasks(self, tasks, agents, current_time=0):
        start_time = time.time()
        # 获取初始分配
        initial_assignments = self.initial_allocator.assign_tasks(tasks, agents, current_time)
        assignment_map = {agent.id: task for agent, task in initial_assignments}
        
        def calc_total_cost(assign_map):
            total = 0.0
            for agent_id, task in assign_map.items():
                agent_obj = next((a for a in agents if a.id == agent_id), None)
                if agent_obj and task:
                    total += np.linalg.norm(agent_obj.location - task.location)
            return total
        
        best_cost = calc_total_cost(assignment_map)
        improved = True
        iter_count = 0
        message_count = 0
        while improved and iter_count < self.max_iterations:
            iter_count += 1
            improved = False
            agent_ids = list(assignment_map.keys())
            for i in range(len(agent_ids)):
                for j in range(i+1, len(agent_ids)):
                    a_id = agent_ids[i]
                    b_id = agent_ids[j]
                    task_a = assignment_map.get(a_id)
                    task_b = assignment_map.get(b_id)
                    if task_a is None or task_b is None:
                        continue
                    agent_a = next((a for a in agents if a.id == a_id), None)
                    agent_b = next((a for a in agents if a.id == b_id), None)
                    if agent_a is None or agent_b is None:
                        continue
                    current_cost = np.linalg.norm(agent_a.location - task_a.location) + np.linalg.norm(agent_b.location - task_b.location)
                    swap_cost = np.linalg.norm(agent_a.location - task_b.location) + np.linalg.norm(agent_b.location - task_a.location)
                    message_count += 1
                    if swap_cost < current_cost:
                        assignment_map[a_id] = task_b
                        assignment_map[b_id] = task_a
                        best_cost = best_cost - current_cost + swap_cost
                        improved = True
                        break
                if improved:
                    break
        if iter_count >= self.max_iterations:
            print("Warning: DBBAAlgorithm reached max iterations during local search.")
        assignments = []
        for agent_id, task in assignment_map.items():
            agent_obj = next((a for a in agents if a.id == agent_id), None)
            if agent_obj and task is not None:
                assignments.append((agent_obj, task))
        self.last_computation_time = time.time() - start_time
        self.last_num_iterations = iter_count
        self.last_communication_cost = message_count
        return assignments
