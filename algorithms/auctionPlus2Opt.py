import time

import numpy as np

from algorithms.auction import AuctionAlgorithm
from algorithms.base import TaskAllocationAlgorithm


class AuctionPlus2Opt(TaskAllocationAlgorithm):
    """
    Auction + 2-opt Task Assignment:
      1. 初始分配：Bertsekas ε-Auction
      2. 局部搜索：成对交换(2-opt)，发现成本下降则立即交换并计数。
    参数：
      max_iterations 限制的是最大交换尝试次数（防止无限循环）。
    """

    def __init__(self, max_iterations=100):
        super().__init__()
        self.max_iterations = max_iterations
        self.initial_allocator = AuctionAlgorithm()

    def assign_tasks(self, tasks, agents, current_time=0):
        start_time = time.time()
        agent_dict = {agent.id: agent for agent in agents}

        initial = self.initial_allocator.assign_tasks(tasks, agents, current_time)
        assignment_map = {agent.id: task for agent, task in initial}

        def total_cost(mapping):
            return sum(
                np.linalg.norm(agent_dict[a_id].location - t.location)
                for a_id, t in mapping.items()
                if a_id in agent_dict and t is not None
            )

        swap_count = 0
        attempts = 0
        improved = True

        # 局部搜索：两两交换
        while improved and attempts < self.max_iterations:
            improved = False
            attempts += 1
            agent_ids = list(assignment_map.keys())
            for i in range(len(agent_ids)):
                for j in range(i + 1, len(agent_ids)):
                    aid, bid = agent_ids[i], agent_ids[j]
                    ta, tb = assignment_map[aid], assignment_map[bid]
                    if ta is None or tb is None:
                        continue
                    # 当前与交换后的成本比较
                    ca = agent_dict[aid].location
                    cb = agent_dict[bid].location
                    curr = np.linalg.norm(ca - ta.location) + np.linalg.norm(
                        cb - tb.location
                    )
                    new = np.linalg.norm(ca - tb.location) + np.linalg.norm(
                        cb - ta.location
                    )
                    if new < curr:
                        # 接受交换
                        assignment_map[aid], assignment_map[bid] = tb, ta
                        swap_count += 1
                        improved = True
                        break
                if improved:
                    break

        if attempts >= self.max_iterations:
            print(
                "Warning: AuctionPlus2Opt reached max_iterations={} without full convergence.".format(
                    self.max_iterations
                )
            )

        result = []
        for aid, task in assignment_map.items():
            if task is not None and aid in agent_dict:
                agent_obj = agent_dict[aid]
                result.append((agent_obj, task))
                setattr(task, "assigned", True)

        self.last_computation_time = time.time() - start_time
        self.last_num_iterations = attempts
        self.last_communication_cost = swap_count
        return result
