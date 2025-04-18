import copy
import heapq
import itertools
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from agents import Agent
from algorithms.auction import AuctionAlgorithm

# from algorithms.auction_plus_2opt import AuctionPlus2Opt
from algorithms.auctionPlus2Opt import AuctionPlus2Opt
from algorithms.cbba import CBBAAlgorithm
from algorithms.hungarian import HungarianAlgorithm
from tasks import Task

# Global event counter used to break ties in the event queue
_event_counter = itertools.count()


def make_tasks(tasks_data):
    """
    Convert raw tasks_data into a sorted list of Task instances.
    Supports both Task objects and tuples.
    """
    tasks = []
    for t in tasks_data:
        if isinstance(t, Task):
            tasks.append(copy.deepcopy(t))
        else:
            tid, x, y, reward, release, duration, req_type = t
            tasks.append(Task(tid, (x, y), reward, release, duration, req_type))
    return sorted(tasks, key=lambda t: t.release_time)


def make_agents(agents_data):
    """
    Convert raw agents_data into a list of Agent instances.
    Supports both Agent objects and tuples.
    """
    agents = []
    for a in agents_data:
        if isinstance(a, Agent):
            ag = copy.deepcopy(a)
        else:
            aid, x, y, speed, atype, fail_time = a
            ag = Agent(aid, (x, y), speed, atype)
            ag.fail_time = fail_time
        agents.append(ag)
    return agents


class SimulationStats:
    """
    Record statistics during simulation.
    """

    def __init__(self):
        self.distance = 0.0
        self.completed = 0
        self.failed = 0
        self.computation_time = 0.0
        self.iterations = 0
        self.messages = 0
        self.total_tasks = 0

    def on_finish(self, agent, task):
        self.completed += 1
        self.distance += np.linalg.norm(agent.location - np.array(task.location))

    def on_assignment_metrics(self, algo):
        self.computation_time += getattr(algo, "last_computation_time", 0)
        self.iterations += getattr(algo, "last_num_iterations", 0)
        self.messages += getattr(algo, "last_communication_cost", 0)

    def finalize(self, all_tasks):
        self.total_tasks = len(all_tasks)
        self.failed = sum(1 for t in all_tasks if not t.completed)

    def to_dict(self):
        return {
            "total_cost": self.distance,
            "tasks_completed": self.completed,
            "tasks_failed": self.failed,
            "total_tasks": self.total_tasks,
            "computation_time": self.computation_time,
            "iterations": self.iterations,
            "messages": self.messages,
        }


def run_simulation(tasks_data, agents_data, algorithm):
    tasks = make_tasks(tasks_data)
    agents = make_agents(agents_data)

    stats = SimulationStats()

    for agent in agents:
        agent.current_task = None
        agent.finish_time = None
        agent.busy = False
        agent.failed = False

    waiting_q = deque()
    waiting_set = set()

    def on_task_release(task, current_time):
        if not task.assigned and not task.completed:
            waiting_q.append(task)
            waiting_set.add(task.id)

    def on_agent_fail(agent, current_time):
        agent.failed = True
        agent.busy = False
        if agent.current_task:
            failed_task = agent.current_task
            failed_task.assigned = False
            if not failed_task.completed and failed_task.id not in waiting_set:
                waiting_q.append(failed_task)
                waiting_set.add(failed_task.id)
            agent.current_task = None

    def on_task_finish(agent_task, current_time):
        agent, task = agent_task
        if agent.failed:
            return
        task.completed = True
        stats.on_finish(agent, task)
        agent.busy = False
        agent.current_task = None
        agent.location = np.array(task.location)

    handlers = {
        "task_release": on_task_release,
        "agent_fail": on_agent_fail,
        "task_finish": on_task_finish,
    }

    events = []
    for task in tasks:
        eid = next(_event_counter)
        heapq.heappush(events, (task.release_time, 0, eid, "task_release", task))
    for agent in agents:
        if hasattr(agent, "fail_time") and agent.fail_time is not None:
            eid = next(_event_counter)
            heapq.heappush(events, (agent.fail_time, 1, eid, "agent_fail", agent))

    current_time = 0.0

    while events:
        time_val, _, _, etype, obj = heapq.heappop(events)
        current_time = time_val
        batch = [(etype, obj)]
        while events and events[0][0] == current_time:
            _, _, _, et2, obj2 = heapq.heappop(events)
            batch.append((et2, obj2))
        for et, o in batch:
            handlers[et](o, current_time)

        free_agents = [ag for ag in agents if not ag.busy and not ag.failed]
        new_q = deque()
        waiting_set.clear()
        for t in waiting_q:
            if not t.completed and not t.assigned and t.release_time <= current_time:
                new_q.append(t)
                waiting_set.add(t.id)
        waiting_q = new_q

        if free_agents and waiting_q:
            assignments = algorithm.assign_tasks(
                list(waiting_q), free_agents, current_time
            )
            stats.on_assignment_metrics(algorithm)
            for agent, task in assignments:
                agent.busy = True
                agent.current_task = task
                task.assigned = True
                waiting_set.discard(task.id)
                travel_time = agent.travel_time_to(task.location)
                finish_time = current_time + travel_time + getattr(task, "duration", 0)
                eid = next(_event_counter)
                heapq.heappush(
                    events, (finish_time, 2, eid, "task_finish", (agent, task))
                )

    stats.finalize(tasks)
    return stats.to_dict()


def summarize(all_results):
    import numpy as np

    summary = {}
    for name, metrics_list in all_results.items():
        arrays = {k: np.array([m[k] for m in metrics_list]) for k in metrics_list[0]}
        summary[name] = {
            k: (arrays[k].mean(), arrays[k].var())
            for k in ["total_cost", "tasks_completed", "computation_time", "messages"]
        }
    print("Algorithm\tAvg Cost\tCost Var\tAvg Completed\tAvg Comp Time\tAvg Messages")
    for name, stats in summary.items():
        avg_cost, var_cost = stats["total_cost"]
        avg_completed, _ = stats["tasks_completed"]
        avg_time, _ = stats["computation_time"]
        avg_messages, _ = stats["messages"]
        print(
            f"{name}\t{avg_cost:.2f}\t{var_cost:.2f}\t{avg_completed:.1f}\t{avg_time:.4f}s\t{avg_messages:.1f}"
        )


def plot_costs(all_results, filename):
    algs = list(all_results.keys())
    costs = [all_results[a][0]["total_cost"] for a in algs]
    plt.figure(figsize=(6, 4))
    plt.bar(algs, costs)

    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(filename)
    plt.title("Total Cost by Algorithm (Run 1)")
    plt.ylabel("Distance Cost")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved cost comparison plot to {filename}")


if __name__ == "__main__":
    np.random.seed(42)
    num_agents = 3
    num_tasks = 5
    runs = 5

    algorithms = {
        "Hungarian": HungarianAlgorithm(),
        "Auction": AuctionAlgorithm(),
        "CBBA": CBBAAlgorithm(),
        "AuctionPlus2Opt": AuctionPlus2Opt(),
    }

    all_results = {name: [] for name in algorithms}
    for _ in range(runs):
        tasks_data = [
            (j, np.random.rand() * 50, np.random.rand() * 50, 0, 0, 0, None)
            for j in range(num_tasks)
        ]
        agents_data = [
            (i, np.random.rand() * 50, np.random.rand() * 50, 1.0, None, None)
            for i in range(num_agents)
        ]
        for name, algo in algorithms.items():
            all_results[name].append(run_simulation(tasks_data, agents_data, algo))

    summarize(all_results)
    plot_costs(all_results, "results/cost_comparison.png")
