import copy
import itertools
import os
import statistics

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from agents import Agent
from algorithms.auction import AuctionAlgorithm
from algorithms.auctionPlus2Opt import AuctionPlus2Opt
from algorithms.cbba import CBBAAlgorithm
from algorithms.hungarian import HungarianAlgorithm
from tasks import Task

# PDF font setup
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.family"] = "Times New Roman"

# global task counter for unique IDs
task_counter = itertools.count()


def initialize_dynamic_agents(num_agents):
    agents = []
    for i in range(num_agents):
        loc = (np.random.rand() * 100, np.random.rand() * 100)
        ag = Agent(agent_id=i, location=loc, speed=1.0, agent_type=None)
        agents.append(ag)
    return agents


def generate_dynamic_tasks(num_tasks, current_time):
    tasks = []
    for _ in range(num_tasks):
        tid = next(task_counter)
        loc = (np.random.rand() * 100, np.random.rand() * 100)
        # no duration for dynamic tasks
        tasks.append(
            Task(
                task_id=tid,
                location=loc,
                reward=0,
                release_time=current_time,
                duration=0,
                required_type=None,
            )
        )
    return tasks


def run_dynamic_simulation(sim_time, interval, tasks_per_interval, agents, algorithm):
    current_time = 0
    pending = []
    total_cost = 0.0
    delays = []
    comp_times = []
    messages = 0
    assign_counts = {ag.id: 0 for ag in agents}
    total_assigned = 0

    # initialize agent state
    for ag in agents:
        ag.busy = False
        ag.current_task = None
        ag.finish_time = 0

    while current_time < sim_time:
        # publish new tasks at intervals
        if current_time % interval == 0:
            pending.extend(generate_dynamic_tasks(tasks_per_interval, current_time))

        # process completed tasks
        for ag in agents:
            if ag.busy and ag.finish_time <= current_time:
                total_cost += ag.travel_time_to(ag.current_task.location)
                ag.location = ag.current_task.location
                ag.busy = False
                ag.current_task = None

        # perform allocation
        free = [ag for ag in agents if not ag.busy]
        available = [
            t for t in pending if (t.release_time <= current_time and not t.assigned)
        ]
        if free and available:
            assigns = algorithm.assign_tasks(available, free, current_time)
            comp_times.append(algorithm.last_computation_time)
            messages += algorithm.last_communication_cost
            for ag, task in assigns:
                ag.busy = True
                ag.current_task = task
                delays.append(current_time - task.release_time)
                task.assigned = True
                assign_counts[ag.id] += 1
                total_assigned += 1
                tt = ag.travel_time_to(task.location)
                ag.finish_time = current_time + tt
                if task in pending:
                    pending.remove(task)

        current_time += 1

    # compute metrics
    total_published = (sim_time // interval) * tasks_per_interval
    complete_rate = total_assigned / total_published if total_published else 0
    avg_delay = np.mean(delays) if delays else 0
    avg_comp = np.mean(comp_times) if comp_times else 0
    fairness = statistics.pstdev(assign_counts.values())

    return {
        "total_cost": total_cost,
        "complete_rate": complete_rate,
        "avg_delay": avg_delay,
        "comp_time": avg_comp,
        "messages": messages,
        "fairness": fairness,
    }


def main():
    scenarios = [
        ("Slow", 10, 2),
        ("Medium", 5, 5),
        ("Fast", 2, 10),
    ]
    sim_time = 100
    runs = 30
    num_agents = 15

    algorithms = {
        "Hungarian": HungarianAlgorithm(),
        "ε‑Auction": AuctionAlgorithm(),
        "CBBA": CBBAAlgorithm(),
        "AuctionPlus2Opt": AuctionPlus2Opt(),
    }

    results = {name: {alg: [] for alg in algorithms} for name, _, _ in scenarios}

    for name, interval, per in scenarios:
        for r in range(runs):
            np.random.seed(r)
            agents = initialize_dynamic_agents(num_agents)
            for alg_name, alg in algorithms.items():
                metrics = run_dynamic_simulation(
                    sim_time, interval, per, copy.deepcopy(agents), alg
                )
                results[name][alg_name].append(metrics)

    # compute averages
    avg = {}
    for name, data in results.items():
        avg[name] = {}
        for alg_name, metrics in data.items():
            avg[name][alg_name] = {
                k: np.mean([m[k] for m in metrics]) for k in metrics[0]
            }

    # plot and save
    os.makedirs("results", exist_ok=True)
    # define color mapping: offline algs in gray shades, CBBA distinct
    color_map = {
        "Hungarian": "#4d4d4d",
        "ε-Auction": "#666666",
        "AuctionPlus2Opt": "#808080",
        "CBBA": "#1f77b4",
    }

    for name in avg:
        algs = list(algorithms.keys())
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        keys = [
            "total_cost",
            "complete_rate",
            "avg_delay",
            "comp_time",
            "messages",
            "fairness",
        ]
        titles = [
            "Total Cost",
            "Completion Rate",
            "Avg Delay",
            "Computation Time",
            "Messages",
            "Fairness",
        ]
        for ax, key, title in zip(axs.flat, keys, titles):
            values = [avg[name][alg][key] for alg in algs]
            colors = [color_map[alg] for alg in algs]
            ax.bar(algs, values, color=colors)
            ax.set_title(title)
            ax.set_xticklabels(algs, rotation=45, ha="right")
        note = (
            "Note: Hungarian, ε-Auction, AuctionPlus2Opt are offline batch algorithms; "
            "they can be rerun at each time step."
        )
        fig.text(0.5, 0.01, note, ha="center", fontsize=8)
        plt.suptitle(f"Dynamic Task Flow: {name}", y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"results/dynamic_{name}.pdf")
        plt.close()

    # summary print
    for name in avg:
        print(f"--- Scenario: {name} ---")
        for alg in algorithms:
            print(f"{alg}: {avg[name][alg]}")


if __name__ == "__main__":
    main()
