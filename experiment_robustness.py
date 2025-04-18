#!/usr/bin/env python3
"""
Experiment Robustness Testing
-----------------------------------
Objective:
    Evaluate the robustness of four task assignment algorithms under abnormal conditions:
      1. Surge scenario: a large burst of tasks is released midway through the simulation.
      2. Failure scenario: agents randomly fail during execution and recover after a delay.
      3. Combined scenario: both task surge and agent failures occur.

All tasks and agents are defined in a 100x100 area. Results are saved as PDF with Times New Roman font.
"""

import copy
import itertools
import os
import random
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

# PDF font configuration
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]

# global task ID counter
task_id_counter = itertools.count()


def initialize_agents_with_failure(num_agents):
    """
    Create agents with failure state tracking.
    Each agent has:
      - active: whether it can perform tasks
      - fail_until: time when it recovers
    """
    agents = []
    for i in range(num_agents):
        loc = (np.random.rand() * 100, np.random.rand() * 100)
        agent = Agent(agent_id=i, location=loc, speed=1.0, agent_type=None)
        agent.active = True
        agent.fail_until = 0
        agent.busy = False
        agent.current_task = None
        agent.finish_time = 0
        agents.append(agent)
    return agents


def generate_tasks(count, current_time):
    """
    Generate `count` new tasks at `current_time` with zero duration.
    """
    tasks = []
    for _ in range(count):
        tid = next(task_id_counter)
        loc = (np.random.rand() * 100, np.random.rand() * 100)
        task = Task(
            task_id=tid,
            location=loc,
            reward=0,
            release_time=current_time,
            duration=0,
            required_type=None,
        )
        tasks.append(task)
    return tasks


def run_robustness_simulation(
    sim_time, publish_interval, tasks_per_interval, agents, algorithm, scenario
):
    """
    Run one robustness test simulation.
    scenario: 'normal', 'surge', 'failure', or 'combined'
    """
    current_time = 0
    pending_tasks = []
    total_cost = 0.0
    total_assigned = 0
    delays = []
    comp_times = []
    messages = 0
    assignments_per_agent = {ag.id: 0 for ag in agents}

    # failure settings
    failure_probability = 0.1
    recovery_range = (3, 7)
    # surge settings
    surge_time = sim_time // 2
    surge_count = 50

    while current_time < sim_time:
        # publish regular tasks
        if current_time % publish_interval == 0:
            pending_tasks.extend(generate_tasks(tasks_per_interval, current_time))
        # surge event
        if scenario in ("surge", "combined") and current_time == surge_time:
            pending_tasks.extend(generate_tasks(surge_count, current_time))
        # failure event
        if scenario in ("failure", "combined"):
            for ag in agents:
                if ag.active and random.random() < failure_probability:
                    ag.active = False
                    ag.fail_until = current_time + random.randint(*recovery_range)
                if not ag.active and current_time >= ag.fail_until:
                    ag.active = True

        # complete tasks for agents that finish now
        for ag in agents:
            if ag.busy and ag.finish_time <= current_time:
                if ag.active:
                    total_cost += ag.travel_time_to(ag.current_task.location)
                    ag.location = ag.current_task.location
                    ag.current_task.completed = True
                ag.busy = False
                ag.current_task = None

        # find free and active agents
        free_agents = [ag for ag in agents if not ag.busy and ag.active]
        available_tasks = [
            t
            for t in pending_tasks
            if t.release_time <= current_time and not t.assigned
        ]
        if free_agents and available_tasks:
            assignments = algorithm.assign_tasks(
                available_tasks, free_agents, current_time
            )
            comp_times.append(algorithm.last_computation_time)
            messages += algorithm.last_communication_cost
            for ag, task in assignments:
                ag.busy = True
                ag.current_task = task
                delays.append(current_time - task.release_time)
                task.assigned = True
                assignments_per_agent[ag.id] += 1
                total_assigned += 1
                travel_time = ag.travel_time_to(task.location)
                ag.finish_time = current_time + travel_time
                pending_tasks.remove(task)

        current_time += 1

    # compute metrics
    total_published = (sim_time // publish_interval) * tasks_per_interval
    if scenario in ("surge", "combined"):
        total_published += surge_count
    complete_rate = total_assigned / total_published if total_published else 0.0
    avg_delay = np.mean(delays) if delays else 0.0
    avg_comp_time = np.mean(comp_times) if comp_times else 0.0
    fairness = (
        statistics.pstdev(list(assignments_per_agent.values()))
        if len(assignments_per_agent) > 1
        else 0.0
    )

    return {
        "total_cost": total_cost,
        "complete_rate": complete_rate,
        "avg_delay": avg_delay,
        "comp_time": avg_comp_time,
        "messages": messages,
        "fairness": fairness,
    }


def main():
    # create results directory
    os.makedirs("results", exist_ok=True)

    scenarios = [
        {"name": "Normal", "scenario": "normal"},
        {"name": "Surge", "scenario": "surge"},
        {"name": "Failure", "scenario": "failure"},
        {"name": "Combined", "scenario": "combined"},
    ]
    publish_interval = 5
    tasks_per_interval = 5
    sim_time = 100
    runs = 20
    num_agents = 20

    algorithms = {
        "Hungarian": HungarianAlgorithm(),
        "ε-Auction": AuctionAlgorithm(),
        "CBBA": CBBAAlgorithm(),
        "AuctionPlus2Opt": AuctionPlus2Opt(),
    }

    results = {sc["name"]: {alg: [] for alg in algorithms} for sc in scenarios}

    for sc in scenarios:
        print(f"Running scenario: {sc['name']}")
        for r in range(runs):
            np.random.seed(r)
            agents = initialize_agents_with_failure(num_agents)
            for name, alg in algorithms.items():
                metrics = run_robustness_simulation(
                    sim_time,
                    publish_interval,
                    tasks_per_interval,
                    copy.deepcopy(agents),
                    alg,
                    sc["scenario"],
                )
                results[sc["name"]][name].append(metrics)

    # compute average metrics
    avg_results = {}
    for sc in scenarios:
        sc_name = sc["name"]
        avg_results[sc_name] = {}
        for name in algorithms:
            mlist = results[sc_name][name]
            avg_results[sc_name][name] = {
                k: np.mean([m[k] for m in mlist]) for k in mlist[0]
            }

    # plot and save PDFs
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

    # prepare a color cycle for bars
    color_cycle = plt.cm.tab10.colors

    for sc in scenarios:
        sc_name = sc["name"]
        alg_names = list(algorithms.keys())
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        for idx, (ax, key, title) in enumerate(zip(axs.flat, keys, titles)):
            vals = [avg_results[sc_name][alg][key] for alg in alg_names]
            colors = color_cycle[: len(alg_names)]
            ax.bar(alg_names, vals, color=colors)
            ax.set_title(title)
            ax.set_xticks(range(len(alg_names)))
            ax.set_xticklabels(alg_names, rotation=45, ha="right")
        plt.suptitle(f"Robustness Scenario - {sc_name}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"results/robustness_{sc_name}.pdf")
        plt.close()

    # print summary
    print("Summary of Robustness Testing:")
    for sc in scenarios:
        print(f"-- {sc['name']} --")
        for alg in algorithms:
            print(f"{alg}: {avg_results[sc['name']][alg]}")
        print()


if __name__ == "__main__":
    main()
