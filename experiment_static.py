#!/usr/bin/env python3
"""
Experiment Static Task Assignment
-----------------------------------
Objective:
    Compare five task assignment algorithms (Hungarian, ε-Auction, CBBA, AuctionPlus2Opt, RL) in a static scenario
    where all tasks arrive at t=0.

Scenarios:
    - Small: 10 tasks / 5 agents
    - Medium: 50 tasks / 15 agents
    - Large: 200 tasks / 40 agents
    Each scenario is run 30 times for statistical significance.

Metrics:
    1. Total cost (sum of travel time)
    2. Completion rate (#assigned / #tasks)
    3. Average response delay (≈0 in static)
    4. Computation time
    5. Communication messages count
    6. Fairness (std dev of #tasks per agent)

Results are plotted as PDF with Times New Roman font.
"""

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
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]


def initialize_static_tasks(num_tasks):
    tasks = []
    for i in range(num_tasks):
        loc = (np.random.rand() * 100, np.random.rand() * 100)
        tasks.append(
            Task(
                task_id=i,
                location=loc,
                reward=0,
                release_time=0,
                duration=0,
                required_type=None,
            )
        )
    return tasks


def initialize_static_agents(num_agents):
    agents = []
    for i in range(num_agents):
        loc = (np.random.rand() * 100, np.random.rand() * 100)
        agents.append(Agent(agent_id=i, location=loc, speed=1.0, agent_type=None))
    return agents


def compute_total_cost(assignments):
    return sum(agent.travel_time_to(task.location) for agent, task in assignments)


def compute_fairness(assignments, agents):
    counts = {ag.id: 0 for ag in agents}
    for ag, task in assignments:
        counts[ag.id] += 1
    vals = list(counts.values())
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def run_static_simulation(num_tasks, num_agents, algorithm):
    tasks = initialize_static_tasks(num_tasks)
    agents = initialize_static_agents(num_agents)
    assignments = algorithm.assign_tasks(tasks, agents, current_time=0)
    total_cost = compute_total_cost(assignments)
    complete_rate = len(assignments) / num_tasks
    avg_delay = 0.0
    fairness = compute_fairness(assignments, agents)
    comp_time = getattr(algorithm, "last_computation_time", 0.0)
    messages = getattr(algorithm, "last_communication_cost", 0)
    return {
        "total_cost": total_cost,
        "complete_rate": complete_rate,
        "avg_delay": avg_delay,
        "fairness": fairness,
        "comp_time": comp_time,
        "messages": messages,
    }


def main():
    scenarios = [("Small", 10, 5), ("Medium", 50, 15), ("Large", 200, 40)]
    runs = 30

    algorithms = {
        "Hungarian": HungarianAlgorithm(),
        "ε-Auction": AuctionAlgorithm(),
        "CBBA": CBBAAlgorithm(),
        "AuctionPlus2Opt": AuctionPlus2Opt(),
        # 'RL': RLAlgorithm(),
    }

    results = {name: {alg: [] for alg in algorithms} for name, _, _ in scenarios}

    for name, tasks_n, agents_n in scenarios:
        for r in range(runs):
            np.random.seed(r)
            for alg_name, alg in algorithms.items():
                metrics = run_static_simulation(tasks_n, agents_n, alg)
                results[name][alg_name].append(metrics)

    # compute averages
    avg = {name: {} for name, _, _ in scenarios}
    for name, data in results.items():
        for alg_name, mlist in data.items():
            avg[name][alg_name] = {k: np.mean([m[k] for m in mlist]) for k in mlist[0]}

    # plotting setup
    os.makedirs("results", exist_ok=True)
    color_map = {
        "Hungarian": "#FF6565",
        "ε-Auction": "#E6FF65",
        "AuctionPlus2Opt": "#65DCFF",
        "CBBA": "#65FF74",
    }

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

    # (Inside the main function, plotting loop)
    for name in avg:  # Iterate through scenarios ('Small', 'Medium', 'Large')
        algs = list(algorithms.keys())
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        axs_flat = axs.flat  # Get a flat iterator for easy indexing

        for i, (k, t) in enumerate(
            zip(keys, titles)
        ):  # i is the index, k is metric key, t is title
            ax = axs_flat[i]  # Select the current subplot
            vals = [
                avg[name][alg][k] for alg in algs
            ]  # Get average values for this metric
            cols = [
                color_map.get(alg, "#BEBEBE") for alg in algs
            ]  # Get colors, use gray if missing

            x_pos = np.arange(len(algs))  # Define numerical positions for bars

            ax.bar(
                x_pos, vals, color=cols
            )  # Create the bar chart using numerical positions
            ax.set_title(t)  # Set subplot title

            ax.set_xticks(x_pos)  # Set the positions of the ticks
            ax.set_xticklabels(
                algs, rotation=45, ha="right"
            )  # Set the labels for the ticks

            # Optional: Add y-axis label for clarity
            ax.set_ylabel("Value")

            # Optional: Add value labels on top of bars for readability
            for j, val in enumerate(vals):
                # Format the label, e.g., 2 decimal places for floats
                label = f"{val:.2f}" if isinstance(val, float) else f"{val}"
                ax.text(j, val, label, ha="center", va="bottom", fontsize=8)

        # Remove or uncomment the note as needed
        # note = (
        #     "Note: Hungarian, ε-Auction, AuctionPlus2Opt are offline batch algorithms; "
        #     "they can be rerun at each time step in dynamic scenarios."
        # )
        # fig.text(0.5, 0.01, note, ha="center", fontsize=8) # Uncomment if using the note

        plt.suptitle(
            f"Static Task Assignment Comparison - Scenario: {name}"
        )  # More descriptive title
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout
        plt.savefig(f"results/static_{name}.pdf")  # Slightly more descriptive filename
        plt.close(fig)  # Close the figure to free memory


if __name__ == "__main__":
    main()
