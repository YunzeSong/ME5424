#!/usr/bin/env python3
"""
Experiment Static Task Assignment
-----------------------------------
目的：
    比较五种任务分配算法（匈牙利、拍卖、CBBA、DBBA、强化学习）在静态场景下的性能。
    静态场景为一次性任务分配，所有任务在 t=0 时发布。
    
场景设置：
    - Small：10 个任务 / 5 个代理
    - Medium：50 个任务 / 15 个代理
    - Large：200 个任务 / 40 个代理
    每种场景重复 30 轮，以保证统计显著性。
    
评价指标：
    1. 总成本（所有代理从当前位置到任务地点的行驶时间之和）
    2. 任务完成率（分配成功任务数/总任务数）
    3. 平均响应延迟（任务从发布到分配的时间；静态场景通常接近 0）
    4. 计算时间（算法求解分配所花费的时间）
    5. 通信消息数（算法内部迭代过程中累计的通信消息数）
    6. 分配公平性（各代理获得任务数的标准差）
    
实验结果将绘制成 PDF 格式图表，采用 Times New Roman 字体和和谐统一的配色。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import copy
import statistics

# 设置绘图字体为 Times New Roman 并保存为 PDF
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times', 'DejaVu Serif']

# 导入本地模块
from tasks import generate_tasks, Task
from agents import Agent
from algorithms.hungarian import HungarianAlgorithm
from algorithms.auction import AuctionAlgorithm
from algorithms.cbba import CBBAAlgorithm
from algorithms.dbba import DBBAAlgorithm
from algorithms.rl_method import RLAlgorithm

# -------------------------
# 辅助函数定义
# -------------------------
def initialize_static_tasks(num_tasks):
    """生成静态任务，所有任务在 t=0 时发布。"""
    tasks = []
    for i in range(num_tasks):
        loc = (np.random.rand() * 100, np.random.rand() * 100)
        reward = np.random.rand() * 100
        tasks.append(Task(task_id=i, location=loc, reward=reward, release_time=0, duration=0, required_type=None))
    return tasks

def initialize_static_agents(num_agents):
    """生成代理，随机位置，速度固定为 1.0。"""
    agents = []
    for i in range(num_agents):
        loc = (np.random.rand() * 100, np.random.rand() * 100)
        agents.append(Agent(agent_id=i, location=loc, speed=1.0, agent_type=None))
    return agents

def compute_total_cost(assignments):
    """计算所有分配的总成本（行驶时间之和）。"""
    total = 0.0
    for agent, task in assignments:
        total += agent.travel_time_to(task.location)
    return total

def compute_fairness(assignments, agents):
    """计算分配公平性：各代理分配任务数的标准差。"""
    count = {agent.id: 0 for agent in agents}
    for agent, task in assignments:
        count[agent.id] += 1
    counts = list(count.values())
    if len(counts) <= 1:
        return 0.0
    return statistics.stdev(counts)

def compute_average_delay(assignments):
    """静态场景中，任务发布均在 t=0，响应延迟为 0。"""
    if len(assignments) == 0:
        return 0.0
    return 0.0

def run_static_simulation(num_tasks, num_agents, algorithm):
    """对静态场景进行一次实验模拟，返回各项指标。"""
    tasks = initialize_static_tasks(num_tasks)
    agents = initialize_static_agents(num_agents)
    assignments = algorithm.assign_tasks(tasks, agents, current_time=0)
    total_cost = compute_total_cost(assignments)
    complete_rate = len(assignments) / num_tasks  # 分配成功任务比例
    avg_delay = compute_average_delay(assignments)
    fairness = compute_fairness(assignments, agents)
    comp_time = algorithm.last_computation_time if hasattr(algorithm, 'last_computation_time') else 0.0
    messages = algorithm.last_communication_cost if hasattr(algorithm, 'last_communication_cost') else 0
    return {
        'total_cost': total_cost,
        'complete_rate': complete_rate,
        'avg_delay': avg_delay,
        'fairness': fairness,
        'comp_time': comp_time,
        'messages': messages
    }

# -------------------------
# 主实验流程
# -------------------------
def main():
    scenarios = [
        {"name": "Small", "tasks": 10, "agents": 5, "runs": 30},
        {"name": "Medium", "tasks": 50, "agents": 15, "runs": 30},
        {"name": "Large", "tasks": 200, "agents": 40, "runs": 30}
    ]
    
    # 初始化各算法实例
    algos = {
        "Hungarian": HungarianAlgorithm(),
        "Auction": AuctionAlgorithm(),
        "CBBA": CBBAAlgorithm(),
        "DBBA": DBBAAlgorithm(),
        "RL": RLAlgorithm()
    }
    
    # 结果字典：results[scenario_name][algorithm] = [指标字典, ...]
    results = {sc["name"]: {alg: [] for alg in algos.keys()} for sc in scenarios}
    
    for sc in scenarios:
        print(f"Running scenario: {sc['name']} (Tasks: {sc['tasks']}, Agents: {sc['agents']}), Runs: {sc['runs']}")
        for run in range(sc["runs"]):
            np.random.seed(run)
            for alg_name, alg_instance in algos.items():
                # 这里直接使用同一算法实例，如有状态问题，可在每轮前重新实例化
                metrics = run_static_simulation(sc["tasks"], sc["agents"], alg_instance)
                results[sc["name"]][alg_name].append(metrics)
    
    # 计算平均指标
    avg_results = {sc["name"]: {} for sc in scenarios}
    for sc in scenarios:
        scenario_name = sc["name"]
        for alg_name in algos.keys():
            metric_list = results[scenario_name][alg_name]
            avg_total_cost = np.mean([m["total_cost"] for m in metric_list])
            avg_complete_rate = np.mean([m["complete_rate"] for m in metric_list])
            avg_delay = np.mean([m["avg_delay"] for m in metric_list])
            avg_fairness = np.mean([m["fairness"] for m in metric_list])
            avg_comp_time = np.mean([m["comp_time"] for m in metric_list])
            avg_messages = np.mean([m["messages"] for m in metric_list])
            avg_results[scenario_name][alg_name] = {
                'total_cost': avg_total_cost,
                'complete_rate': avg_complete_rate,
                'avg_delay': avg_delay,
                'fairness': avg_fairness,
                'comp_time': avg_comp_time,
                'messages': avg_messages
            }
    
    # 绘图：每个场景生成一份 PDF 报告图（6 个子图）
    for sc in scenarios:
        scenario_name = sc["name"]
        alg_names = list(algos.keys())
        total_costs = [avg_results[scenario_name][alg]['total_cost'] for alg in alg_names]
        complete_rates = [avg_results[scenario_name][alg]['complete_rate'] for alg in alg_names]
        delays = [avg_results[scenario_name][alg]['avg_delay'] for alg in alg_names]
        comp_times = [avg_results[scenario_name][alg]['comp_time'] for alg in alg_names]
        messages = [avg_results[scenario_name][alg]['messages'] for alg in alg_names]
        fairnesses = [avg_results[scenario_name][alg]['fairness'] for alg in alg_names]
        
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        axs[0,0].bar(alg_names, total_costs, color='tab:blue')
        axs[0,0].set_title('Average Total Cost')
        axs[0,0].set_ylabel('Cost (Travel Time)')
        
        axs[0,1].bar(alg_names, complete_rates, color='tab:green')
        axs[0,1].set_title('Assignment Completion Rate')
        axs[0,1].set_ylabel('Rate')
        
        axs[0,2].bar(alg_names, delays, color='tab:orange')
        axs[0,2].set_title('Average Response Delay')
        axs[0,2].set_ylabel('Delay (s)')
        
        axs[1,0].bar(alg_names, comp_times, color='tab:red')
        axs[1,0].set_title('Average Computation Time')
        axs[1,0].set_ylabel('Time (s)')
        
        axs[1,1].bar(alg_names, messages, color='tab:purple')
        axs[1,1].set_title('Average Communication Messages')
        axs[1,1].set_ylabel('Message Count')
        
        axs[1,2].bar(alg_names, fairnesses, color='tab:gray')
        axs[1,2].set_title('Fairness (Std Dev of Assignments)')
        axs[1,2].set_ylabel('Std Dev')
        
        plt.suptitle(f"Static Task Assignment - {scenario_name} Scenario")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"experiment_static_{scenario_name}.pdf")
        plt.close()
    
    # 输出实验汇总表
    print("Summary of Static Task Assignment Experiment:")
    for sc in scenarios:
        scenario_name = sc["name"]
        print(f"Scenario: {scenario_name}")
        print("Algorithm\tTotalCost\tCompleteRate\tCompTime(s)\tMessages\tFairness")
        for alg in alg_names:
            res = avg_results[scenario_name][alg]
            print(f"{alg}\t{res['total_cost']:.2f}\t{res['complete_rate']:.2f}\t{res['comp_time']:.4f}\t{res['messages']:.1f}\t{res['fairness']:.2f}")
        print("\n")
        
if __name__ == "__main__":
    main()
