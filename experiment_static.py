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
    
实验结果将绘制成 PDF 格式图表，采用 Times New Roman 字体和和谐统一的配色，同时将详细数据和汇总数据保存为 JSON 文件，
打印输出时以 “均值 ± 波动” 的形式自动适应数据量级。例如对于
    0.2000000000000000666 ± 0.0000000000000000565
将自动归约显示为
    0.2 ± 0
以免出现过多冗余小数位。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import copy
import statistics
import json
import os
import math

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
def determine_decimal_places(std_value, default=3):
    """
    根据标准差计算需要保留的小数位数，保证至少展示 3 个有效数字。
    当 std_value 为 0 时，返回默认小数位数（default）。
    """
    if std_value == 0:
        return default
    exponent = math.floor(math.log10(abs(std_value)))
    decimals = max(0, 3 - exponent - 1)
    return decimals

def format_metric(mean_val, std_val, tol=1e-10):
    """
    根据均值与标准差生成格式化字符串：
    - 如果 std_val 极小（低于 tol），则认为该指标数值没有明显波动，直接使用科学计数法（保留 3 位有效数字），
      并将标准差显示为 0；
    - 否则根据 std_val 的数量级确定保留的小数位数，并自动剔除无效的尾随 0。
    """
    if abs(std_val) < tol:
        return f"{mean_val:.3g} ± 0"
    else:
        decimals = determine_decimal_places(std_val)
        mean_str = f"{mean_val:.{decimals}f}"
        std_str = f"{std_val:.{decimals}f}"
        # 去除尾随的 0 和可能多余的小数点
        if '.' in mean_str:
            mean_str = mean_str.rstrip('0').rstrip('.')
        if '.' in std_str:
            std_str = std_str.rstrip('0').rstrip('.')
        return f"{mean_str} ± {std_str}"

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
    
    # detailed: 保存每一轮的详细结果，按场景和算法分类
    detailed = {sc["name"]: {alg: [] for alg in algos.keys()} for sc in scenarios}
    
    for sc in scenarios:
        print(f"Running scenario: {sc['name']} (Tasks: {sc['tasks']}, Agents: {sc['agents']}), Runs: {sc['runs']}")
        for run in range(sc["runs"]):
            np.random.seed(run)
            for alg_name, alg_instance in algos.items():
                # 这里直接使用同一算法实例，如有状态问题，可在每轮前重新实例化
                metrics = run_static_simulation(sc["tasks"], sc["agents"], alg_instance)
                detailed[sc["name"]][alg_name].append(metrics)
    
    # concluded: 保存总体统计数据——均值及标准差，自动调整数值显示格式
    concluded = {sc["name"]: {} for sc in scenarios}
    metrics_keys = ['total_cost', 'complete_rate', 'avg_delay', 'fairness', 'comp_time', 'messages']
    
    for sc in scenarios:
        scenario_name = sc["name"]
        for alg_name in algos.keys():
            metric_list = detailed[scenario_name][alg_name]
            metrics_concluded = {}
            for key in metrics_keys:
                values = [m[key] for m in metric_list]
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                metrics_concluded[key] = format_metric(mean_val, std_val)
            concluded[scenario_name][alg_name] = metrics_concluded
            
    # 保存结果到 JSON 文件
    if not os.path.exists("results"):
        os.makedirs("results")
    with open("results/experiment_static_concluded.json", "w") as f:
        json.dump(concluded, f, indent=4)
    with open("results/experiment_static_detailed.json", "w") as f:
        json.dump(detailed, f, indent=4)
    
    # 绘图：每个场景生成一份 PDF 报告图（6 个子图），图中仅展示各指标的平均值
    for sc in scenarios:
        scenario_name = sc["name"]
        alg_names = list(algos.keys())
        total_costs = [np.mean([m["total_cost"] for m in detailed[scenario_name][alg]]) for alg in alg_names]
        complete_rates = [np.mean([m["complete_rate"] for m in detailed[scenario_name][alg]]) for alg in alg_names]
        delays = [np.mean([m["avg_delay"] for m in detailed[scenario_name][alg]]) for alg in alg_names]
        comp_times = [np.mean([m["comp_time"] for m in detailed[scenario_name][alg]]) for alg in alg_names]
        messages = [np.mean([m["messages"] for m in detailed[scenario_name][alg]]) for alg in alg_names]
        fairnesses = [np.mean([m["fairness"] for m in detailed[scenario_name][alg]]) for alg in alg_names]
        
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
        
        # plt.suptitle(f"Static Task Assignment - {scenario_name} Scenario")
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig(f"results/experiment_static_{scenario_name}.pdf")
        plt.close()
    
    # 打印输出实验汇总结果（均值 ± 波动），数值自动归约
    print("Summary of Static Task Assignment Experiment (mean ± std):")
    for sc in scenarios:
        scenario_name = sc["name"]
        print(f"Scenario: {scenario_name}")
        header = "Algorithm\tTotalCost\tCompleteRate\tAvgDelay\tCompTime(s)\tMessages\tFairness"
        print(header)
        for alg in alg_names:
            res = concluded[scenario_name][alg]
            print(f"{alg}\t{res['total_cost']}\t{res['complete_rate']}\t{res['avg_delay']}\t{res['comp_time']}\t{res['messages']}\t{res['fairness']}")
        print("\n")
        
if __name__ == "__main__":
    main()
