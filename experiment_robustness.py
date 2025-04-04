#!/usr/bin/env python3
"""
Experiment Robustness Testing
-----------------------------------
目的：
    测试五种任务分配算法在异常情况下的鲁棒性，包括：
      1. Surge 情景：在模拟中期突然发布大量任务（任务骤增）。
      2. Failure 情景：代理在运行过程中随机发生故障，无法分配任务，直到恢复。
      3. Combined 情景：同时出现任务骤增和代理故障。
      
每种情景重复 20 轮，模拟总时长为 100 个时间单位。
      
评价指标：
    1. 总成本：累计代理为完成任务所行驶的距离之和
    2. 完成率：完成任务数 / 总发布任务数
    3. 平均响应延迟：任务从发布到分配的平均等待时间
    4. 计算时间：算法在分配决策上的平均耗时
    5. 通信消息数：累计通信消息数
    6. 分配公平性：各代理获得任务数的标准差
    
结果将绘制为 PDF 图表，统一采用 Times New Roman 字体。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import copy
import statistics
import random

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.family'] = 'Times New Roman'

from tasks import generate_tasks, Task
from agents import Agent
from algorithms.hungarian import HungarianAlgorithm
from algorithms.auction import AuctionAlgorithm
from algorithms.cbba import CBBAAlgorithm
from algorithms.dbba import DBBAAlgorithm
from algorithms.rl_method import RLAlgorithm

# -------------------------
# 辅助函数
# -------------------------
def initialize_agents_with_failure(num_agents):
    """生成代理，并增加状态字段用于故障模拟。"""
    agents = []
    for i in range(num_agents):
        loc = (np.random.rand() * 100, np.random.rand() * 100)
        agent = Agent(agent_id=i, location=loc, speed=1.0, agent_type=None)
        agent.active = True      # 是否正常工作
        agent.fail_until = 0     # 故障恢复时间，0 表示当前正常
        agents.append(agent)
    return agents

def generate_dynamic_tasks(num_tasks, current_time):
    """生成动态任务，发布时间为 current_time。"""
    tasks = []
    for i in range(num_tasks):
        loc = (np.random.rand() * 100, np.random.rand() * 100)
        reward = np.random.rand() * 100
        tasks.append(Task(task_id=i, location=loc, reward=reward, release_time=current_time, duration=1, required_type=None))
    return tasks

def run_robust_simulation(sim_time, publish_interval, tasks_per_interval, agents, algorithm, scenario):
    """
    运行鲁棒性测试仿真。
    参数 scenario: "normal"、"surge"、"failure" 或 "combined"
    - 在 surge 或 combined 情景下，在模拟中期（sim_time/2）突增大量任务。
    - 在 failure 或 combined 情景下，每个时间步代理有一定概率发生故障，故障后在一定时间内恢复。
    """
    current_time = 0
    pending_tasks = []
    total_cost = 0.0
    total_assigned = 0
    delays = []
    comp_times = []
    messages_count = 0
    assignments_per_agent = {agent.id: 0 for agent in agents}
    
    # 故障参数
    fail_prob = 0.1  # 每个时间步，每个代理故障的概率
    recovery_time_range = (3, 7)  # 故障持续时间随机在 3-7 时间单位内
    
    # Surge 参数
    surge_time = sim_time // 2  # 模拟中期发生
    surge_task_count = 50       # 突增任务数量
    
    # 初始化代理状态
    for agent in agents:
        agent.busy = False
        agent.current_task = None
        agent.finish_time = 0
        agent.active = True
    
    while current_time < sim_time:
        # 正常任务发布
        if current_time % publish_interval == 0:
            new_tasks = generate_dynamic_tasks(tasks_per_interval, current_time)
            pending_tasks.extend(new_tasks)
        # Surge 情景：在 surge_time 突然增加大量任务
        if (scenario == "surge" or scenario == "combined") and current_time == surge_time:
            surge_tasks = generate_dynamic_tasks(surge_task_count, current_time)
            pending_tasks.extend(surge_tasks)
        # Failure 情景：代理可能随机故障
        if scenario == "failure" or scenario == "combined":
            for agent in agents:
                if agent.active and random.random() < fail_prob:
                    agent.active = False
                    agent.fail_until = current_time + random.randint(*recovery_time_range)
                if not agent.active and current_time >= agent.fail_until:
                    agent.active = True
        # 检查任务完成：仅在代理 active 时完成任务
        for agent in agents:
            if agent.busy and agent.finish_time <= current_time:
                if agent.active:
                    total_cost += agent.travel_time_to(agent.current_task.location)
                    agent.location = agent.current_task.location
                    agent.current_task.completed = True
                agent.busy = False
                agent.current_task = None
        # 获取空闲且 active 的代理
        free_agents = [agent for agent in agents if (not agent.busy and agent.active)]
        available_tasks = [t for t in pending_tasks if t.release_time <= current_time and not t.assigned]
        if free_agents and available_tasks:
            assignments = algorithm.assign_tasks(available_tasks, free_agents, current_time)
            comp_times.append(algorithm.last_computation_time)
            messages_count += algorithm.last_communication_cost
            for agent, task in assignments:
                agent.busy = True
                agent.current_task = task
                delays.append(current_time - task.release_time)
                task.assigned = True
                assignments_per_agent[agent.id] += 1
                total_assigned += 1
                travel_time = agent.travel_time_to(task.location)
                agent.finish_time = current_time + travel_time + task.duration
                if task in pending_tasks:
                    pending_tasks.remove(task)
        current_time += 1
    total_published = (sim_time // publish_interval) * tasks_per_interval
    if scenario == "surge" or scenario == "combined":
        total_published += surge_task_count
    complete_rate = total_assigned / total_published
    avg_delay = np.mean(delays) if delays else 0.0
    avg_comp_time = np.mean(comp_times) if comp_times else 0.0
    fairness = statistics.stdev(list(assignments_per_agent.values())) if len(assignments_per_agent) > 1 else 0.0
    return {
        'total_cost': total_cost,
        'complete_rate': complete_rate,
        'avg_delay': avg_delay,
        'comp_time': avg_comp_time,
        'messages': messages_count,
        'fairness': fairness
    }

# -------------------------
# 主实验流程
# -------------------------
def main():
    scenarios = [
        {"name": "Normal", "scenario": "normal"},
        {"name": "Surge", "scenario": "surge"},
        {"name": "Failure", "scenario": "failure"},
        {"name": "Combined", "scenario": "combined"}
    ]
    publish_interval = 5
    tasks_per_interval = 5
    sim_time = 100
    runs = 20
    num_agents = 20
    
    algos = {
        "Hungarian": HungarianAlgorithm(),
        "Auction": AuctionAlgorithm(),
        "CBBA": CBBAAlgorithm(),
        "DBBA": DBBAAlgorithm(),
        "RL": RLAlgorithm()
    }
    
    results = {sc["name"]: {alg: [] for alg in algos.keys()} for sc in scenarios}
    
    for sc in scenarios:
        print(f"Running Robustness Scenario: {sc['name']}")
        for run in range(runs):
            np.random.seed(run)
            agents = initialize_agents_with_failure(num_agents)
            for alg_name, alg_instance in algos.items():
                metrics = run_robust_simulation(sim_time, publish_interval, tasks_per_interval, copy.deepcopy(agents), alg_instance, sc["scenario"])
                results[sc["name"]][alg_name].append(metrics)
    
    # 计算平均指标
    avg_results = {sc["name"]: {} for sc in scenarios}
    for sc in scenarios:
        sc_name = sc["name"]
        for alg_name in algos.keys():
            metric_list = results[sc_name][alg_name]
            avg_total_cost = np.mean([m["total_cost"] for m in metric_list])
            avg_complete_rate = np.mean([m["complete_rate"] for m in metric_list])
            avg_delay = np.mean([m["avg_delay"] for m in metric_list])
            avg_comp_time = np.mean([m["comp_time"] for m in metric_list])
            avg_messages = np.mean([m["messages"] for m in metric_list])
            avg_fairness = np.mean([m["fairness"] for m in metric_list])
            avg_results[sc_name][alg_name] = {
                'total_cost': avg_total_cost,
                'complete_rate': avg_complete_rate,
                'avg_delay': avg_delay,
                'comp_time': avg_comp_time,
                'messages': avg_messages,
                'fairness': avg_fairness
            }
    
    # 绘制图表，每种鲁棒性情景生成一份 PDF 文件
    for sc in scenarios:
        sc_name = sc["name"]
        alg_names = list(algos.keys())
        total_costs = [avg_results[sc_name][alg]['total_cost'] for alg in alg_names]
        complete_rates = [avg_results[sc_name][alg]['complete_rate'] for alg in alg_names]
        delays = [avg_results[sc_name][alg]['avg_delay'] for alg in alg_names]
        comp_times = [avg_results[sc_name][alg]['comp_time'] for alg in alg_names]
        messages = [avg_results[sc_name][alg]['messages'] for alg in alg_names]
        fairnesses = [avg_results[sc_name][alg]['fairness'] for alg in alg_names]
        
        fig, axs = plt.subplots(2, 3, figsize=(15,8))
        axs[0,0].bar(alg_names, total_costs, color='tab:blue')
        axs[0,0].set_title('Average Total Cost')
        axs[0,0].set_ylabel('Cost')
        
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
        axs[1,1].set_ylabel('Messages')
        
        axs[1,2].bar(alg_names, fairnesses, color='tab:gray')
        axs[1,2].set_title('Fairness (Std Dev)')
        axs[1,2].set_ylabel('Std Dev')
        
        plt.suptitle(f"Robustness Scenario - {sc_name}")
        plt.tight_layout(rect=[0,0.03,1,0.95])
        plt.savefig(f"results/experiment_robustness_{sc_name}.pdf")
        plt.close()
    
    # 输出汇总结果
    print("Summary of Robustness Experiment:")
    for sc in scenarios:
        sc_name = sc["name"]
        print(f"Scenario: {sc_name}")
        print("Algorithm\tTotalCost\tCompleteRate\tAvgDelay\tCompTime(s)\tMessages\tFairness")
        for alg in alg_names:
            res = avg_results[sc_name][alg]
            print(f"{alg}\t{res['total_cost']:.2f}\t{res['complete_rate']:.2f}\t{res['avg_delay']:.2f}\t{res['comp_time']:.4f}\t{res['messages']:.1f}\t{res['fairness']:.2f}")
        print("\n")
    
if __name__ == "__main__":
    main()
