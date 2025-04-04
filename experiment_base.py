import numpy as np
import matplotlib.pyplot as plt
import copy
import heapq
import itertools  # 新增，用于生成自增的事件ID

from tasks import generate_tasks, Task
from agents import Agent
from network import CommNetwork
from algorithms.hungarian import HungarianAlgorithm
from algorithms.auction import AuctionAlgorithm
from algorithms.cbba import CBBAAlgorithm
from algorithms.dbba import DBBAAlgorithm
from algorithms.rl_method import RLAlgorithm

# 全局事件计数器，用于在事件队列中打破平局
_event_counter = itertools.count()

def run_simulation(tasks_data, agents_data, algorithm):
    """
    在给定任务和代理数据下运行一次仿真，使用指定算法进行任务分配。
    返回包含各项指标的字典。
    """
    # 初始化任务和代理的深拷贝，避免影响原始数据
    if isinstance(tasks_data[0], Task):
        tasks = [copy.deepcopy(t) for t in tasks_data]
    else:
        tasks = [Task(tid, (x, y), reward, release, duration, req_type) 
                 for (tid, x, y, reward, release, duration, req_type) in tasks_data]
    agents = []
    for a in agents_data:
        if isinstance(a, Agent):
            agents.append(copy.deepcopy(a))
        else:
            (aid, x, y, speed, atype, fail_time) = a
            agent = Agent(aid, (x, y), speed, atype)
            agent.fail_time = fail_time  # 可选：代理的失效时间
            agents.append(agent)
    # 按任务发布时间排序
    tasks.sort(key=lambda t: t.release_time)
    # 构建事件队列，事件元组结构为 (time, sub_priority, event_id, event_type, object)
    events = []
    # 调度任务发布事件，sub_priority: 0
    for task in tasks:
        event_id = next(_event_counter)
        heapq.heappush(events, (task.release_time, 0, event_id, 'task_release', task))
    # 调度代理故障事件，sub_priority: 1
    for agent in agents:
        if hasattr(agent, 'fail_time') and agent.fail_time is not None:
            event_id = next(_event_counter)
            heapq.heappush(events, (agent.fail_time, 1, event_id, 'agent_fail', agent))
    current_time = 0.0
    total_distance = 0.0  # 总成本（距离之和）
    tasks_completed = 0
    tasks_failed = 0
    waiting_tasks = []  # 等待分配的任务列表
    # 初始化代理状态
    for agent in agents:
        agent.current_task = None
        agent.finish_time = None
    # 算法性能累计指标
    total_computation_time = 0.0
    total_iterations = 0
    total_messages = 0

    # 主事件循环
    while events:
        # 取出队列中最早的事件
        time_val, sub_priority, eid, etype, obj = heapq.heappop(events)
        current_time = time_val  # 前进仿真时间到该事件时刻
        # 如果有多个事件在同一时间，将它们全部取出处理
        events_at_time = [(etype, obj)]
        while events and events[0][0] == current_time:
            t2, sub_priority2, eid2, etype2, obj2 = heapq.heappop(events)
            events_at_time.append((etype2, obj2))
        # 处理当前时间的所有事件
        for etype, obj in events_at_time:
            if etype == 'task_release':
                # 有新任务发布
                task = obj
                if not task.assigned and not task.completed:
                    waiting_tasks.append(task)
            elif etype == 'agent_fail':
                # 有代理失效
                agent = obj
                agent.failed = True
                agent.busy = False
                # 如果该代理正在执行任务，则将该任务标记为未完成并重新加入等待队列
                if hasattr(agent, 'current_task') and agent.current_task:
                    failed_task = agent.current_task
                    failed_task.assigned = False
                    if not failed_task.completed:
                        waiting_tasks.append(failed_task)
                    agent.current_task = None
            elif etype == 'task_finish':
                # 任务完成事件
                agent, task = obj
                # 如果代理在完成之前失效，则跳过（任务未真正完成）
                if agent.failed:
                    continue
                # 标记任务完成
                task.completed = True
                tasks_completed += 1
                # 代理变为空闲
                agent.busy = False
                agent.current_task = None
                agent.finish_time = None
                # 更新代理位置到任务位置（假设代理移动到任务地点）
                # 累积距离成本（这里简单计算）
                dist = np.linalg.norm(agent.location - np.array(task.location))
                total_distance += dist

        # 当前时间事件处理完毕，检查是否有空闲代理和待分配任务，执行任务分配
        free_agents = [agent for agent in agents if (not agent.busy and not getattr(agent, 'failed', False))]
        # 过滤等待队列中未完成且未分配的任务（确保任务已发布）
        waiting_tasks = [t for t in waiting_tasks if (not t.completed and not t.assigned and t.release_time <= current_time)]
        if free_agents and waiting_tasks:
            # 调用任务分配算法为空闲代理分配任务
            assignments = algorithm.assign_tasks(waiting_tasks, free_agents, current_time=current_time)
            # 获取算法执行过程的指标并累加
            if hasattr(algorithm, 'last_computation_time'):
                total_computation_time += algorithm.last_computation_time
            if hasattr(algorithm, 'last_num_iterations'):
                total_iterations += algorithm.last_num_iterations
            if hasattr(algorithm, 'last_communication_cost'):
                total_messages += algorithm.last_communication_cost
            # 将分配结果应用到代理和任务上
            for agent, task in assignments:
                agent.busy = True
                agent.current_task = task
                task.assigned = True
                # 计算完成该任务的时间（行驶时间 + 执行时间）
                travel_time = agent.travel_time_to(task.location)
                finish_time = current_time + travel_time + getattr(task, 'duration', 0)
                agent.finish_time = finish_time
                # 调度任务完成事件，sub_priority: 2
                event_id = next(_event_counter)
                heapq.heappush(events, (finish_time, 2, event_id, 'task_finish', (agent, task)))
        # 如果事件队列空了但仍有未完成的任务，认为这些任务无法完成，计入失败
        if not events:
            remaining_tasks = [t for t in waiting_tasks if not t.completed]
            if remaining_tasks:
                tasks_failed += len(remaining_tasks)
            break

    # 统计最终未完成的任务（未完成的均记为失败）
    for t in tasks:
        if not t.completed:
            tasks_failed += 1
    metrics = {
        'total_cost': total_distance,
        'tasks_completed': tasks_completed,
        'tasks_failed': tasks_failed,
        'total_tasks': len(tasks),
        'computation_time': total_computation_time,
        'iterations': total_iterations,
        'messages': total_messages
    }
    return metrics

if __name__ == "__main__":
    # 设置实验参数
    np.random.seed(42)
    num_agents = 3
    num_tasks = 5
    # 生成初始场景的数据（代理位置、任务位置等）
    agents_data = [(i, np.random.rand()*50, np.random.rand()*50, 1.0, None, None) for i in range(num_agents)]
    tasks_data = [(j, np.random.rand()*50, np.random.rand()*50, 0, 0, 0, None) for j in range(num_tasks)]

    # 初始化五种算法实例
    algorithms = {
        "Hungarian": HungarianAlgorithm(),
        "Auction": AuctionAlgorithm(),
        "CBBA": CBBAAlgorithm(),
        "DBBA": DBBAAlgorithm(),
        "RL": RLAlgorithm()
    }

    # 执行多轮随机实验以统计平均性能
    runs = 5  # 重复运行次数
    all_results = {name: [] for name in algorithms}
    for r in range(runs):
        # 每轮生成新随机场景（确保各算法在相同条件下比较）
        tasks_data_run = [(j, np.random.rand()*50, np.random.rand()*50, 0, 0, 0, None) for j in range(num_tasks)]
        agents_data_run = [(i, np.random.rand()*50, np.random.rand()*50, 1.0, None, None) for i in range(num_agents)]
        for name, algo in algorithms.items():
            metrics = run_simulation(tasks_data_run, agents_data_run, algo)
            all_results[name].append(metrics)

    # 计算各算法指标的均值和方差
    summary = {}
    for name, metrics_list in all_results.items():
        if not metrics_list:
            continue
        avg_metrics = {}
        var_metrics = {}
        for key in ['total_cost', 'tasks_completed', 'computation_time', 'messages']:
            values = [m[key] for m in metrics_list]
            avg_metrics[key] = np.mean(values)
            var_metrics[key] = np.var(values)
        summary[name] = (avg_metrics, var_metrics)

    # 打印结果对比表格
    print("算法\t平均总成本\t总成本方差\t平均完成任务数\t平均计算时间\t平均消息数")
    for name, (avg, var) in summary.items():
        print(f"{name}\t{avg['total_cost']:.2f}\t{var['total_cost']:.2f}\t{avg['tasks_completed']:.1f}/{num_tasks}\t"
              f"{avg['computation_time']:.4f}s\t{avg['messages']:.1f}")

    # 绘制总成本对比柱状图（以第1轮结果为示例）
    alg_names = list(all_results.keys())
    costs_run1 = [all_results[a][0]['total_cost'] for a in alg_names]
    plt.figure(figsize=(6,4))
    plt.bar(alg_names, costs_run1, color='skyblue')
    plt.title("Total Cost by Algorithm (Run 1)")
    plt.ylabel("Total Distance Cost")
    plt.tight_layout()
    plt.savefig("results/cost_comparison.png")
    print("Cost comparison chart saved as cost_comparison.png")
