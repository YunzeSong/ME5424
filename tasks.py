import numpy as np


class Task:
    def __init__(
        self,
        task_id,
        location=(0, 0),
        reward=0,
        release_time=0,
        duration=0,
        required_type=None,
    ):
        """
        :param task_id: ID
        :param location: task location (x, y)
        :param reward: task return
        :param release_time: when it appear
        :param duration: task duration
        :param required_type: what kind of agent does it need
        """
        self.id = task_id
        self.location = np.array(location, dtype=float)
        self.reward = reward
        self.release_time = release_time
        self.duration = duration
        self.required_type = required_type

        self.assigned = False
        self.completed = False

    def __repr__(self):
        return f"Task(id={self.id}, loc={tuple(self.location)}, reward={self.reward}, rel={self.release_time})"


def generate_tasks(
    num_tasks,
    area_size=100,
    max_release_time=0,
    max_duration=0,
    task_types=None,
    seed=None,
):
    """
    randomly create task array,
    :param num_tasks: task amount
    :param area_size: square area (area_size * area_size)
    :param max_release_time: if not 0, tasks may not be released at the beginning
    :param max_duration: it takes `max_duration` time to finish this task
    :param task_types: task type array; if not none, assign a task tupe to agent randomly
    :param seed: rd seed
    :return: array of the Task class
    """
    if seed is not None:
        np.random.seed(seed)
    tasks = []
    for i in range(num_tasks):
        x = np.random.rand() * area_size
        y = np.random.rand() * area_size
        release_time = 0
        if max_release_time > 0:
            release_time = np.random.rand() * max_release_time
        duration = 0
        if max_duration > 0:
            duration = np.random.rand() * max_duration
        reward = np.random.rand() * 100
        t_type = None
        if task_types:
            t_type = np.random.choice(task_types)
        task = Task(
            task_id=i,
            location=(x, y),
            reward=reward,
            release_time=release_time,
            duration=duration,
            required_type=t_type,
        )
        tasks.append(task)
    # sort based on the release time
    tasks.sort(key=lambda t: t.release_time)
    return tasks
