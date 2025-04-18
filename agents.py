import numpy as np


class Agent:
    def __init__(self, agent_id, location=(0, 0), speed=1.0, agent_type=None):
        """
        :param agent_id: ID
        :param location: init location (x, y)
        :param speed: move speed of the agent (unitDistance / unitTme)
        :param agent_type: agent type or ability (to select the suitable task)
        """
        self.id = agent_id
        self.location = np.array(location, dtype=float)
        self.speed = speed
        self.type = agent_type
        self.busy = False  # if working on a task
        self.failed = False  # used to simulate a accident faliure

    def travel_time_to(self, task_location):
        dist = np.linalg.norm(self.location - task_location)
        if self.speed <= 0:
            return float("inf")
        return dist / self.speed

    def __repr__(self):
        return f"Agent(id={self.id}, loc={tuple(self.location)}, type={self.type})"
