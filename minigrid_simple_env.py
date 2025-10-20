from __future__ import annotations
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import  Goal
from minigrid.minigrid_env import MiniGridEnv
import random

class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=19,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.size = size
        self.key_positions = []
        self.lava_positions = []

        self.start_agent_pos=(1,1)
        
        mission_space = MissionSpace(mission_func=self._gen_mission)
        
        if max_steps is None:
            max_steps = 4 * size**2
        
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Reach the goal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        
        # Place walls in straight lines
        # Vertical walls
        ##for y in range(1, height-1):
        #    self.put_obj(Wall(), width // 2, y)
        
        # Horizontal walls
        #for x in range(1, width-1):
        #    self.put_obj(Wall(), x, height//2)
        
        # Create openings in the walls
        #openings = [(width//2,5),(width//2,15),(5,height//2),(15,height//2),]
        
        #for x, y in openings:
        #    self.grid.set(x, y, None)

        
        # Place a goal square in the bottom-right corner
        self.goal_pos = (width - 2, height - 2)
        self.put_obj(Goal(), *self.goal_pos)
        
        self._place_agent()
        
        self.mission = "Reach the goal"

    def _place_agent(self):
        # Evitar colocar al agente cerca del objetivo
        min_distance = self.size // 2  # distancia mínima al goal

        while True:
            x = random.randint(1, self.size - 2)
            y = random.randint(1, self.size - 2)
            pos = (x, y)

            # Calcular distancia Manhattan al goal
            goal_x, goal_y = self.goal_pos
            distance = abs(goal_x - x) + abs(goal_y - y)

            # Asegurarse de que el lugar esté vacío y lejos del objetivo
            if (
                self.grid.get(*pos) is None and
                pos != self.goal_pos and
                distance >= min_distance
            ):
                self.agent_pos = pos
                self.agent_dir = random.randint(0, 3)
                break


    def reset(self, **kwargs):
        #print("resetting")
        self.stepped_floors = set()
        obs = super().reset(**kwargs)
        # self._place_agent()  # Place the agent in a new random position
        return obs

    def step(self, action):
        prev_pos=self.agent_pos
        prev_dir=self.agent_dir
        obs, reward, terminated, truncated, info = super().step(action)

        SIZE = self.size-2

        reward = -0.2  # base penalty

        if self.agent_pos[0] > SIZE//2 and self.agent_pos[1] > SIZE//2:
            reward += 0.3  # incentivo por acercarse al goal

        if prev_dir == self.agent_dir and prev_pos == self.agent_pos:
            reward -= 0.3  # castigo por chocar

        if isinstance(self.grid.get(*self.agent_pos), Goal):
            reward = 10
            terminated = True
            
        return obs, reward, terminated, truncated, info