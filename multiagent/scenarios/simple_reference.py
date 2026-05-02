import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from multiagent.scenarios.config import indexed_color, positive_int

class Scenario(BaseScenario):
    def make_world(self, num_agents=2, num_landmarks=3):
        num_agents = positive_int("num_agents", num_agents)
        if num_agents < 2:
            raise ValueError("simple_reference needs at least two agents.")
        num_landmarks = positive_int("num_landmarks", num_landmarks)
        world = World()
        # set any world properties first
        world.dim_c = 10
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            # agent.u_noise = 1e-1
            # agent.c_noise = 1e-1
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want other agent to go to the goal landmark
        for i, agent in enumerate(world.agents):
            agent.goal_a = world.agents[(i + 1) % len(world.agents)]
            agent.goal_b = np.random.choice(world.landmarks)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = indexed_color(i, len(world.landmarks))
        # special colors for goals
        for agent in world.agents:
            agent.goal_a.color = agent.goal_b.color
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        if agent.goal_a is None or agent.goal_b is None:
            return 0.0
        dist2 = np.sum(np.square(agent.goal_a.state.p_pos - agent.goal_b.state.p_pos))
        return -dist2 #np.exp(-dist2)

    def observation(self, agent, world):
        # goal positions
        # goal_pos = [np.zeros(world.dim_p), np.zeros(world.dim_p)]
        # if agent.goal_a is not None:
        #     goal_pos[0] = agent.goal_a.state.p_pos - agent.state.p_pos
        # if agent.goal_b is not None:
        #     goal_pos[1] = agent.goal_b.state.p_pos - agent.state.p_pos
        # goal color
        goal_color = [np.zeros(world.dim_color), np.zeros(world.dim_color)]
        # if agent.goal_a is not None:
        #     goal_color[0] = agent.goal_a.color
        if agent.goal_b is not None:
            goal_color[1] = agent.goal_b.color

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks: #world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks: #world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
        return np.concatenate([agent.state.p_vel] + entity_pos + [goal_color[1]] + comm)
