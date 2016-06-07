import numpy as np
import math

import gym
from gym import spaces
from gym.utils import seeding


class Disk(object):
    def __init__(self, x=0.0, y=0.0, r=1.0):
        self.x = x
        self.y = y
        self.r = r
        self.transform = None
        self.shape = None

    def randomize(self, min_state, max_state):
        self.x, self.y = np.random.uniform(min_state, max_state)

    @classmethod
    def random(cls, min_state, max_state, *args):
        d = cls(*args)
        d.randomize(min_state, max_state)
        return d

    def in_collision(self, other):
        return np.sqrt((other.x - self.x) ** 2 + (other.y - self.y) ** 2) < (other.r + self.r)

    def state(self):
        return np.array([self.x, self.y])


class Obstacle(Disk):
    RADIUS = 1.0  # [m]

    def __init__(self, x=0.0, y=0.0):
        super(self.__class__, self).__init__(x, y, self.RADIUS)


class Quad(Disk):
    RADIUS = 0.7  # [m]

    def __init__(self, x=0.0, y=0.0, vx=0.0, vy=0.0):
        super(self.__class__, self).__init__(x, y, self.RADIUS)
        self.vx = vx
        self.vy = vy

    def state(self):
        return np.array([self.x, self.y, self.vx, self.vy])

    def randomize(self, min_state, max_state):
        self.x, self.y, self.vx, self.vy = np.random.uniform(min_state, max_state)

    def step(self, u, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vx += u[0] * dt
        self.vy += u[1] * dt
        # self.x += u[0] * dt
        # self.y += u[1] * dt

    def set_from_state(self, state):
        self.x, self.y, self.vx, self.vy = state

    def clip(self, min_state, max_state):
        self.set_from_state(np.clip(self.state(), min_state, max_state))


class QuadEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    # Map size
    XMIN, XMAX = [0.0, 30.0]  # [m]
    YMIN, YMAX = [0.0, 30.0]  # [m]

    # TODO(hayk): Make this a tighter bound, and the goal part of the state
    GOAL_RADIUS = 4.0  # [m]
    GOAL_POSITION = [25.0, 25.0]  # [m]

    # TODO(hayk): Make this dynamic
    NUM_OBSTACLES = 0

    MAX_ACCEL = np.array([10.0, 10.0])  # [m/s^2]
    MAX_VELOCITY = 10.0  # [m/s]

    def __init__(self):
        self.viewer = None

        self._seed(0)
        np.random.seed(0)

        self.dt = 1.0 / 30  # [s]

        # Generate obstacles
        self.obstacles = []
        self.obstacle_state_min = np.array([self.XMIN, self.YMIN])
        self.obstacle_state_max = np.array([self.XMAX, self.YMAX])

        # Generate goal
        self.goal = Disk(x=self.GOAL_POSITION[0], y=self.GOAL_POSITION[1], r=self.GOAL_RADIUS)

        # Generate quad
        self.quad = Quad(x=12.5, y=12.5)

        # Generate obstacles
        self.obstacles = [Obstacle() for _ in range(self.NUM_OBSTACLES)]
        self.randomize_obstacles()

        # The action space is a 2D commanded acceleration
        self.action_space = spaces.Box(low=-self.MAX_ACCEL, high=self.MAX_ACCEL)

        self.state_min = np.array([self.XMIN, self.YMIN, -self.MAX_VELOCITY, -self.MAX_VELOCITY])
        self.state_max = np.array([self.XMAX, self.YMAX, self.MAX_VELOCITY, self.MAX_VELOCITY])

        # The observation space is the vehicle's position and velocity
        total_min = np.hstack([self.state_min, np.tile(self.obstacle_state_min, [self.NUM_OBSTACLES])])
        total_max = np.hstack([self.state_max, np.tile(self.obstacle_state_max, [self.NUM_OBSTACLES])])
        self.observation_space = spaces.Box(low=total_min, high=total_max)

        self.iteration = 0

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, u):
        px, py, vx, vy = self.quad.state()

        self.last_u = u  # for rendering

        u = np.clip(u, -self.MAX_ACCEL, self.MAX_ACCEL)

        self.quad.step(u, self.dt)

        hit_wall = self.out_of_bounds()
        hit_obstacle = self.in_collision()
        done = hit_obstacle or hit_wall

        self.iteration += 1

        self.quad.clip(self.state_min, self.state_max)

        dist_based_cost = np.sqrt((self.goal.x-self.quad.x)**2 + (self.goal.y-self.quad.y)**2)
        dist_based_positive_bias = np.sqrt(((self.XMAX-self.XMIN)**2) + ((self.YMAX-self.YMIN))**2)

        reward = 0
        if hit_wall:
            reward = -1.0
        elif hit_obstacle:
            reward = -1.0
        elif self.quad.in_collision(self.goal):
            reward = 0.5
        else:
            reward = 0.0

        # reward += 0.01 * (dist_based_positive_bias - dist_based_cost)
        # print('dist based cost: {}, dist based positive bias: {}'.format(dist_based_cost, dist_based_positive_bias))

        return self._get_obs(), reward, done, {}

    def out_of_bounds(self):
        d = self.quad
        return d.x < self.XMIN + d.r or d.x > self.XMAX - d.r or \
               d.y < self.YMIN + d.r or d.y > self.YMAX - d.r

    def in_collision(self):
        return np.any([self.quad.in_collision(o) for o in self.obstacles])

    def randomize_obstacles(self):
        for obstacle in self.obstacles:
            obstacle.randomize(self.obstacle_state_min, self.obstacle_state_max)
            while obstacle.in_collision(self.quad) or obstacle.in_collision(self.goal):
                obstacle.randomize(self.obstacle_state_min, self.obstacle_state_max)

    def _reset(self):
        # Start quad at rest at bottom left
        self.quad.set_from_state(np.array([12.5, 12.5, 0.0, 0.0]))

        self.randomize_obstacles()

        # Always start near the bottom left at rest
        self.last_u = None
        self.iteration = 0
        return self._get_obs()

    def _get_obs(self):
        if self.NUM_OBSTACLES == 0:
            return self.quad.state()
        return np.hstack([self.quad.state(), np.hstack([o.state() - self.quad.state()[:2] for o in self.obstacles])])

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(self.XMIN, self.XMAX, self.YMIN, self.YMAX)

            self.goal.shape = rendering.make_circle(self.goal.r)
            self.goal.shape.set_color(0.5, 1, 0.5)
            self.goal.transform = rendering.Transform()
            self.goal.shape.add_attr(self.goal.transform)
            self.viewer.add_geom(self.goal.shape)

            for obstacle in self.obstacles:
                obstacle.shape = rendering.make_circle(obstacle.r)
                obstacle.shape.set_color(0, 0, 0)
                obstacle.transform = rendering.Transform()
                obstacle.shape.add_attr(obstacle.transform)
                self.viewer.add_geom(obstacle.shape)

            self.quad.shape = rendering.make_circle(self.quad.r)
            self.quad.shape.set_color(0, 0, 1)
            self.quad.transform = rendering.Transform()
            self.quad.shape.add_attr(self.quad.transform)
            self.viewer.add_geom(self.quad.shape)

            self.accel_shape = rendering.make_capsule(1.0, 0.3)
            self.accel_shape.set_color(1, 0, 0)
            self.accel_shape_transform = rendering.Transform()
            self.accel_shape.add_attr(self.accel_shape_transform)
            self.viewer.add_geom(self.accel_shape)

        self.quad.transform.set_translation(self.quad.x, self.quad.y)
        self.goal.transform.set_translation(self.goal.x, self.goal.y)

        for obstacle in self.obstacles:
            obstacle.transform.set_translation(*obstacle.state())

        if self.last_u is not None:
            ux, uy = self.last_u
            u_mag = np.sqrt(ux**2 + uy**2)
            self.accel_shape_transform.set_translation(self.quad.x, self.quad.y)
            self.accel_shape_transform.set_scale(u_mag, 1.0)
            self.accel_shape_transform.set_rotation(math.atan2(uy, ux))

        self.viewer.render()
        if mode == 'rgb_array':
            return self.viewer.get_array()
        elif mode == 'human':
            pass
