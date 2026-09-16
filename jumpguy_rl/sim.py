import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random


class JumpGuySim(gym.Env):
    """
    Local physics simulation of Jump Guy, matching the exact game mechanics.
    Used for fast RL training without browser overhead.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Game constants (from source code)
        self.WIDTH = 960
        self.HEIGHT = 540
        self.GROUND_Y = 388  # Math.floor(540 * 0.72)
        self.PLAYER_X = 211  # 960 * 0.22
        self.PLAYER_W = 72
        self.PLAYER_H = 84

        # Physics
        self.GRAVITY = 1800  # px/s^2
        self.JUMP_VELOCITY = -700  # px/s (upward)
        self.COYOTE_TIME_MS = 100
        self.JUMP_BUFFER_MS = 130

        # Obstacles
        self.INITIAL_SPEED = 260  # px/s
        self.MAX_SPEED = 520
        self.SPEED_ACCEL = 4.5  # px/s per second
        self.INITIAL_COOLDOWN = 800  # ms (after reset)
        self.OBSTACLE_MIN_SIZE = 28
        self.OBSTACLE_MAX_SIZE = 34

        # Observation: [player_y_norm, player_vy_norm, grounded, obs1_dx, obs1_dy, obs2_dx, obs2_dy, speed_norm]
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        self.reset_sim()

    def reset_sim(self):
        self.player_y = float(self.GROUND_Y)
        self.player_vy = 0.0
        self.grounded = True
        self.state = 'ready'  # ready, running, gameover
        self.score = 0
        self.obstacles = []
        self.spawn_cooldown = self.INITIAL_COOLDOWN
        self.speed = self.INITIAL_SPEED
        self.time_elapsed = 0.0
        self.last_grounded_time = 0.0
        self.jump_queued = False
        self.jump_queue_time = -999.0
        self.dt = 1.0 / 60.0  # 60 fps simulation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_sim()
        return self._get_obs(), {'score': 0, 'game_state': 'ready'}

    def _spawn_obstacle(self):
        size = random.randint(self.OBSTACLE_MIN_SIZE, self.OBSTACLE_MAX_SIZE)
        x = self.WIDTH + random.randint(40, 120)
        self.obstacles.append({
            'x': float(x),
            'y': float(self.GROUND_Y),
            'size': size,
            'passed': False,
        })

    def _get_obs(self):
        # Find obstacles ahead of player
        ahead = [o for o in self.obstacles if not o['passed'] and o['x'] > self.PLAYER_X]
        ahead.sort(key=lambda o: o['x'])

        obs_features = []
        for o in ahead[:2]:
            dx = (o['x'] - self.PLAYER_X) / self.WIDTH
            dy = (o['y'] - self.GROUND_Y) / self.HEIGHT
            obs_features.extend([dx, dy])

        while len(obs_features) < 4:
            obs_features.extend([1.0, 0.0])

        player_y_norm = self.player_y / self.HEIGHT
        player_vy_norm = self.player_vy / 1000.0
        grounded_val = 1.0 if self.grounded else 0.0
        speed_norm = self.speed / self.MAX_SPEED

        return np.array([
            player_y_norm, player_vy_norm, grounded_val,
            obs_features[0], obs_features[1],
            obs_features[2], obs_features[3],
            speed_norm,
        ], dtype=np.float32)

    def step(self, action):
        dt = self.dt
        dt_ms = dt * 1000

        # Handle jump queue
        if action == 1:
            self.jump_queued = True
            self.jump_queue_time = self.time_elapsed * 1000

        # Physics update
        if self.state == 'running' or self.state == 'ready':
            # Apply gravity
            self.player_vy += self.GRAVITY * dt
            self.player_y += self.player_vy * dt

            # Ground collision
            if self.player_y >= self.GROUND_Y:
                self.player_y = float(self.GROUND_Y)
                self.player_vy = 0.0
                self.grounded = True
                self.last_grounded_time = self.time_elapsed * 1000
            else:
                self.grounded = False

            # Try to execute queued jump
            if self.jump_queued:
                time_since_queue = self.time_elapsed * 1000 - self.jump_queue_time
                time_since_grounded = self.time_elapsed * 1000 - self.last_grounded_time
                if time_since_queue <= self.JUMP_BUFFER_MS and time_since_grounded <= self.COYOTE_TIME_MS:
                    self.player_vy = self.JUMP_VELOCITY
                    self.grounded = False
                    self.jump_queued = False
                    if self.state == 'ready':
                        self.state = 'running'

        # Update obstacles (only when running)
        reward = 0.0
        if self.state == 'running':
            self.spawn_cooldown -= dt_ms
            if self.spawn_cooldown <= 0:
                self._spawn_obstacle()
                offset = random.randint(-120, 240)
                self.spawn_cooldown = max(700, min(1500, 1200 + offset))

            self.speed = min(self.speed + self.SPEED_ACCEL * dt, self.MAX_SPEED)

            # Move obstacles
            for obs in self.obstacles:
                obs['x'] -= self.speed * dt

            # Remove off-screen obstacles
            self.obstacles = [o for o in self.obstacles if o['x'] > -50]

            # Collision detection (AABB)
            player_rect = (
                self.PLAYER_X - self.PLAYER_W / 2,
                self.player_y - self.PLAYER_H,
                self.PLAYER_W,
                self.PLAYER_H,
            )

            # Find the closest obstacle ahead of the player
            closest_obs_dist = float('inf')
            for obs in self.obstacles:
                obs_x = obs['x']
                dist = obs_x - self.PLAYER_X
                if dist > 0 and dist < closest_obs_dist:
                    closest_obs_dist = dist

                obs_rect = (
                    obs['x'] - obs['size'] / 2,
                    obs['y'] - obs['size'],
                    obs['size'],
                    obs['size'],
                )

                # AABB intersection
                if (player_rect[0] < obs_rect[0] + obs_rect[2] and
                    player_rect[0] + player_rect[2] > obs_rect[0] and
                    player_rect[1] < obs_rect[1] + obs_rect[3] and
                    player_rect[1] + player_rect[3] > obs_rect[1]):
                    self.state = 'gameover'
                    reward = -100.0
                    break

                # Score: obstacle passes when its right edge passes player's left edge
                if not obs['passed'] and obs['x'] + obs['size'] / 2 < self.PLAYER_X - self.PLAYER_W / 2:
                    obs['passed'] = True
                    self.score += 1
                    reward += 50.0

            # --- Penalty shaping based on obstacle distance ---
            # Critical distance: how far ahead an obstacle needs to be for a jump
            # to clear it. Scales with speed.
            # Jump takes ~0.78s, obstacle travels speed * 0.78 during that time.
            # We want to jump when obstacle is within that window.
            jump_duration = 0.78
            critical_dist = self.speed * jump_duration * 0.6  # 60% of jump travel

            obstacle_in_range = closest_obs_dist < critical_dist
            obstacle_far = closest_obs_dist > critical_dist * 2.5

            jumped_this_frame = (action == 1)

            if obstacle_in_range and not jumped_this_frame and self.grounded:
                # PENALTY: obstacle is here and you didn't jump — you will die
                penalty = -5.0 * (1.0 - closest_obs_dist / critical_dist)
                reward += penalty

            if jumped_this_frame and obstacle_far and self.grounded:
                # PENALTY: jumped when nothing is nearby — wasted jump, you'll
                # be vulnerable when the real obstacle arrives
                reward -= 2.0

        self.time_elapsed += dt

        # Small survival reward in running state
        if self.state == 'running':
            reward += 0.1

        # Penalize staying in ready state (not starting)
        if self.state == 'ready':
            reward += -0.05

        terminated = self.state == 'gameover'
        if terminated and reward > -100.0:
            reward = -100.0

        info = {
            'score': self.score,
            'game_state': self.state,
            'player_y': self.player_y,
            'speed': self.speed,
        }

        return self._get_obs(), reward, terminated, False, info

    def render(self):
        # ASCII rendering for debugging
        screen = [[' ' for _ in range(80)] for _ in range(30)]

        # Ground line
        ground_row = int(self.GROUND_Y / (self.HEIGHT / 30))
        for x in range(80):
            screen[ground_row][x] = '-'

        # Player
        player_col = int(self.PLAYER_X / (self.WIDTH / 80))
        player_row = int(self.player_y / (self.HEIGHT / 30))
        if 0 <= player_row < 30 and 0 <= player_col < 80:
            screen[player_row][player_col] = 'P'

        # Obstacles
        for obs in self.obstacles:
            obs_col = int(obs['x'] / (self.WIDTH / 80))
            obs_row = int(obs['y'] / (self.HEIGHT / 30))
            if 0 <= obs_row < 30 and 0 <= obs_col < 80:
                screen[obs_row][obs_col] = '#'

        print(f"Score: {self.score:5d} | State: {self.state:8s} | Speed: {self.speed:.0f} | Player Y: {self.player_y:.0f}")
        for row in screen:
            print(''.join(row))
