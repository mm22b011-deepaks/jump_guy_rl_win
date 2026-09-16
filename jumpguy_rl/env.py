import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import io
import base64
from PIL import Image

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By

try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False


def crop_canvas(screenshot_path, canvas_rect):
    """Crop the canvas region from a full-page screenshot."""
    img = Image.open(screenshot_path)
    x, y = int(canvas_rect['x']), int(canvas_rect['y'])
    w, h = int(canvas_rect['width']), int(canvas_rect['height'])
    cropped = img.crop((x, y, x + w, y + h))
    return cropped.resize((960, 540), Image.LANCZOS)


def read_score(img):
    """Read score from top-left region using OCR."""
    if not HAS_OCR:
        return 0
    region = img.crop((10, 10, 250, 55))
    text = pytesseract.image_to_string(
        region, config='--psm 7 -c tessedit_char_whitelist=SCORE0123456789 '
    ).strip()
    # Extract digits
    digits = ''.join(c for c in text if c.isdigit())
    return int(digits) if digits else 0


def detect_game_state(img):
    """Detect game state from center screen text using OCR."""
    if not HAS_OCR:
        return 'unknown'
    region = img.crop((300, 100, 700, 220))
    text = pytesseract.image_to_string(region, config='--psm 7').strip().upper()
    if 'GAME OVER' in text:
        return 'gameover'
    if 'PRESS' in text or 'TAP' in text or 'JUMP' in text:
        return 'ready'
    return 'running'


def detect_player_y(img):
    """Detect player Y position by finding blue pants pixels."""
    arr = np.array(img)
    # Scan expected player region (x: 140-280, y: 250-430)
    region = arr[250:430, 140:280]
    # Blue pants: high blue, low red, moderate green
    blue_mask = (region[:, :, 2] > 150) & (region[:, :, 0] < 120) & (region[:, :, 1] < 160)
    coords = np.where(blue_mask)
    if len(coords[0]) > 5:
        return 250 + np.mean(coords[0])
    return 388.0  # ground level


def detect_obstacles(img):
    """Detect obstacle positions from ground region using color analysis."""
    arr = np.array(img)
    ground = arr[350:420, :, :]

    # Find colorful blocks: high saturation, not too dark, not white
    r, g, b = ground[:, :, 0].astype(int), ground[:, :, 1].astype(int), ground[:, :, 2].astype(int)
    sat = np.max(ground[:, :, :3], axis=2).astype(int) - np.min(ground[:, :, :3], axis=2).astype(int)
    colorful = sat > 40
    bright = np.max(ground[:, :, :3], axis=2) > 80
    not_white = np.min(ground[:, :, :3], axis=2) < 200
    not_ground_line = np.abs(np.mean(ground[:, :, :3], axis=2) - 50) > 20  # not the ground line itself

    mask = colorful & bright & not_white
    coords = np.where(mask)

    if len(coords[1]) == 0:
        return []

    # Cluster x coordinates
    xs = np.sort(np.unique(coords[1]))
    clusters = []
    if len(xs) > 0:
        cluster = [xs[0]]
        for x in xs[1:]:
            if x - cluster[-1] > 15:
                clusters.append(cluster)
                cluster = [x]
            else:
                cluster.append(x)
        clusters.append(cluster)

    obstacles = []
    for c in clusters:
        mask_x = (coords[1] >= min(c)) & (coords[1] <= max(c))
        obs_y = np.mean(coords[0][mask_x])
        obs_x = np.mean(c)
        width = len(c)
        # Only count if reasonably sized and on ground
        if width > 5 and 10 < obs_y < 70:  # relative to ground crop
            obstacles.append({
                'x': obs_x,
                'y': 350 + obs_y,  # absolute y in 540px image
                'width': width
            })

    return obstacles


class JumpGuyEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, headless=False, screenshot_dir="/tmp"):
        super().__init__()
        self.headless = headless
        self.driver = None
        self.canvas_rect = None
        self.screenshot_dir = screenshot_dir
        self.screenshot_counter = 0
        self.last_score = 0
        self.step_count = 0
        self.max_steps = 3000
        self.episode_count = 0

        # Observation: [player_y_norm, closest_obs_dx, closest_obs_dy, second_obs_dx, second_obs_dy, speed_norm, game_state]
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(7,), dtype=np.float32
        )
        # 0 = do nothing, 1 = jump
        self.action_space = spaces.Discrete(2)

    def _start_browser(self):
        opts = Options()
        if self.headless:
            opts.add_argument("--headless")
        opts.add_argument("--width=960")
        opts.add_argument("--height=600")
        opts.set_preference("media.volume_scale", "0.0")

        self.driver = webdriver.Firefox(options=opts)
        self.driver.set_window_size(960, 600)

        print("[ENV] Opening game...")
        self.driver.get("https://game.jumpguy.net")
        time.sleep(5)

        canvas = self.driver.find_element(By.TAG_NAME, "canvas")
        self.canvas_rect = canvas.rect
        print(f"[ENV] Canvas found: {self.canvas_rect}")

    def _take_screenshot(self):
        """Take screenshot, crop canvas, return PIL Image."""
        path = f"{self.screenshot_dir}/jg_{self.screenshot_counter:06d}.png"
        self.driver.save_screenshot(path)
        self.screenshot_counter += 1
        return crop_canvas(path, self.canvas_rect)

    def _click(self):
        """Send a trusted click to the canvas center."""
        canvas = self.driver.find_element(By.TAG_NAME, "canvas")
        ActionChains(self.driver).move_to_element(canvas).click().perform()

    def _read_state(self):
        """Read game state from screenshot."""
        img = self._take_screenshot()
        score = read_score(img)
        game_state = detect_game_state(img)
        player_y = detect_player_y(img)
        obstacles = detect_obstacles(img)

        return {
            'score': score,
            'game_state': game_state,
            'player_y': player_y,
            'obstacles': obstacles,
            'img': img,
        }

    def _build_obs(self, state):
        if state is None:
            return np.zeros(7, dtype=np.float32)

        player_y = state.get('player_y', 388) / 540.0
        gs = state.get('game_state', 'unknown')
        gs_val = 0.0 if gs == 'ready' else 1.0 if gs == 'running' else 2.0

        obstacles = state.get('obstacles', [])
        # Sort by x, find closest ahead of player (x > 150, since player is at ~211)
        ahead = [o for o in obstacles if o['x'] > 150]
        ahead.sort(key=lambda o: o['x'])

        obs_features = []
        for o in ahead[:2]:
            dx = (o['x'] - 211) / 960.0  # relative to player
            dy = (o['y'] - 388) / 540.0  # relative to ground
            obs_features.extend([dx, dy])

        while len(obs_features) < 4:
            obs_features.extend([1.0, 0.0])

        return np.array(
            [player_y, obs_features[0], obs_features[1],
             obs_features[2], obs_features[3], gs_val / 2.0],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.driver is None:
            self._start_browser()

        self.episode_count += 1
        self.step_count = 0
        self.last_score = 0

        # Read current state
        state = self._read_state()

        # If game over, click to restart
        if state['game_state'] == 'gameover':
            self._click()
            time.sleep(0.5)
            state = self._read_state()

        # If still game over, click again
        if state['game_state'] == 'gameover':
            self._click()
            time.sleep(0.5)
            state = self._read_state()

        # If in running state, we need to wait for ready
        if state['game_state'] == 'running':
            time.sleep(2)
            state = self._read_state()

        # If still not ready, restart by refreshing
        if state['game_state'] not in ('ready', 'running'):
            self.driver.get("https://game.jumpguy.net")
            time.sleep(5)
            canvas = self.driver.find_element(By.TAG_NAME, "canvas")
            self.canvas_rect = canvas.rect
            state = self._read_state()

        obs = self._build_obs(state)
        info = {
            'score': state.get('score', 0),
            'game_state': state.get('game_state', 'unknown'),
        }
        return obs, info

    def step(self, action):
        if action == 1:
            self._click()

        # Wait for physics to update
        time.sleep(0.08)  # ~12 fps for state reading

        state = self._read_state()
        obs = self._build_obs(state)

        score = state.get('score', 0)
        game_state = state.get('game_state', 'unknown')

        # Reward
        reward = 0.1  # survive reward
        if score > self.last_score:
            reward += 10.0 * (score - self.last_score)
            self.last_score = score

        terminated = game_state == 'gameover'
        if terminated:
            reward -= 5.0

        truncated = self.step_count >= self.max_steps
        self.step_count += 1

        info = {'score': score, 'game_state': game_state}

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        if self.driver:
            self.driver.quit()
            self.driver = None
