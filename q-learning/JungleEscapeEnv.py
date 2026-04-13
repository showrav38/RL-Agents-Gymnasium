import sys
import pygame
import numpy as np
import gymnasium as gym

class JungleEscapeEnv(gym.Env):
    def __init__(self, grid_size=6, fast_mode=False, random_initialization=False):
        super().__init__()
        self.grid_size = grid_size
        self.cell_size = 100
        self.fast_mode = fast_mode
        self.random_initialization = random_initialization

        self.start_pos = np.array([0, 0])
        self.exit_pos = np.array([grid_size - 1, grid_size - 1])

        self.danger_cells = [
            np.array([2, 2]),
            np.array([3, 4])
        ]

        self.food_cells = []
        self._init_foods()

        self.state = self.start_pos.copy()
        self.life = 40

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32)

        self.done = False
        self.reward = 0
        self.info = {}

        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.cell_size * self.grid_size, self.cell_size * self.grid_size))
        pygame.display.set_caption("Jungle Escape")

        self.emoji_font = pygame.font.SysFont("Segoe UI Emoji", 48)
        self.hud_font = pygame.font.SysFont("Arial", 24)

    def _init_foods(self):
        positions = [
            np.array([0, 1]),
            np.array([1, 0]),
            np.array([1, 2]),
            np.array([2, 1]),
            np.array([3, 0])
        ]
        emojis = ["🍌", "🍎", "🍇", "🍉", "🍒"]
        self.food_cells = list(zip(positions, emojis))

    def reset(self, seed=None, options=None):
        if self.random_initialization:
            while True:
                pos = np.array([
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size)
                ])
                # avoid danger or goal positions
                if not any(np.array_equal(pos, d) for d in self.danger_cells) \
                   and not np.array_equal(pos, self.exit_pos):
                    self.state = pos
                    break
        else:
            self.state = self.start_pos.copy()

        self.life = 40
        self.done = False
        self.reward = 0
        self._init_foods()
        self.info = {"Life": self.life, "Distance to exit": self._distance_to_exit()}
        return self.state.copy(), self.info

    def step(self, action):
        if self.done:
            return self.state.copy(), self.done, self.reward, self.info

        moved = False

        if action == 0 and self.state[0] > 0:       # Up
            self.state[0] -= 1; moved = True
        elif action == 1 and self.state[0] < self.grid_size - 1:  # Down
            self.state[0] += 1; moved = True
        elif action == 2 and self.state[1] < self.grid_size - 1:  # Right
            self.state[1] += 1; moved = True
        elif action == 3 and self.state[1] > 0:     # Left
            self.state[1] -= 1; moved = True

        self.reward = 0
        if moved:
            self.life -= 1
            self.reward = -1
        else:
            self.reward = -2  # ❗ Penalty for invalid move

        for i, (food_pos, emoji) in enumerate(self.food_cells):
            if np.array_equal(self.state, food_pos):
                self.life += 6
                self.reward += 2
                del self.food_cells[i]
                break

        for danger in self.danger_cells:
            if np.array_equal(self.state, danger):
                self.life -= 5
                self.reward -= 10
                self.done = True
                break

        if np.array_equal(self.state, self.exit_pos):
            self.reward += 20
            self.done = True

        if self.life <= 0:
            self.life = 0
            self.reward -= 5
            self.done = True

        self.info = {"Life": self.life, "Distance to exit": round(self._distance_to_exit(), 2)}
        return self.state.copy(), self.done, self.reward, self.info

    def render(self):
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()

        self.screen.fill((230, 255, 230))

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        for food_pos, emoji in self.food_cells:
            self._draw_emoji(food_pos, emoji)

        self._draw_emoji(self.danger_cells[0], "🐍")
        self._draw_emoji(self.danger_cells[1], "🐅")
        self._draw_emoji(self.exit_pos, "🏕")
        self._draw_emoji(self.state, "🧍")

        life_text = self.hud_font.render(f"Life: {self.life}", True, (0, 0, 0))
        self.screen.blit(life_text, (5, 5))

        pygame.display.flip()

        if not self.fast_mode:
            pygame.time.wait(100)

    def _draw_emoji(self, pos, emoji):
        emoji_surface = self.emoji_font.render(emoji, True, (0, 0, 0))
        x = pos[1] * self.cell_size + self.cell_size // 4
        y = pos[0] * self.cell_size + self.cell_size // 6
        self.screen.blit(emoji_surface, (x, y))

    def _distance_to_exit(self):
        return np.linalg.norm(self.state - self.exit_pos)

    def close(self):
        pygame.quit()
