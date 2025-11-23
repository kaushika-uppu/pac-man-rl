import pygame
from pygame.locals import *
from vector import Vector2
from constants import *
from entity import Entity
from modes import ModeController
from sprites import GhostSprites
import json

Action_to_direc = {
    (-1, 0): UP,
    (1, 0): DOWN,
    (0, -1): LEFT,
    (0, 1): RIGHT
}

class Ghost(Entity):
    def __init__(self, node, pacman = None, blinky = None):
        Entity.__init__(self, node)
        self.name = GHOST
        self.points = 200
        self.goal = Vector2()
        self.directionMethod = self.goalDirection
        self.pacman = pacman
        self.mode = ModeController(self)
        self.blinky = blinky
        self.homeNode = node

        # added for QL 
        self._ql_policy = None
        self._ql_policy_path = "QL_policy.json"

    def update(self, dt):
        self.sprites.update(dt)
        self.mode.update(dt)
        if self.mode.current is SCATTER:
            self.scatter()
        elif self.mode.current is CHASE:
            self.chase()
        Entity.update(self, dt)

    def reset(self):
        Entity.reset(self)
        self.points = 200
        self.directionMethod = self.goalDirection
    
    def scatter(self):
        self.goal = Vector2()

    def chase(self):
        self.goal = self.pacman.position
    
    def startFreight(self):
        self.mode.setFreightMode()
        if self.mode.current == FREIGHT:
            self.setSpeed(50)
            self.directionMethod = self.randomDirection
    
    def normalMode(self):
        self.setSpeed(100)
        self.directionMethod = self.goalDirection
        self.homeNode.denyAccess(DOWN, self)

    def spawn(self):
        self.goal = self.spawnNode.position

    def setSpawnNode(self, node):
        self.spawnNode = node
    
    def startSpawn(self):
        self.mode.setSpawnMode()
        if self.mode.current == SPAWN:
            self.setSpeed(150)
            self.directionMethod = self.goalDirection
            self.spawn()
    
    # added for QL
    def load_QL_policy(self, filename):
        with open(filename, "r") as f:
            raw = json.load(f)
        return {eval(k): tuple(v) for k, v in raw.items()}

    def _ensure_policy_loaded(self):
        if self._ql_policy is None:
            try:
                self._ql_policy = self.load_QL_policy(self._ql_policy_path)
                print(f"Loaded QL policy with {len(self._ql_policy)} entries")
            except Exception as e:
                print("Failed to load QL policy", e)
                self._ql_policy = {} 

# classes for the four different ghosts

class Blinky(Ghost):
    def __init__(self, node, pacman = None, blinky = None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = BLINKY
        self.color = RED
        self.sprites = GhostSprites(self)

    # added for QL
    def chase(self):
        # Load policy once
        self._ensure_policy_loaded()

        ghost_row = int(self.position.y // TILE_HEIGHT)
        ghost_col = int(self.position.x // TILE_WIDTH)
        pac_row = int(self.pacman.position.y // TILE_HEIGHT)
        pac_col = int(self.pacman.position.x // TILE_WIDTH)

        state = ((ghost_row, ghost_col), (pac_row, pac_col))
        action = self._ql_policy.get(state)

        if action is not None:
            print("choosing from policy")
            direction = Action_to_direc.get(action)

            # print("current position ", self.position)
            # print(action, " chosen in policy")
            # print(direction, " in ui")
            # print(f"Policy action (dr,dc)={action} → UI dir={direction} ({'UP' if direction==UP else 'DOWN' if direction==DOWN else 'LEFT' if direction==LEFT else 'RIGHT' if direction==RIGHT else direction})")
            
            # Finds the neighbor node that is in the direction and set that as the goal for ghost
            if direction is not None and direction in self.node.neighbors:
                next_node = self.getNewTarget(direction)
                if next_node is not None:
                    self.goal = next_node.position
                    # print("Goal: ", self.goal)
                    return 

        # if no QL state found
        valid_dir = self.validDirections()
        print("choosing random")
        if valid_dir:
            random = self.randomDirection(valid_dir)
            next_node = self.node.neighbors.get(random)
            if next_node is not None:
                self.goal = next_node.position
            return
        
        print("choosing pacman")
        self.goal = self.pacman.position


class Pinky(Ghost):
    def __init__(self, node, pacman = None, blinky = None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = PINKY
        self.color = PINK
        self.sprites = GhostSprites(self)

    def scatter(self):
        self.goal = Vector2(TILE_WIDTH * NCOLS, 0)

    def chase(self):
        self.goal = self.pacman.position + self.pacman.directions[self.pacman.direction] * TILE_WIDTH * 4

class Inky(Ghost):
    def __init__(self, node, pacman = None, blinky = None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = INKY
        self.color = TEAL
        self.sprites = GhostSprites(self)

    def scatter(self):
        self.goal = Vector2(TILE_WIDTH * NCOLS, TILE_HEIGHT * NROWS)
    
    def chase(self):
        vec1 = self.pacman.position + self.pacman.directions[self.pacman.direction] * TILE_WIDTH * 2
        vec2 = (vec1 - self.blinky.position) * 2
        self.goal = self.blinky.position + vec2

class Clyde(Ghost):
    def __init__(self, node, pacman = None, blinky = None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = CLYDE
        self.color = ORANGE
        self.sprites = GhostSprites(self)
    
    def scatter(self):
        self.goal = Vector2(0, TILE_HEIGHT * NROWS)

    def chase(self):
        dist = self.pacman.position - self.position
        dist_sq = dist.magnitudeSquared()
        if dist_sq <= (TILE_WIDTH * 8) ** 2:
            self.scatter()
        else:
            self.goal = self.pacman.position + self.pacman.directions[self.pacman.direction] * TILE_WIDTH * 4

class GhostGroup(object):
    def __init__(self, node, pacman):
        self.blinky = Blinky(node, pacman)
        self.pinky = Pinky(node, pacman)
        self.inky = Inky(node, pacman, self.blinky)
        self.clyde = Clyde(node, pacman)
        self.ghosts = [self.blinky, self.pinky, self.inky, self.clyde]

    def __iter__(self):
        return iter(self.ghosts)
    
    def update(self, dt):
        for ghost in self:
            ghost.update(dt)

    def startFreight(self):
        for ghost in self:
            ghost.startFreight()
        self.resetPoints()

    def setSpawnNode(self, node):
        for ghost in self:
            ghost.setSpawnNode(node)

    def updatePoints(self):
        for ghost in self:
            ghost.points *= 2

    def resetPoints(self):
        for ghost in self:
            ghost.points = 200
    
    def reset(self):
        for ghost in self:
            ghost.reset()

    def hide(self):
        for ghost in self:
            ghost.visible = False
    
    def show(self):
        for ghost in self:
            ghost.visible = True
    
    def render(self, screen):
        for ghost in self:
            ghost.render(screen)