class GhostEnv: 
    def __init__(self, maze_layout, ghost_position_start, pacman_position_start, max_steps=10000):
        """
        Inputs:
        maze_layout: 2D numpy array where
            0 represents a wall,
            1 represents a path,
            2 represents a pellet,
        ghost_position_start: tuple (row, col) representing the starting position of the ghost.
        pacman_position_start: tuple (row, col) representing the starting position of the pacman.
        max_steps: maximum steps per episode before termination
        """
        self.maze_layout = maze_layout
        self.ghost_position_start = ghost_position_start
        self.pacman_position_start = pacman_position_start
        self.max_steps = max_steps
        self.steps = 0

        # find all intersections, since that is the only place we can make a decision
        self.intersection = self.get_intersections()

        # initialize Pac-Man AI
        self.pacman = SimplePacmanAI(maze_layout, pacman_position_start)

        # initialize ghost position
        self.ghost_pos = ghost_position_start

        # initialize the starting state
        self.current_state = self.reset()
        
    # break removed for when a pellet is eaten, added for pac-man winning
    def step(self, ghost_action):
        """
        Execute one step of the environment.
        ghost_action: tuple (dr, dc) representing direction change
        Returns: (state, reward, done)
        """
        reward = 0
        done = False
        self.steps += 1
        while True:

            # update pac-man's knowledge of ghost position
            self.pacman.set_ghost_position(self.ghost_pos)

            # get pacman's next move
            pacman_pos, pellet_eaten = self.pacman.get_next_move()

            # move ghost based on algorithm (with validation)
            row, col = self.ghost_pos
            new_ghost_pos = (row + ghost_action[0], col + ghost_action[1])

            # Only move if valid (not a wall)
            if self._is_valid_position(new_ghost_pos):
                self.ghost_pos = new_ghost_pos
            else: 
                reward += -20 # Penalty for hitting a dead end
                done = False
                break

            # Check collision with Pac-Man
            if pacman_pos == self.ghost_pos:
                reward += 200  # Ghost caught Pac-Man!
                done = True
                break
            elif pellet_eaten:
                reward += -5  # Pac-Man ate a pellet
            else:
                # Stronger penalty per step Pac-Man remains uncaught
                reward += -2.0


            # Check if we've reached an intersection (decision point)
            if self.ghost_pos in self.intersection:
                # print('Reached intersection')
                break

            # Check if max steps reached
            if self.steps >= self.max_steps:
                # print('Reached max steps')
                done = True
                break

        # Check if all pellets eaten (Pac-Man won)
        if not self.pacman.pellets:
            done = True
            reward += -100  # Penalty for Pac-Man winning

        return self.get_state_representation(self.ghost_pos, pacman_pos), reward, done, pacman_pos == self.ghost_pos


    def _is_valid_position(self, pos):
        """Check if position is valid (not a wall, within bounds)"""
        row, col = pos
        if 0 <= row < self.maze_layout.shape[0] and 0 <= col < self.maze_layout.shape[1]:
            return self.maze_layout[row][col] != 0  # 0 = wall
        return False
    
    # added for ghost agent, yun
    def _get_valid_moves_from_position(self, position):
        """Get all valid adjacent moves from position (up, down, left, right)"""
        row, col = position
        valid_moves= []

        # Check all 4 directions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_position = (row + dr, col + dc)
            if self._is_valid_position(new_position):
                valid_moves.append((dr, dc))

        # If no valid moves, stay in place
        if not valid_moves:
            valid_moves.append((0,0))

        return valid_moves
    
    def reset(self):
        """Reset environment for a new episode"""
        self.ghost_pos = self.ghost_position_start
        self.pacman.position = self.pacman_position_start
        self.pacman.pellets = self.pacman._extract_pellets_from_maze()
        self.score = 0
        self.steps = 0
        return self.get_state_representation(self.ghost_pos, self.pacman_position_start)

    def get_state_representation(self, ghost_position, pacman_position):
        return ghost_position, pacman_position

    def get_intersections(self):
        intersections = []
        rows, cols = self.maze_layout.shape
        for row in range(rows):
            for col in range(cols):
                cell = self.maze_layout[row][col]

                # a wall cannot be an intersection
                if cell == 0:
                    continue

                # check if there are at least 3 paths connected to this cell
                count = 0
                if row > 0 and self.maze_layout[row - 1][col] != 0:
                    count += 1
                if row < rows - 1 and self.maze_layout[row + 1][col] != 0:
                    count += 1
                if col > 0 and self.maze_layout[row][col - 1] != 0:
                    count += 1
                if col < cols - 1 and self.maze_layout[row][col + 1] != 0:
                    count += 1
                if count >= 3:
                    intersections.append((row, col))
        return intersections
    
class SimplePacmanAI:
    """Simple AI Pac-Man for ghost training - NOT used in actual game"""
    def __init__(self, maze_layout, start_pos, ghost_pos=None):
        self.position = start_pos
        self.maze_layout = maze_layout
        self.ghost_pos = ghost_pos
        self.pellets = self._extract_pellets_from_maze()

    def _extract_pellets_from_maze(self):
        """Extract all pellet positions from maze layout (value = 2)"""
        pellets = []
        rows, cols = self.maze_layout.shape
        for row in range(rows):
            for col in range(cols):
                if self.maze_layout[row][col] == 2:  # 2 = pellet
                    pellets.append((row, col))
        return pellets

    def set_ghost_position(self, ghost_pos):
        """Update ghost position for Pac-Man to consider"""
        self.ghost_pos = ghost_pos

    def get_next_move(self):
        """
        Weighted score: move towards pellet while avoiding ghost.
        Score = -pellet_distance + (ghost_distance * 2)
        """
        if not self.pellets:
            return self.position, False  # No pellets, stay still

        # Get all valid moves from current position
        possible_moves = self._get_valid_moves()

        if not possible_moves:
            return self.position, False  # Stuck, can't move

        # Find nearest pellet
        nearest_pellet = min(self.pellets,
                            key=lambda p: self._distance(self.position, p))

        # Score each possible move
        best_move = None
        best_score = float('-inf')

        for move in possible_moves:
            pellet_dist = self._distance(move, nearest_pellet)

            # Calculate score based on pellet distance and ghost distance
            score = -pellet_dist  # Negative: closer pellet is better

            # If ghost exists, factor in ghost distance
            if self.ghost_pos is not None:
                ghost_dist = self._distance(move, self.ghost_pos)
                score += ghost_dist  # Positive: farther ghost is better

            if score > best_score:
                best_score = score
                best_move = move

        # Move to best position
        self.position = best_move

        # Remove pellet if eaten
        pellet_eaten = False
        if self.position in self.pellets:
            self.pellets.remove(self.position)
            pellet_eaten = True

        return self.position, pellet_eaten

    def _get_valid_moves(self):
        """Get all valid adjacent moves (up, down, left, right)"""
        row, col = self.position
        valid_moves = []

        # Check all 4 directions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (row + dr, col + dc)
            if self._is_valid_move(new_pos):
                valid_moves.append(new_pos)

        # If no valid moves, stay in place
        if not valid_moves:
            valid_moves.append(self.position)

        return valid_moves

    def _distance(self, pos1, pos2):
        """Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_valid_move(self, pos):
        """Check if position is valid (not a wall, within bounds)"""
        row, col = pos
        if 0 <= row < self.maze_layout.shape[0] and 0 <= col < self.maze_layout.shape[1]:
            return self.maze_layout[row][col] != 0  # 0 = wall
        return False

