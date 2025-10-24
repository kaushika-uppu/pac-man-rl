class GhostEnv: 
    def __init__(self, maze_layout, ghost_position_start, pacman_position_start):
        """
        Inputs:
        maze_layout: 2D numpy array where 
            0 represents a wall, 
            1 represents a path,
            2 represents a pellet,
            3 represents a fruit.
        ghost_position_start: tuple (row, col) representing the starting position of the ghost.
        pacman_position_start: tuple (row, col) representing the starting position of the pacman.
        """
        self.maze_layout = maze_layout
        self.ghost_position_start = ghost_position_start
        self.pacman_position_start = pacman_position_start

        # find all intersections, since that is the only place we can make a decision
        self.intersection = self.get_intersections()

        # initialize the starting state
        self.current_state = self.reset()
        self.steps_since_intersection = 0 # is this necessary?
        
    
    def step(self, action):
        pass
    
    def reset(self):
        ghost_pos = self.ghost_position_start
        pacman_pos = self.pacman_position_start
        self.score = 0
        return self.get_state_representation(ghost_pos, pacman_pos)

    def get_state_representation(self, ghost_position, pacman_position):
        pass

    def get_intersections(self):
        intersections = []
        rows, cols = self.maze_layout.shape
        for row in range(rows):
            for col in range(cols):
                cell = self.maze_layout[row][col]

                # a wall cannot be an intersection
                if cell == 1:
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
    
    def pacman_movement(self, pacman_position):
        pass
