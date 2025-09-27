from constants import *

class MazeBase(object):
    def __init__(self):
        self.portalPairs = {}
        self.homeOffset = (0, 0)
        self.ghostNodeDeny = {UP: (),
                              DOWN: (),
                              LEFT: (),
                              RIGHT: ()
        }

    def setPortalPairs(self, nodes):
        for pair in list(self.portalPairs.values()):
            nodes.setPortalPair(*pair)
    
    def connectHomeNodes(self, nodes):
        key = nodes.createHomeNodes(*self.homeOffset)
        nodes.connectHomeNodes(key, self.homeNodeConnectLeft, LEFT)
        nodes.connectHomeNodes(key, self.homeNodeConnectRight, RIGHT)
    
    def addOffset(self, x, y):
        return x + self.homeOffset[0], y + self.homeOffset[1]
    
    def denyGhostsAccess(self, ghosts, nodes):
        nodes.denyAccessList(*(self.addOffset(2,3) + (LEFT, ghosts)))
        nodes.denyAccessList(*(self.addOffset(2,3) + (RIGHT, ghosts)))

        for direction in list(self.ghostNodeDeny.keys()):
            for values in self.ghostNodeDeny[direction]:
                nodes.denyAccessList(*(values + (direction, ghosts)))

class MazeData(object):
    def __init__(self):
        self.obj = None
        self.mazeDict = {0: Maze1, 1: Maze2}
    
    def loadMaze(self, level):
        self.obj = self.mazeDict[level % len(self.mazeDict)]()

class Maze1(MazeBase):
    def __init__(self):
        MazeBase.__init__(self)
        self.name = "maze1"
        self.portalPairs = {0: ((0,17), (27,17))}
        self.homeOffset = (11.5,14)
        self.homeNodeConnectLeft = (12,14)
        self.homeNodeConnectRight = (15,14)
        self.pacmanStart = (15,26)
        self.fruitStart = (9,20)
        self.ghostNodeDeny = {UP: ((12,14), (15,14), (12,26), (15,26)),
                              LEFT: (self.addOffset(2,3),),
                              RIGHT: (self.addOffset(2,3),)
        }

class Maze2(MazeBase):
    def __init__(self):
        MazeBase.__init__(self)
        self.name = "maze2"
        self.portalPairs = {0: ((0,4), (27,4)),
                            1: ((0,26), (27, 26))}
        self.homeOffset = (11.5,14)
        self.homeNodeConnectLeft = (9,14)
        self.homeNodeConnectRight = (18,14)
        self.pacmanStart = (16,26)
        self.fruitStart = (11,20)
        self.ghostNodeDeny = {UP: ((9,14), (18,14), (11,23), (16,23)),
                              LEFT: (self.addOffset(2,3),),
                              RIGHT: (self.addOffset(2,3),)
        }