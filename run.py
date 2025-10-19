import pygame
from pygame.locals import *
from constants import *
import constants
from pacman import Pacman
from nodes import NodeGroup
from pellets import PelletGroup
from ghosts import GhostGroup
from fruit import Fruit
from pauser import Pause
from text import TextGroup, Text
from sprites import LifeSprites, MazeSprites, PacmanSprites
from mazedata import MazeData
from button import Button
from entity import DummyEntity

class GameController(object):
    def __init__(self):
        self.screen = pygame.display.set_mode(SCREEN_SIZE, 0, 32)
        self.background = None
        self.background_norm = None
        self.background_flash = None
        self.clock = pygame.time.Clock()
        self.fruit = None
        self.pause = Pause(True)
        self.level = 0
        self.lives = 5
        self.score = 0
        self.textgroup = TextGroup()
        self.lifesprites = LifeSprites(self.lives)
        self.flashBG = False
        self.flashTime = 0.2
        self.flashTimer = 0
        self.fruitCaptured = []
        self.mazeData = MazeData()
        self.buttons = []
        print("Num Channels: ", pygame.mixer.get_num_channels())

    # game set-up #
    def setBackground(self):
        self.background_norm = pygame.surface.Surface(SCREEN_SIZE).convert()
        self.background_norm.fill(BLACK)
        self.background_flash = pygame.surface.Surface(SCREEN_SIZE).convert()
        self.background_flash.fill(BLACK)
        self.background_norm = self.mazeSprites.constructBackground(self.background_norm, self.level % 5)
        self.background_flash = self.mazeSprites.constructBackground(self.background_flash, 5)
        self.flashBG = False
        self.background = self.background_norm

    def loadTitleScreen(self):
        self.titleScreen = True
        self.setTitleScreenBackground()
        self.buttons.append(Button(SCREEN_WIDTH // 2 - 60, SCREEN_HEIGHT // 2 - 30, 120, 60, WHITE, "PRESS TO START", self.startGame))
        constants.MOUSE_CLICK = 0
        self.titleText = Text("PACMAN", YELLOW, 0, SCREEN_HEIGHT // 4, 64)
        self.titleText.position.x = (SCREEN_WIDTH - self.titleText.width) // 2

        # pacman title animation
        self.titlePacman = DummyEntity()
        self.titlePacmanSprites = PacmanSprites(self.titlePacman, animSpeed = 10)
        self.titlePacman.direction = RIGHT

        self.button_sound = pygame.mixer.Sound("sounds/button_click.ogg")
        pygame.mixer.music.load("sounds/music.ogg")
        pygame.mixer.music.play(loops = -1) # -1 --> loops forever
        pygame.mixer.music.set_volume(0.3)

    def setTitleScreenBackground(self):
        self.background = pygame.surface.Surface(SCREEN_SIZE).convert()
        self.background.fill(BLACK)

    def startGame(self):
        # music and sounds
        pygame.mixer.music.stop()
        pygame.mixer.music.set_volume(0.1)
        self.small_pellet_sound = pygame.mixer.Sound("sounds/small_pellet.wav")
        self.eat_ghost_sound = pygame.mixer.Sound("sounds/ghost_eat.wav")
        self.eat_fruit_sound = pygame.mixer.Sound("sounds/fruit_eat.wav")
        self.pacman_death_sound = pygame.mixer.Sound("sounds/death.wav")
        self.fright_sound = pygame.mixer.Sound("sounds/fright_mode_short.wav")
        self.channel0 = pygame.mixer.Channel(0)

        self.titleScreen = False
        self.mazeData.loadMaze(self.level)
        self.mazeSprites = MazeSprites("mazes/structure/" + self.mazeData.obj.name + ".txt", "mazes/rotation/" + self.mazeData.obj.name + "_rotation.txt")
        self.setBackground()

        self.nodes = NodeGroup("mazes/structure/" + self.mazeData.obj.name + ".txt")
        self.mazeData.obj.setPortalPairs(self.nodes)
        self.mazeData.obj.connectHomeNodes(self.nodes)

        self.pacman = Pacman(self.nodes.getNodeFromTiles(*self.mazeData.obj.pacmanStart))
        self.pellets = PelletGroup("mazes/structure/" + self.mazeData.obj.name + ".txt")

        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)
        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazeData.obj.addOffset(2,3)))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(*self.mazeData.obj.addOffset(0,3)))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(*self.mazeData.obj.addOffset(4,3)))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(*self.mazeData.obj.addOffset(2,3)))
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazeData.obj.addOffset(2,0)))
        
        # limiting entity movement
        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.mazeData.obj.denyGhostsAccess(self.ghosts, self.nodes)

    def restartGame(self):
        self.lives = 5
        self.level = 0
        self.pause.paused = True
        self.fruit = None
        self.startGame()
        self.score = 0
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        self.textgroup.showText(READY_TXT)
        self.lifesprites.resetLives(self.lives)
        self.fruitCaptured = []
        self.loadTitleScreen()

    def resetLevel(self):
        self.pause.paused = True
        self.pacman.reset()
        self.ghosts.reset()
        self.fruit = None
        self.textgroup.showText(READY_TXT)

    # updating #
    def update(self):
        dt = self.clock.tick(30) / 1000.0
        if not self.titleScreen:
            self.textgroup.update(dt)
            self.pellets.update(dt)
            if not self.pause.paused:
                self.ghosts.update(dt)
                if self.fruit is not None:
                    self.fruit.update(dt)
                self.checkPelletEvents()
                self.checkGhostEvents()
                self.checkFruitEvents()
            
            if self.pacman.alive:
                if not self.pause.paused:
                    self.pacman.update(dt)
            else:
                self.pacman.update(dt)

            if self.flashBG:
                self.flashTimer += dt
                if self.flashTimer >= self.flashTime:
                    self.flashTimer = 0
                    if self.background == self.background_norm:
                        self.background = self.background_flash
                    else:
                        self.background = self.background_norm

            afterPauseMethod = self.pause.update(dt)
            if afterPauseMethod is not None:
                afterPauseMethod()
        else:
            for button in self.buttons:
                button.update()
            self.titlePacmanSprites.update(dt)
        
        self.checkEvents()
        self.render()

    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)

    def render(self):
        self.screen.blit(self.background, (0,0))
        if not self.titleScreen:
            self.pellets.render(self.screen)
            if self.fruit is not None:
                self.fruit.render(self.screen)
            self.pacman.render(self.screen)
            self.ghosts.render(self.screen)
            self.textgroup.render(self.screen)

            for i in range(len(self.lifesprites.images)):
                x = self.lifesprites.images[i].get_width() * i
                y = SCREEN_HEIGHT - self.lifesprites.images[i].get_height()
                self.screen.blit(self.lifesprites.images[i], (x,y))
            
            for i in range(len(self.fruitCaptured)):
                x = SCREEN_WIDTH - self.fruitCaptured[i].get_width() * (i + 1)
                y = SCREEN_HEIGHT - self.fruitCaptured[i].get_height()
                self.screen.blit(self.fruitCaptured[i], (x, y))
        else:
            self.titleText.render(self.screen)
            for button in self.buttons:
                button.render(self.screen)

            # pacman animation
            scaledPacman = pygame.transform.scale(
                self.titlePacman.image,
                (self.titlePacman.image.get_width() * 3, self.titlePacman.image.get_height() * 3)
            )
            scaledPacman.set_colorkey((255, 0, 255))
            scaledWidth = scaledPacman.get_width()

            pacmanX = SCREEN_WIDTH // 2 - scaledWidth // 2
            pacmanY = SCREEN_HEIGHT // 2 + 40
            self.screen.blit(scaledPacman, (pacmanX, pacmanY))

        pygame.display.update()
    
    def nextLevel(self):
        self.showEntities()
        self.level += 1
        self.pause.paused = True
        self.startGame()
        self.textgroup.updateLevel(self.level)

    # events #
    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN:
                if not self.titleScreen:
                    if event.key == K_SPACE:
                        if self.pacman.alive:
                            self.pause.setPause(playerPaused = True)
                            if not self.pause.paused:
                                self.textgroup.hideText()
                                self.showEntities()
                                self.unpausePausedSounds()
                            else:
                                self.textgroup.showText(PAUSE_TXT)
                                self.hideEntities()
                                self.pauseSounds()
                else:
                    if event.key == K_RETURN:
                        self.startGame()
            if self.titleScreen:
                if event.type == MOUSEBUTTONDOWN:
                    constants.MOUSE_CLICK = event.button
                elif event.type == MOUSEBUTTONUP:
                    constants.MOUSE_CLICK = 0

    def checkPelletEvents(self):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            self.playSmallPelletSound()
            self.pellets.numEaten += 1
            self.updateScore(pellet.points)
            if self.pellets.numEaten == 30:
                self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky)
            if self.pellets.numEaten == 70:
                self.ghosts.clyde.startNode.allowAccess(LEFT, self.ghosts.clyde)
            self.pellets.pelletList.remove(pellet)
            if pellet.name == POWERPELLET:
                self.ghosts.startFreight()
                self.playFrightSound()
            if self.pellets.isEmpty():
                self.flashBG = True
                self.hideEntities()
                self.pause.setPause(pauseTime = 3, func = self.nextLevel)

    def checkGhostEvents(self):
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    self.playEatGhostSound()
                    self.pacman.visible = False
                    ghost.visible = False
                    self.updateScore(ghost.points)
                    self.textgroup.addText(str(ghost.points), WHITE, ghost.position.x, ghost.position.y, 8, time = 1)
                    self.ghosts.updatePoints()
                    self.pause.setPause(pauseTime = 1, func = self.showEntities)
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                elif ghost.mode.current is not SPAWN:
                    if self.pacman.alive:
                        self.lives -= 1
                        self.lifesprites.removeImage()
                        self.pacman.die()
                        self.ghosts.hide()
                        self.pauseSounds()
                        self.playDeathSound()
                        pygame.mixer.music.stop()

                        if self.lives <= 0:
                            self.textgroup.showText(GAMEOVER_TXT)
                            self.pause.setPause(pauseTime = 3, func = self.restartGame)
                        else:
                            self.pause.setPause(pauseTime = 3, func = self.resetLevel)

    def checkFruitEvents(self):
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                self.fruit = Fruit(self.nodes.getNodeFromTiles(9,20), self.level)
        if self.fruit is not None:
            if self.pacman.collideCheck(self.fruit):
                self.playEatFruitSound()
                self.updateScore(self.fruit.points)
                self.textgroup.addText(str(self.fruit.points), WHITE, self.fruit.position.x, self.fruit.position.y, 8, time = 1)
                fruitCaptured = False
                for fruit in self.fruitCaptured:
                    if fruit.get_offset() == self.fruit.image.get_offset():
                        fruitCaptured = True
                        break
                if not fruitCaptured:
                    self.fruitCaptured.append(self.fruit.image)
                self.fruit = None
            elif self.fruit.destroy:
                self.fruit = None
    
    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def hideEntities(self):
        self.pacman.visible = False
        self.ghosts.hide()

    # sounds #
    def playSmallPelletSound(self):
        self.channel0.play(self.small_pellet_sound)

    def playEatGhostSound(self):
        pygame.mixer.find_channel(True).play(self.eat_ghost_sound)

    def playEatFruitSound(self):
        pygame.mixer.find_channel(True).play(self.eat_fruit_sound)

    def playDeathSound(self):
        pygame.mixer.find_channel(True).play(self.pacman_death_sound)

    def playFrightSound(self):
        pygame.mixer.find_channel(True).play(self.fright_sound, -1, maxtime = 7000)
    
    def stopFrightSound(self):
        self.fright_sound.stop()

    def pauseSounds(self):
        pygame.mixer.music.pause()
        # pause active sound channels
        self.pausedChannels = []
        for i in range(pygame.mixer.get_num_channels()):
            channel = pygame.mixer.Channel(i)
            if channel.get_busy():
                channel.pause()
                self.pausedChannels.append(i)

    def unpausePausedSounds(self):
        pygame.mixer.music.unpause()
        # resume paused sound channels
        if hasattr(self, 'pausedChannels'):
            for i in self.pausedChannels:
                pygame.mixer.Channel(i).unpause()
            self.pausedChannels = []



if __name__ == "__main__":
    pygame.init()
    game = GameController()
    game.startGame()
    game.loadTitleScreen()
    while True:
        game.update()