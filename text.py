import pygame
from vector import Vector2
from constants import *

class Text(object):
    def __init__(self, text, color, x, y, size, time = None, id = None, visible = True):
        self.id = id
        self.text = text
        self.color = color
        self.size = size
        self.visible = visible
        self.position = Vector2(x,y)
        self.timer = 0
        self.lifespan = time
        self.label = None
        self.destroy = False
        self.width = size * len(text)
        self.setupFont("font/PressStart2PRegular.ttf")
        self.createLabel()

    def setupFont(self, fontpath):
        self.font = pygame.font.Font(fontpath, self.size)

    def createLabel(self):
        self.label = self.font.render(self.text, 1, self.color)

    def setText(self, newText):
        self.text = str(newText)
        self.createLabel()

    def setColor(self, newColor):
        if newColor != self.color:
            self.color = newColor
            self.createLabel()

    def update(self, dt):
        if self.lifespan is not None:
            self.timer += dt
            if self.timer >= self.lifespan:
                self.timer = 0
                self.lifespan = None
                self.destroy = True

    def render(self, screen):
        if self.visible:
            x, y = self.position.asTuple()
            screen.blit(self.label, (x,y))

class TextGroup(object):
    def __init__(self):
        self.nextId = 10
        self.allText = {}
        self.setupText()
        self.showText(READY_TXT)

    def addText(self, text, color, x, y, size, time = None, id = None):
        self.nextId += 1
        self.allText[self.nextId] = Text(text, color, x, y, size, time, id)
        return self.nextId

    def removeText(self, id):
        self.allText.pop(id)

    def setupText(self):
        size = TILE_HEIGHT
        self.allText[SCORE_TXT] = Text("0".zfill(8), WHITE, 0, TILE_HEIGHT, size)
        self.allText[LEVEL_TXT] = Text(str(1).zfill(3), WHITE, 23 * TILE_WIDTH, TILE_HEIGHT, size)
        self.allText[READY_TXT] = Text("READY!", YELLOW, 11.25 * TILE_WIDTH, 20 * TILE_HEIGHT, size, visible = False)
        self.allText[PAUSE_TXT] = Text("PAUSED!", YELLOW, 10.625 * TILE_WIDTH, 20 * TILE_HEIGHT, size, visible = False)
        self.allText[GAMEOVER_TXT] = Text("GAME OVER!", YELLOW, 10 * TILE_WIDTH, 20 * TILE_HEIGHT, size, visible = False)

        self.addText("SCORE", WHITE, 0, 0, size)
        self.addText("LEVEL", WHITE, 23 * TILE_WIDTH, 0, size)

    def showText(self, id):
        self.hideText()
        self.allText[id].visible = True
    
    def hideText(self):
        self.allText[READY_TXT].visible = False
        self.allText[PAUSE_TXT].visible = False
        self.allText[GAMEOVER_TXT].visible = False

    def update(self, dt):
        for txt_key in list(self.allText.keys()):
            self.allText[txt_key].update(dt)
            if self.allText[txt_key].destroy:
                self.removeText(txt_key)
    
    def updateScore(self, score):
        self.updateText(SCORE_TXT, str(score).zfill(8))

    def updateLevel(self, level):
        self.updateText(LEVEL_TXT, str(level + 1).zfill(3))

    def updateText(self, id, value):
        if id in self.allText.keys():
            self.allText[id].setText(value)
    
    def render(self, screen):
        for txt_key in list(self.allText.keys()):
            self.allText[txt_key].render(screen)