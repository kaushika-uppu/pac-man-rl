import pygame
from constants import *
import constants
from text import Text

class Button(object):
    def __init__(self, x, y, w, h, color, text = "Button", func = None, textsize = 16):
        self.position = pygame.Vector2(x, y)
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.function = func
        self.text = Text(text, color, self.x, self.y, size = textsize)
        self.text.position.x = self.x + (w - self.text.width) // 2
        self.text.position.y = self.y + (h - 8) // 2
        self.color = color
        self.clicked = False
        self.hovered = False

    # checks to see if mouse is hovering and it it is clicking
    def update(self):
        mouse_pos = pygame.Vector2(pygame.mouse.get_pos())
        if constants.MOUSE_CLICK == 0:
            self.clickLeave()
        if ((self.x <= mouse_pos.x <= (self.x + self.w)) and 
            (self.y <= mouse_pos.y <= (self.y + self.h))):
            if not self.hovered:
                self.hoverEnter()
            if not self.clicked:
                self.click()
        else:
            if self.hovered:
                self.hoverExit()

    def click(self):
        if constants.MOUSE_CLICK == 1:
            self.color = BUTTON_TEXT_HOVER
            self.clicked = True
            self.function()

    def clickLeave(self):
        self.clicked = False

    def hoverEnter(self):
        if constants.MOUSE_CLICK == 0:
            self.text.setColor(BUTTON_TEXT_HOVER)
            self.hovered = True

    def hoverExit(self):
        self.text.setColor(BUTTON_TEXT_NORM)
        self.hovered = False

    def render(self, screen):
        self.text.render(screen)