#! /usr/bin/env python

import os
import random
import pygame

os.environ["SDL_VIDEO_CENTERED"] = "1"
walls = []
rec_width = 16
width = 20*rec_width
height =rec_width*20
player = None
BLACK = (  0,   0,   0)
previous_rect =None

# Class for the orange dude
class Player(object):

    def __init__(self, positionx,positionY):
        self.rect = pygame.Rect(positionx,positionY,rec_width,rec_width)

    def move(self, dx, dy):

        # Move each axis separately. Note that this checks for collisions both times.
        if dx != 0:
            self.move_single_axis(dx, 0)
        if dy != 0:
            self.move_single_axis(0, dy)

    def move_single_axis(self, dx, dy):

        # Move the rect
        self.rect.x += dx
        self.rect.y += dy

        # If you collide with a wall, move out based on velocity
        for wall in walls:
            if self.rect.colliderect(wall.rect):
                if dx > 0:  # Moving right; Hit the left side of the wall
                    self.rect.right = wall.rect.left
                if dx < 0:  # Moving left; Hit the right side of the wall
                    self.rect.left = wall.rect.right
                if dy > 0:  # Moving down; Hit the top side of the wall
                    self.rect.bottom = wall.rect.top
                if dy < 0:  # Moving up; Hit the bottom side of the wall
                    self.rect.top = wall.rect.bottom


# Nice class to hold a wall rect
class Wall(object):

    def __init__(self, pos):
        walls.append(self)
        self.rect = pygame.Rect(pos[0], pos[1], rec_width, rec_width)


def build_game(player_position, startX, startY):
    # Holds the level layout in a list of strings.
    global player
    global previous_rect
    player = Player(*player_position)
    previous_rect = (player.rect.x, player.rect.y)
    level = [
        "WWWWWWWWWWWWWWWWWWWW",
        "W                  W",
        "W         WWWWWW   W",
        "W   WWWW       W   W",
        "W   W        WWWW  W",
        "W WWW  WWWW        W",
        "W   W     W W      W",
        "W   W     W   WWW WW",
        "W   WWW WWW   W W  W",
        "W     W   W   W W  W",
        "WWW   W   WWWWW W  W",
        "W W      WW        W",
        "W W   WWWW   WWW   W",
        "W     W    E   W   W",
        "WWWWWWWWWWWWWWWWWWWW",
    ]

    # Parse the level string above. W = wall, E = exit
    x = startX
    y = startY
    for row in level:
        for col in row:
            if col == "W":
                Wall((x, y))
            if col == "E":
                end_rect = pygame.Rect(x, y, rec_width, rec_width)
            x += rec_width
        y += rec_width
        x = startX
    return end_rect,player

def draw(screen, end_rect, player):
    for wall in walls:
        pygame.draw.rect(screen, (255, 255, 255), wall.rect)
    pygame.draw.rect(screen, (255, 0, 0), end_rect)
    pygame.draw.rect(screen, (255, 200, 0), player.rect)
    pygame.display.flip()


def update(screen,end_rect):
    if player.rect.colliderect(end_rect):
        return True
    global previous_rect
    current = (player.rect.x, player.rect.y)
    if previous_rect is None or previous_rect != current:
        if previous_rect is not None:
            blackRect = pygame.Rect(*previous_rect,rec_width,rec_width)
            pygame.draw.rect(screen, BLACK, blackRect)
        pygame.draw.rect(screen, (255, 200, 0), player.rect)
        pygame.display.update()
        previous_rect = (player.rect.x, player.rect.y)
    return False


def run_game(player, end_rect, screen):
    # Just added this to make it slightly fun ;)
    if player.rect.colliderect(end_rect):
        print("You win!")
    pygame.draw.rect(screen, (255, 200, 0), player.rect)
    pygame.display.flip()

def move_left():
    player.move(-rec_width, 0)

def move_right():
    player.move(rec_width, 0)

def move_up():
    player.move(0, -rec_width)

def move_bottom():
    player.move(0, rec_width)