from win32api import GetSystemMetrics
import pygame
import sys
import math
import win32api
import stimuli as st
import maze as maze
import move as mv
import numpy as np

BLACK = (  0,   0,   0)

width = GetSystemMetrics(0)
height = GetSystemMetrics(1)

def get_start():
    blankWidth = width - 2*st.rect_width
    blankHeight = height - 2*st.rect_width
    halfMazeWidth = maze.width//2
    halfMazeHeight = maze.height //2
    startX = (width-maze.width-st.rect_width)//2
    startY = (height - maze.height - st.rect_width)//2
    return startX, startY



def main():
    pygame.init()
    # fenetre = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
    fenetre = pygame.display.set_mode((width,height), pygame.RESIZABLE)
    fenetre.fill(BLACK)
    x,y = get_start()
    player_position = (x+32,y+32)
    end_rect, player = maze.build_game(player_position, x, y)
    maze.draw(fenetre, end_rect, player)
    # time in millisecond from start program
    current_time = pygame.time.get_ticks()

    # how long to show or hide

    delay = st.get_system_refresh_period()
    move_delay = 7000


    # time of next change
    change_time = current_time + delay
    j = 0
    th= mv.save_data()
    maze.draw(fenetre, end_rect, player)
    win = False


    while not win:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()

        win = maze.update(fenetre,end_rect)
        current_time = pygame.time.get_ticks()
        if current_time >= change_time:
            change_time = current_time + delay
            st.draw_stimuli(j, fenetre)
            j += 1
        if win:
            mv.stop(th)
            win_message(fenetre)

def win_message(screen):
    screen.fill(BLACK)
    myfont = pygame.font.SysFont('Comic Sans MS', 30)
    textsurface = myfont.render('You Win', False, (255, 0, 0))
    screen.blit(textsurface, (st.centerX, st.centerY))
    pygame.display.update()

main()