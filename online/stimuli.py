from win32api import GetSystemMetrics
import pygame
import sys
import math
import win32api

# --- constants ---

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
RED=(255,0,0)

up = 6.67
down = 8.57
left = 10
right = 7.5
frequencies = [up,down,left,right]
width = GetSystemMetrics(0)
height = GetSystemMetrics(1)
rect_width = 50
half_rect_width = rect_width//2
centerX = width//2
centerY = height//2
topY = 10
leftX = 10
upPosition = (centerX-half_rect_width, topY, rect_width,rect_width)
bottomPosition =(centerX-half_rect_width, height-rect_width, rect_width,rect_width)
leftPosition = (leftX, centerY-half_rect_width, rect_width,rect_width)
rightPosition = (width-rect_width, centerY-half_rect_width, rect_width,rect_width)
positions = [upPosition,bottomPosition,leftPosition,rightPosition]

def draw_stimuli(j,fenetre):
    i=0
    for frequency in frequencies:
        result = math.sin(2 * math.pi * frequency * j / 60)
        sign = lambda x: (1, 0)[x < 0]
        color = 255 * sign(result)
        colors = [color, color, color]
        pygame.draw.rect(fenetre, colors, positions[i])
        pygame.display.update()
        i+=1
# --- main ---
def get_system_refresh_period():
    device = win32api.EnumDisplayDevices()
    settings = win32api.EnumDisplaySettings(device.DeviceName, -1)
    screenRate = getattr(settings, 'DisplayFrequency')
    delay = 1000. / screenRate
    return delay

def train_switch(gazePosition,fenetre):
    i = 0
    for index in range(len(frequencies)):
        if index == gazePosition:
            colors = RED
        else:
            colors = WHITE
        pygame.draw.rect(fenetre, colors, positions[i])
        pygame.display.update()
        i += 1

def main():
    pygame.init()
    fenetre = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
    fenetre.fill(BLACK)

    # time in millisecond from start program
    current_time = pygame.time.get_ticks()

    # how long to show or hide

    delay = get_system_refresh_period()
    # time of next change
    change_time = current_time + delay
    show = True
    j=0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # --- updates ---

        current_time = pygame.time.get_ticks()

        # is time to change ?
        if current_time >= change_time:
            # time of next change
            change_time = current_time + delay
            show = not show
            draw_stimuli(j,fenetre)
            pygame.display.update()
            j+=1
        # --- draws ---
# main()