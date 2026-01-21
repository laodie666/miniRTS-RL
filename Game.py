import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pygame
import random
import numpy as np

BLACK = (0, 0, 0)
WHITE = (200, 200, 200)

MAP_H = 10
MAP_W = 10
BLOCKSIZE = 20 # for gridsize 
WINDOW_H = MAP_H * BLOCKSIZE
WINDOW_W = MAP_W * BLOCKSIZE


# Tile info (player number, actor type, health_points, is_carrying_gold)
    # Player 2 empty, 0 and 1 are players 3 bits
    
NO_PLAYER = 2    

# Initial hp 4 bits
TC_HP = 10
VILLAGER_HP = 2
TROOP_HP = 3
GOLD_HP = -1

    
MAX_PLAYERS = 4
MAX_ACTORS = 8
MAX_HP = 15
CARRY_CAPACITY = 10

# Actor type 4 bits
EMPTY_TYPE = 4
TC_TYPE = 0
BARRACK_TYPE = 1
TROOP_TYPE = 2
GOLD_TYPE = 3

# carry gold 1 bit

# character representation for each unit type
ASCII_CHARS = {
    TC_TYPE: 'T',       
    BARRACK_TYPE: 'B',  
    TROOP_TYPE: 'S',    
    GOLD_TYPE: 'G',     
    EMPTY_TYPE: '.',    
}

# colors for specific players
PLAYER_COLORS = {
    0: (255, 100, 100), # p0: red
    1: (100, 100, 255), # p1: blue
    NO_PLAYER: (200, 200, 200) # grey
}


class tile:

    def __init__(self, player_n, actor_type, hp, carry_gold):
        self.player_n = player_n      
        self.actor_type = actor_type  
        self.hp = hp                 
        self.carry_gold = carry_gold  


    def onehotEncode(self):

        features = []

        player_vec = [0] * MAX_PLAYERS
        if 0 <= self.player_n < MAX_PLAYERS:
            player_vec[self.player_n] = 1
        features.extend(player_vec)

        actor_vec = [0] * MAX_ACTORS
        if 0 <= self.actor_type < MAX_ACTORS:
            actor_vec[self.actor_type] = 1
        features.extend(actor_vec)

        features.append(self.hp / MAX_HP)
        features.append(self.carry_gold / CARRY_CAPACITY)

        return np.array(features, dtype=np.float32)

def bitpackTile(tile: tile):
    value = 0
    index = 0

    value |= (tile.player_n) << index
    index += MAX_PLAYERS.bit_length()

    value |= (tile.actor_type) << index
    index += MAX_ACTORS.bit_length()

    value |= (tile.hp) << index
    index += MAX_HP.bit_length()

    value |= (tile.carry_gold) << index

    return value

def bitunpackTile(value: int):
    player_n = value & (1 << MAX_PLAYERS.bit_length()) - 1
    value >>= MAX_PLAYERS.bit_length()

    actor_type = value & (1<<MAX_ACTORS.bit_length()) - 1
    value >>= MAX_ACTORS.bit_length()

    hp = value & (1<<MAX_HP.bit_length()) - 1
    value >>= MAX_HP.bit_length()

    carry_gold = value & (1<<CARRY_CAPACITY.bit_length())-1

    return tile(player_n, actor_type, hp, carry_gold)


class RTSGame():

    # Referencing this https://github.com/suragnair/alpha-zero-general/tree/master/rts
    # NOTES: one hot encode every tile
    # Archer, spear, horseman rock paper scissor style.
    # Resource gathering.
    
    # TODO
    # Set up game env
    #   Set up map
    #   Set up different units
    #   Unit interaction
    #   Set up available moves (Will need optimization)
    #   Playable
    
    # TODO
    # Training
    #   Learn and set up DQN

    def setScreen(self, screen):
        self.screen = screen

    def __init__(self):
        empty_val = bitpackTile(tile(NO_PLAYER, EMPTY_TYPE, 0, 0))
        
        self.map = np.full((MAP_W, MAP_H), empty_val)

        self.map[2, 4] = bitpackTile(tile(0, TC_TYPE, TC_HP, 0))
        self.map[7, 4] = bitpackTile(tile(1, TC_TYPE, TC_HP, 0))

        self.map[0, 4] = bitpackTile(tile(NO_PLAYER, GOLD_TYPE, GOLD_HP, 0))
        self.map[9, 4] = bitpackTile(tile(NO_PLAYER, GOLD_TYPE, GOLD_HP, 0))


        if not pygame.font.get_init():
            pygame.font.init()
        self.font = pygame.font.SysFont("monospace", 15, bold=True)


    def drawGrid(self):

        for x in range(MAP_W):
            for y in range(MAP_H):
                
                rect_x = x * BLOCKSIZE
                rect_y = y * BLOCKSIZE
                center_x = rect_x + (BLOCKSIZE // 2)
                center_y = rect_y + (BLOCKSIZE // 2)

                # Draw border
                rect = pygame.Rect(rect_x, rect_y, BLOCKSIZE, BLOCKSIZE)
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)

                packed_val = self.map[x, y]
                tile_data = bitunpackTile(int(packed_val))

                # default to '?' if type not found in dictionary
                char_to_draw = ASCII_CHARS.get(tile_data.actor_type, '?') 

                text_color = PLAYER_COLORS.get(tile_data.player_n, WHITE)

                # render text
                text_surf = self.font.render(char_to_draw, True, text_color)
                text_rect = text_surf.get_rect(center=(center_x, center_y))
                
                self.screen.blit(text_surf, text_rect)

    def display (self):
        pygame.init()
        SCREEN = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        CLOCK = pygame.time.Clock()
        SCREEN.fill(BLACK)
        while True:
            self.drawGrid()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            pygame.display.update()


new = RTSGame()
pygame.init()
SCREEN = pygame.display.set_mode((WINDOW_W, WINDOW_H))

new.setScreen(SCREEN)
new.display()