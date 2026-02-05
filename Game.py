import numpy as np
import torch
import random
import numpy as np
import pygame

from Player import *
from Constant import *



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
    index += MAX_PLAYER_BITS

    value |= (tile.actor_type) << index
    index += MAX_ACTORS_BITS

    value |= (tile.hp) << index
    index += MAX_HP_BITS

    value |= (tile.carry_gold) << index

    return value

def bitunpackTile(value: int):
    player_n = value & (1 << MAX_PLAYER_BITS) - 1
    value >>= MAX_PLAYER_BITS

    actor_type = value & (1<<MAX_ACTORS_BITS) - 1
    value >>= MAX_ACTORS_BITS

    hp = value & (1<<MAX_HP_BITS) - 1
    value >>= MAX_HP_BITS

    carry_gold = value & (1<<CARRY_CAPACITY_BITS)-1

    return tile(player_n, actor_type, hp, carry_gold)

def newBarrackTile(side):
    return tile(side, BARRACK_TYPE, BARRACK_HP, 0)

def newTCTile(side):
    return tile(side, TC_TYPE, TC_HP, 0)

def newVillagerTile(side):
    return tile(side, VILLAGER_TYPE, VILLAGER_HP, 0)

def newTroopTile(side):
    return tile(side, TROOP_TYPE, TROOP_HP, 0)

class RTSGame():

    # Referencing this https://github.com/suragnair/alpha-zero-general/tree/master/rts
    # NOTES: one hot encode every tile
    # Archer, spear, horseman rock paper scissor style.
    # Resource gathering.

    def setScreen(self, screen):
        self.screen = screen

    # TODO: RANDOMIZE RESOURCE AND START POSITION

    def __init__(self):
        empty_val = bitpackTile(tile(NO_PLAYER, EMPTY_TYPE, 0, 0))
        
        self.map = np.full((MAP_W, MAP_H), empty_val)
        self.left_side = 0
        self.right_side = 1
        
        self.kills = np.zeros((2), np.int16)
        
        self.map[LEFT_MAIN_TC_POS] = bitpackTile(tile(self.left_side, TC_TYPE, TC_HP, 3))
        self.map[RIGHT_MAIN_TC_POS] = bitpackTile(tile(self.right_side, TC_TYPE, TC_HP, 3))

        self.map[LEFT_MAIN_TC_POS[0] - 1, LEFT_MAIN_TC_POS[1]] = bitpackTile(newVillagerTile(self.left_side))
        self.map[RIGHT_MAIN_TC_POS[0] + 1, RIGHT_MAIN_TC_POS[1]] = bitpackTile(newVillagerTile(self.right_side))

        self.map[0, 4] = bitpackTile(tile(NO_PLAYER, GOLD_TYPE, GOLD_HP, GOLD_GOLD_COUNT))
        self.map[9, 4] = bitpackTile(tile(NO_PLAYER, GOLD_TYPE, GOLD_HP, GOLD_GOLD_COUNT))
        
        self.map[0, 1] = bitpackTile(tile(NO_PLAYER, GOLD_TYPE, GOLD_HP, GOLD_GOLD_COUNT))
        self.map[9, 1] = bitpackTile(tile(NO_PLAYER, GOLD_TYPE, GOLD_HP, GOLD_GOLD_COUNT))
        
        self.map[0, 7] = bitpackTile(tile(NO_PLAYER, GOLD_TYPE, GOLD_HP, GOLD_GOLD_COUNT))
        self.map[9, 7] = bitpackTile(tile(NO_PLAYER, GOLD_TYPE, GOLD_HP, GOLD_GOLD_COUNT))

        self.update_onehot_encoding()

        if not pygame.font.get_init():
            pygame.font.init()
        self.unit_font = pygame.font.SysFont("monospace", 15, bold=True)
        self.stats_font = pygame.font.SysFont("Arial", 20, bold=False)
                
    def update_onehot_encoding(self):
        self.onehot_encoded_tiles = np.array([(bitunpackTile(self.map[x][y])).onehotEncode() for x in range(MAP_W) for y in range(MAP_H)]).reshape(MAP_W, MAP_H, -1)
    
    
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

                tile_info = bitunpackTile(self.map[x][y])

                # default to '?' if type not found in dictionary
                char_to_draw = ASCII_CHARS.get(tile_info.actor_type, '?') 

                text_color = PLAYER_COLORS.get(tile_info.player_n, WHITE)

                # render text
                unit_text_surf = self.unit_font.render(char_to_draw, True, text_color)
                unit_text_rect = unit_text_surf.get_rect(center=(center_x, center_y))
            
                self.screen.blit(unit_text_surf, unit_text_rect)

    def display (self):
        self.screen.fill(BLACK)
        self.drawGrid()
        pygame.display.update()
        
    def move_unit(self, cur_pos, target_pos):
        self.map[target_pos] = self.map[cur_pos]
        self.map[cur_pos] = bitpackTile(tile(NO_PLAYER, EMPTY_TYPE, 0, 0))


    # Return the features of every tile
    def get_state(self):
        return self.onehot_encoded_tiles

    def get_state_tensor(self):
        state_tensor = torch.from_numpy(self.get_state()).float().unsqueeze(0)

        # Move number channels to the front.
        state_tensor = state_tensor.permute(0, 3, 1, 2) 
        return state_tensor.to(device)

    def get_kills_tensor(self):
        return torch.tensor(self.kills)

    # TODO: Vectorize game loop to make it more efficient.
    # Alternative, Numba
    # Alternative, keep track of set of units
    def step(self, action, side):
        processed = np.zeros((MAP_W, MAP_H), dtype=bool) 
        reward = 0
        for x in range(MAP_W):
            for y in range(MAP_H):
                # Say a unit moves right, this is to cover the case for the unit to move again.
                if processed[x][y]: continue 

                tile_info = bitunpackTile(self.map[x][y])
                if tile_info.player_n == side:
                    tx, ty = x, y
                    
                    if action[x][y] == ACT_DOWN:    
                        ty += 1
                    elif action[x][y] == ACT_UP:
                        ty -= 1
                    elif action[x][y] == ACT_LEFT:
                        tx -= 1
                    elif action[x][y] == ACT_RIGHT:
                        tx += 1
                    
                    
                    if not (0 <= tx < MAP_W and 0 <= ty < MAP_H):
                        continue

                    target_tile_info = bitunpackTile(self.map[tx][ty])
                    # Now actually make a move.
                    # TC makes villager
                    if tile_info.actor_type == TC_TYPE:
                        if target_tile_info.actor_type == EMPTY_TYPE and tile_info.carry_gold >= VILLAGER_COST:
                            self.map[tx, ty] = bitpackTile(newVillagerTile(side))
                            tile_info.carry_gold -= VILLAGER_COST
                            self.map[x, y] = bitpackTile(tile_info)
                            processed[x][y] = 1
                            processed[tx][ty] = 1

                    # Barrack makes troop
                    elif tile_info.actor_type == BARRACK_TYPE:
                        if target_tile_info.actor_type == EMPTY_TYPE and tile_info.carry_gold >= TROOP_COST:
                            self.map[tx, ty] = bitpackTile(newTroopTile(side))
                            tile_info.carry_gold -= TROOP_COST
                            self.map[x, y] = bitpackTile(tile_info)
                            processed[x][y] = 1
                            processed[tx][ty] = 1


                    # Villager collect, return gold, make tc and barrack
                    elif tile_info.actor_type == VILLAGER_TYPE:
                        if action[x][y] == TURN_BARRACK and tile_info.carry_gold >= BARRACK_COST:
                            self.map[x][y] = bitpackTile(newBarrackTile(side))
                        elif action[x][y] == TURN_TC and tile_info.carry_gold >= TC_COST:
                            self.map[x][y] = bitpackTile(newTCTile(side))
                        elif target_tile_info.actor_type == GOLD_TYPE:
                            if tile_info.carry_gold <= 5:
                                tile_info.carry_gold += 1
                                target_tile_info.carry_gold -= 1
                                if target_tile_info.carry_gold == 0:
                                    target_tile_info = tile(NO_PLAYER, EMPTY_TYPE, 0, 0)
                            self.map[x, y] = bitpackTile(tile_info)
                            self.map[tx,ty] = bitpackTile(target_tile_info)
                        elif target_tile_info.actor_type == EMPTY_TYPE:
                            self.move_unit((x, y), (tx, ty))
                            processed[x][y] = 1
                            processed[tx][ty] = 1
                        elif target_tile_info.player_n == side and (target_tile_info.actor_type == TC_TYPE or target_tile_info.actor_type == BARRACK_TYPE):
                            # Let TC and Barrack only carry 6 gold
                            transfered_gold = min(6 - target_tile_info.carry_gold, tile_info.carry_gold)
                            target_tile_info.carry_gold += transfered_gold
                            tile_info.carry_gold -= transfered_gold
                            self.map[x, y] = bitpackTile(tile_info)
                            self.map[tx, ty] = bitpackTile(target_tile_info)
                        
                    elif tile_info.actor_type == TROOP_TYPE:
                        if target_tile_info.actor_type == EMPTY_TYPE:
                            self.move_unit((x, y), (tx, ty))
                            processed[x][y] = 1
                            processed[tx][ty] = 1
                        elif target_tile_info.player_n != side and target_tile_info.player_n != NO_PLAYER:
                            # Opponent unit do dmg
                            target_tile_info.hp -= 1
                            self.kills[side] += 1
                            if target_tile_info.hp <= 0:
                                target_tile_info = tile(NO_PLAYER, EMPTY_TYPE, 0, 0)
                                
                            self.map[tx,ty] = bitpackTile(target_tile_info)

        self.update_onehot_encoding()
        
        score_self = self.get_score(side)
        score_enemy = self.get_score((side + 1) % 2)
        
        if score_self is None:
            print(f"player {side} died")
            return action, self.get_state(), (side + 1) % 2, -100
        
        elif score_enemy is None:
            print(f"player {(side+1)%2} died")
            return action, self.get_state(), side, 100
        else:
            return action, self.get_state(), -1, reward



    def get_score(self, side):
        score = -12
        villager_count = 0
        tc_count = 0

        left_main_tc_tile = bitunpackTile(self.map[LEFT_MAIN_TC_POS])
        right_main_tc_tile = bitunpackTile(self.map[RIGHT_MAIN_TC_POS])

        if left_main_tc_tile.player_n != side and right_main_tc_tile.player_n != side:
            # TODO: This is SUPER SCUFFED. Since any number can be reached with the current score, to distinguish game ending return None
            return None

        for x in range(MAP_W):
            for y in range(MAP_H):
                tile_info = bitunpackTile(self.map[x][y])
                if tile_info.player_n == side:
                    if tile_info.actor_type == TC_TYPE:
                        score += tile_info.hp
                        score += tile_info.carry_gold * 1.5
                        tc_count += 1
                    elif tile_info.actor_type == VILLAGER_TYPE:
                        score += tile_info.hp
                        score += tile_info.carry_gold
                        villager_count += 1
                    else:
                        score += tile_info.hp
                        score += tile_info.carry_gold
        if villager_count == 0:
            return None
        score += self.kills[side] * 5
        return score
                        


