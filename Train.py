import torch
import torch.optim as optim
import torch.nn.functional as F
import pygame

from Player import *
from Game import *

from Constant import *


def train(trainee: NNPlayer, opponent: Player, episodes  = 500, gamma = 0.95, entropy_coef = 0.1):
    # ACTOR CRITIC?
    # Initialize optimizer
    
    policy_optimizer = optim.Adam(trainee.policy.parameters())
    critic_optimizer = optim.Adam(trainee.critic.parameters())
    
    side = 0
    # Episodes
    for episode in range(episodes):
        step = 0
        done = False
        game = RTSGame()
        
        rewards = []
        log_probs = []
        state_values = []
        # Mask out empty spaces
        masks = []
        
        policy_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        
        last_reward = 0


        while not done and step <= 200:

            if side == 0: 
                state_tensor = game.get_state_tensor()
                
                player_map = state_tensor[0, side, :, :]

                # Create a mask for where the player has units, then only consider those tiles when calculating loss to avoid being distracted by empty tiles.
                mask = player_map > 0
                masks.append(mask)

                state_values.append(trainee.critic(state_tensor))
                action = trainee.getAction(game)
                
                log_prob = trainee.getProbabilities(action)
                log_probs.append(log_prob)
                
                action, state_tensor, win, reward = game.step(action, side)
                
                rewards.append(reward - last_reward)
                last_reward = reward
                
            else:
                action, state_tensor, win, reward = game.step(opponent.getAction(game), side)

            if win != -1:
                done = True

            side = (side + 1) % 2
            step += 1 
        
        # Reward, change of game score
        rewards = np.array(rewards)
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0,R)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        # Make sure dont divide by 0
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # Loss functions
        log_probs = torch.stack(log_probs) 
        state_values = torch.cat(state_values).squeeze()
        masks = torch.stack(masks)

        advantage = returns - state_values.detach()
        
        entropy = trainee.m.entropy() * masks.float()  
        entropy = entropy.sum() / (masks.sum() + 1e-7)
        
        # Advantage above is [step] shaped while log_prob is probility for [step, MAP_W, MAP_H], need to expand out 
        advantage = advantage.view(-1, 1, 1).expand_as(log_probs)
        policy_loss = -(log_probs * advantage) * masks.float()
        policy_loss = policy_loss.sum() / (masks.sum() + 1e-7)

        policy_loss = policy_loss - entropy * entropy_coef

        critic_loss = F.huber_loss(state_values, returns)
        policy_loss.backward()
        critic_loss.backward()
        # Optimze
        policy_optimizer.step()
        critic_optimizer.step()

        print(f"Ep {episode}: Policy_loss {policy_loss}: Critic_loss {critic_loss}")
        
    
    return None

def pit(p1: Player, p2: Player, num_games):
    pygame.init()
    clock = pygame.time.Clock()
    
    side = 0
    win_rate = [0,0]
    # Episodes
    for game_num in range(num_games):
        step = 0
        done = False
        game = RTSGame()
        printed_side = 0
        slow = False
        skip = False
        while not done and step <= 200:
            
            if  game_num == 0:
                if not printed_side:
                    print("sides ", game.left_side, game.right_side)
                    printed_side = True
                screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
                game.setScreen(screen)
                game.display()
                if skip:
                    clock.tick(0)
                elif not slow:
                    clock.tick(15)
                else:
                    clock.tick(2)
                
                if pygame.mouse.get_pressed()[0]:
                    if 0 <= pygame.mouse.get_pos()[0] <= WINDOW_W and 0 <= pygame.mouse.get_pos()[1] <= WINDOW_H:
                        grid_x = pygame.mouse.get_pos()[0] // BLOCKSIZE
                        grid_y = pygame.mouse.get_pos()[1] // BLOCKSIZE
                        tile_info = bitunpackTile(game.map[grid_x][grid_y])
                        
                        print(f"mouse click at grid {(grid_x, grid_y)}")
                        print(f"tile info: hp:{tile_info.hp}, gold:{tile_info.carry_gold}")
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE and not slow:
                            slow = True
                        elif event.key == pygame.K_SPACE and slow:
                            slow = False
                        elif event.key == pygame.K_s:
                            skip = True

            if side == 0: 
                action, state_tensor, win, reward = game.step(p1.getAction(game), side)
            else:
                action, state_tensor, win, reward = game.step(p2.getAction(game), side)

            
            side = (side + 1) % 2
            step += 1 

            if win != -1:
                done = True
        if done:
            win_rate[win] += 1
        else:
            if game.get_score(0) > game.get_score(1):
                win_rate[0] += 1
            elif game.get_score(0) < game.get_score(1):
                win_rate[1] += 1
    return win_rate

def show(p1: Player, p2: Player, num_games):
    pygame.init()
    clock = pygame.time.Clock()
    
    side = 0

    # Episodes
    for game_num in range(num_games):
        step = 0
        done = False
        game = RTSGame()
        printed_side = 0
        slow = False
        skip = False
        while not done and step <= 150:
            
            if not printed_side:
                print("sides ", game.left_side, game.right_side)
                printed_side = True
            screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
            game.setScreen(screen)
            game.display()
            if skip:
                clock.tick(0)
            elif not slow:
                clock.tick(15)
            else:
                clock.tick(2)
            
            if pygame.mouse.get_pressed()[0]:
                if 0 <= pygame.mouse.get_pos()[0] <= WINDOW_W and 0 <= pygame.mouse.get_pos()[1] <= WINDOW_H:
                    grid_x = pygame.mouse.get_pos()[0] // BLOCKSIZE
                    grid_y = pygame.mouse.get_pos()[1] // BLOCKSIZE
                    tile_info = bitunpackTile(game.map[grid_x][grid_y])
                    
                    print(f"mouse click at grid {(grid_x, grid_y)}")
                    print(f"tile info: hp:{tile_info.hp}, gold:{tile_info.carry_gold}")
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and not slow:
                        slow = True
                    elif event.key == pygame.K_SPACE and slow:
                        slow = False
                    elif event.key == pygame.K_s:
                        skip = True

            if side == 0: 
                action, state_tensor, win, reward = game.step(p1.getAction(game), side)
            else:
                action, state_tensor, win, reward = game.step(p2.getAction(game), side)

            
            side = (side + 1) % 2
            step += 1 

            if win != -1:
                done = True
