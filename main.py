import copy
import csv
import math
from random import randrange
import random
from ale_py import ALEInterface, SDL_SUPPORT, Action
import numpy as np

###########################################################

'''State'''
'''
Information obtained by testing and comparing on ale.getscreen(), we
want information from ale.getscreen()(in pixel) to be translated into
state that we want
'''
# 0: road without food
# 1: road with food
# 2: power pellet
# 9: wall
game_board = [
    [9, 1, 1, 1, 1, 9, 1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 1, 9],
    [9, 2, 9, 9, 1, 9, 1, 9, 9, 9, 9, 9, 9, 1, 9, 1, 9, 9, 2, 9],
    [9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9],
    [9, 9, 1, 9, 1, 9, 9, 1, 9, 9, 9, 9, 1, 9, 9, 1, 9, 1, 9, 9],
    [0, 1, 1, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 0],
    [9, 9, 1, 9, 9, 9, 9, 1, 9, 9, 9, 9, 1, 9, 9, 9, 9, 1, 9, 9],
    [0, 9, 1, 1, 1, 1, 1, 1, 9, 0, 0, 9, 1, 1, 1, 1, 1, 1, 9, 0],
    [9, 9, 1, 9, 9, 9, 9, 1, 9, 9, 9, 9, 1, 9, 9, 9, 9, 1, 9, 9],
    [0, 1, 1, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 0],
    [9, 9, 1, 9, 1, 9, 1, 9, 1, 9, 9, 1, 9, 1, 9, 1, 9, 1, 9, 9],
    [9, 1, 1, 1, 1, 9, 1, 9, 1, 1, 1, 1, 9, 1, 9, 1, 1, 1, 1, 9],
    [9, 1, 9, 9, 1, 9, 1, 1, 1, 9, 9, 1, 1, 1, 9, 1, 9, 9, 1, 9],
    [9, 2, 9, 9, 1, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 1, 9, 9, 2, 9],
    [9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9]
]
n_rows = 14 # 14 rows of blocks
n_cols = 20 # 20 col of blocks
block_size = (int(168/n_rows), int(160/n_cols)) # a block in displayscreen , row x col, 12 by 8 remove 2 pixels on the top(a black line and a pink line)
pacman_position = [0,0] # initially somewhere between [8,9] and [8,10]
# The last bit is the scared bit, 1 means the ghost is scared
yellow_ghost = [0,0,0] # 38
red_ghost = [0,0,0] # 70
pink_ghost = [0,0,0] # 88
blue_ghost = [0,0,0] # 184
pacman_position_list = []
'''
74: pink
144: blue
38: yellow
70: red
88: pink
184: blue
42: pacman yellow
'''
def read_state(screen):
    '''
    read screen in pixel and return the state information that i want
    '''
    global game_board
    global n_rows
    global n_cols
    global block_size
    global pacman_position
    global yellow_ghost
    global red_ghost
    global pink_ghost
    global blue_ghost
    global pacman_position_list

    pacman_position_list=[]
    state = copy.deepcopy(game_board)
    for row in range(n_rows): # 14 rows of blocks
        for col in range(n_cols): # 20 col of blocks
            p_occ = {} # record occurance of pixels
            # in each block, collect information of diff color pixel
            for i in range(block_size[0]):
                for j in range(block_size[1]):
                    pixel = screen[row*block_size[0]+i+2][col*block_size[1]+j] # ignore the pixels on the top
                    occurrence = 0 if p_occ.get(pixel) is None else p_occ.get(pixel)
                    p_occ[pixel] = occurrence + 1 # increase occurrence by 1
            block_code = 0 #by default it is a block of road without food
            # update food distribution on the board
            if (74 in p_occ and p_occ[74]>=sum(p_occ.values())//2):
                block_code = 9 # deal with half-half case on the left and right sides of the game board and wall case in general
            elif 74 in p_occ and p_occ[74]==8:
                block_code = 1 # road with food
            elif 74 in p_occ and p_occ[74]==28:
                block_code = 2 # power pellet

            # update the position of pacman and ghost    
            elif 42 in p_occ:
                pacman_position_list.append([row,col])
                # if col <= n_cols // 2:
            elif 38 in p_occ:
                yellow_ghost[2] = 0
                if row != yellow_ghost[0] or col != yellow_ghost[1]: # if the position of yellow ghost has changed
                    yellow_ghost[:2] = [row,col]
            elif 70 in p_occ:
                red_ghost[2] = 0
                if row != red_ghost[0] or col != red_ghost[1]: # if the position of red ghost has changed
                    red_ghost[:2] = [row,col]
            elif 88 in p_occ:
                pink_ghost[2] = 0
                if row != pink_ghost[0] or col != pink_ghost[1]: # if the position of pink ghost has changed
                    pink_ghost[:2] = [row,col]
            elif 184 in p_occ:
                blue_ghost[2] = 0
                if row != blue_ghost[0] or col != blue_ghost[1]: # if the position of blue ghost has changed
                    blue_ghost[:2] = [row,col]
            elif 150 in p_occ:
                # scared ghost
                yellow_ghost[2] = 1
                red_ghost[2] = 1
                pink_ghost[2] = 1
                blue_ghost[2] = 1
                if(abs(row-pink_ghost[0])+abs(col-pink_ghost[1]) < 2):
                    pink_ghost[:2] = [row,col]
                elif(abs(row-yellow_ghost[0])+abs(col-yellow_ghost[1]) < 2):
                    yellow_ghost[:2] = [row,col]
                elif(abs(row-red_ghost[0])+abs(col-red_ghost[1]) < 2):
                    red_ghost[:2] = [row,col]
                elif(abs(row-blue_ghost[0])+abs(col-blue_ghost[1]) < 2):
                    blue_ghost[:2] = [row,col]             
                continue
            state[row][col] = block_code
    big_col = 0
    small_col = 100
    for positions in pacman_position_list:
        if positions[1] <= (n_cols-1) // 2: # if col is on the right, squeeze it toward the center
            big_col = positions[1] if positions[1] > big_col else big_col
            r = positions[0] if positions[0] != pacman_position[0] else pacman_position[0]
            pacman_position = [r, big_col]
        elif positions[1] > (n_cols-1) // 2: # if col is on the left, squeeze it toward the center
            small_col = positions[1] if positions[1] < small_col else small_col
            r = positions[0] if positions[0] != pacman_position[0] else pacman_position[0]
            pacman_position = [r,small_col]
    
    state[pacman_position[0]][pacman_position[1]] = 3
    state[yellow_ghost[0]][yellow_ghost[1]] = 4
    state[red_ghost[0]][red_ghost[1]] = 5
    state[pink_ghost[0]][pink_ghost[1]] = 6
    state[blue_ghost[0]][blue_ghost[1]] = 7
    return state

''' Helper function for reading, storing and testing '''
# read trained weights
def read_weights(file = './weights.csv'):
    global impacts
    impacts = []
    with open(file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            for col in row:
                impacts.append(float(col))
    if not impacts:
        impacts = [0,0,0,0]

# store trained weights
def store_weights(file = './weights.csv'):
    global impacts
    with open(file, 'w', newline = '') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(impacts)
###########################################################

'''Q-Learning'''
alpha = None
gamma = None
epsilon = None

# Get the list of legal actions
# 2: up; 3:right; 4:left; 5:down
legal_actions = [2,3,4,5] #up, right, left, down

prev_state = None
prev_action = None
def Q_learning(state, reward):
    global alpha
    global gamma
    global epsilon
    global prev_state
    global prev_action
    global pacman_position
    global impacts

    normalize_weights()

    if prev_state is not None:
        Q = valueQ(prev_state, prev_action)
        _,Qmax = maxQ(state)
        food_dist, num_food = findFood(state, pacman_position)
        scared_dist, active_dist = findGhost(pacman_position)
        # modify alpha accordingly
        try:
            impacts[0] = impacts[0] + alpha / food_dist * (reward + gamma*Qmax - Q)
        except ZeroDivisionError:
            impacts[0] = impacts[0] + alpha*(reward + gamma*Qmax - Q)
        try:
            impacts[1] = impacts[1] + alpha / scared_dist * (reward + gamma*Qmax - Q)
        except ZeroDivisionError:
            impacts[1] = impacts[1] + alpha*(reward + gamma*Qmax - Q)
        impacts[2] = impacts[2] + alpha * active_dist * (reward + gamma*Qmax - Q)

        try:
            impacts[3] = impacts[3] + alpha / num_food * (reward + gamma*Qmax - Q)
        except ZeroDivisionError:
            impacts[3] = impacts[3] + alpha * (reward + gamma*Qmax - Q)

    prev_state = state
    prev_action = explore(state)
    return prev_action


def valueQ(state, action):
    global pacman_position
    tmp_position = pacman_position # in roder to reset the position of the pacman
    Q = 0
    next_state = copy.deepcopy(state)
    next_state[tmp_position[0]][tmp_position[1]] = 0

    try:
        if (action == 2 and next_state[tmp_position[0]-1][tmp_position[1]] != 9): # move up and not against a wall
            next_state[tmp_position[0]-1][tmp_position[1]] = 3
            tmp_position = [tmp_position[0]-1, tmp_position[1]]
        elif (action == 3 and next_state[tmp_position[0]][tmp_position[1]+1] != 9): # move right and not against a wall
            next_state[tmp_position[0]][tmp_position[1]+1] = 3
            tmp_position = [tmp_position[0], tmp_position[1]+1]
        elif (action == 4 and next_state[tmp_position[0]][tmp_position[1]-1] != 9): # move left and not against a wall
            next_state[tmp_position[0]][tmp_position[1]-1] = 3
            tmp_position = [tmp_position[0], tmp_position[1]-1]
        elif (action == 5 and next_state[tmp_position[0]+1][tmp_position[1]] != 9): # move down and not against a wall
            next_state[tmp_position[0]+1][tmp_position[1]] = 3
            tmp_position = [tmp_position[0]+1, tmp_position[1]]
        else:
            next_state[tmp_position[0]][tmp_position[1]] = 3

    except IndexError:
        # left and right tunnel
        if (tmp_position[1] == 0 and action == 4): # move left using tunnel
            next_state[tmp_position[0]][len(next_state[0])-1] = 3
            tmp_position = [tmp_position[0], len(next_state[0])-1]
        elif (tmp_position[1] == len(next_state[0])-1 and action ==3): # move right using tunnel
            next_state[tmp_position[0]][0] = 3
            tmp_position = [tmp_position[0], 0]
        # there is no tunnel on top and on bottom
        else:
            next_state[tmp_position[0]][tmp_position[1]] = 3
    
    dist_food, num_food = findFood(next_state, tmp_position)
    scared_dist, active_dist = findGhost(tmp_position)

    try:
        # larger the dist, smaller the Q
        Q += impacts[0] / dist_food
    except ZeroDivisionError:
        Q += impacts[0]
    try:
        # larger the sacred dist, smaller the Q
        Q += impacts[1] / scared_dist
    except ZeroDivisionError:
        Q += impacts[1]
    if active_dist:
        # larger the active dist, larger the Q 
        Q += impacts[2] * active_dist
    else: 
        Q -= 100
    try:
        # smaller the fodd, bigger the Q
        Q += impacts[3] / num_food
    except ZeroDivisionError:
        Q += impacts[3]
    return Q


def maxQ(state):
    '''
    given the current state find the best action
    '''
    global legal_actions
    global pacman_position
    Qmax = float('-inf')
    action = 0
    for act in legal_actions:
        Q = valueQ(state, act)
        if (Q > Qmax):
            try:
                # if the next move is toward a wall, choose another one, to avoid stuck
                if(
                    (act == 2 and state[pacman_position[0]-1][pacman_position[1]] == 9) or  # up
                    (act == 3 and state[pacman_position[0]][pacman_position[1]+1] == 9) or  # right
                    (act == 4 and state[pacman_position[0]][pacman_position[1]-1] == 9) or  # left
                    (act == 5 and state[pacman_position[0]+1][pacman_position[1]] == 9)     # down
                ):
                    continue
                    
                else:
                    action = act
                    Qmax = Q
            except IndexError:
                continue
    return action, Qmax

def findFood(state, myposition):
    dist = 0
    num_food=0
    for i in range(0, len(state)):
        for j in range(0, len(state[0])):
            if state[i][j] == 1:
                num_food += 1
                dx = abs(myposition[0]-i)
                dy = abs(myposition[1]-j)
                # using tunnel for the distance
                if dy > len(state) // 2:
                    dy = abs(myposition[1]) + abs(len(state)-1-j) 
                dist += dx
                dist += dy
    return dist, num_food

def findGhost(myposition):
    global yellow_ghost
    global red_ghost
    global pink_ghost
    global blue_ghost

    scared_dist = float('inf')
    active_dist = float('inf')

    ghost_list = []
    ghost_list.append(yellow_ghost)
    ghost_list.append(red_ghost)
    ghost_list.append(pink_ghost)
    ghost_list.append(blue_ghost)

    for ghost in ghost_list:
        dx = abs(myposition[0]-ghost[0])
        dy = abs(myposition[1]-ghost[1])
        delta = dx+dy
        if ghost[2]:

            # find the closest scared dist
            scared_dist = min(delta, scared_dist)
            active_dist = 0
        else:
            # find the closest active dist
            active_dist = min(delta, active_dist)
    return scared_dist, active_dist


def explore(state):
    """
    Epsilon-Greedy Policy
    """
    global epsilon

    actions = [2,3,4,5] #up, right, left, down
    # Randomly Select move
    if(np.random.uniform(0,1) < epsilon):
        return actions[random.randint(0,len(actions)-1)]
    # Select best move
    else:
        pi, _ = maxQ(state)
        return pi

def normalize_weights():
    global impacts
    # Calculate the sum of absolute values of weights
    weight_sum = sum(map(abs, impacts))

    # Normalize each weight by dividing it by the sum
    impacts = [w / weight_sum if weight_sum != 0 else w for w in impacts]

''' Set up'''
ale = ALEInterface()

# Get & Set the desired settings
# ale.setInt("random_seed", 0)
ale.setInt("frame_skip", 5)

# Check if we can display the screen
if SDL_SUPPORT:
    ale.setBool("sound", False)
    ale.setBool("display_screen", False)

ale.loadROM("./MSPACMAN.BIN")

def train(train_episode = 5000):
    # Q-learning parameters
    global alpha # Learning rate
    global gamma  # Discount factor
    global epsilon  # Exploration-exploitation trade-off 
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.1  # Exploration-exploitation trade-off 
    # Play 100 episodes for training
    for episode in range(train_episode):
        read_weights()
        print(impacts)
        reward = 0
        total_reward = 0
        while not ale.game_over():
            state = read_state(ale.getScreen())
            a = Q_learning(state, reward)
            a = 0 if a is None else a
            reward = ale.act(a)
            total_reward += reward
        store_weights()
        print("Episode %d ended with score: %d" % (episode, total_reward))

        ale.reset_game()

def test():
    global alpha    # Learning rate
    global gamma    # Discount factor
    global epsilon  # Exploration-exploitation trade-off 
        # Q-learning parameters
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0  # Exploration-exploitation trade-off 

    read_weights()
    print(impacts)
    reward = 0
    total_reward = 0
    while not ale.game_over():
        state = read_state(ale.getScreen())
        a = Q_learning(state, reward)
        a = 0 if a is None else a
        reward = ale.act(a)
        total_reward += reward
    print("Score: %d" % (total_reward))
    ale.reset_game()
    return total_reward


if __name__ == "__main__":
    # train(5000)



    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    for i in range(100):
        r = test()
        if r <= 500:
            a += 1
        elif r <= 1000:
            b += 1
        elif r <= 2000:
            c += 1
        elif r <= 3000:
            d += 1
        else:
            e += 1
    print('<=500: ', a)
    print('500-1000: ', b)
    print('1000-2000: ', c)
    print('2000-3000: ', d)
    print('3000+: ', e)

