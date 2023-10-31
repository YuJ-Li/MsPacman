import copy
from random import randrange
from ale_py import ALEInterface, SDL_SUPPORT, Action
import numpy as np

''' Set up'''
ale = ALEInterface()

# Get & Set the desired settings
ale.setInt("random_seed", 123)

# Check if we can display the screen
if SDL_SUPPORT:
    ale.setBool("sound", False)
    ale.setBool("display_screen", True)

ale.loadROM("./MSPACMAN.BIN")

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
pacman_position = [8,9] # initially somewhere between [8,9] and [8,10]
yellow_ghost = [6,9] # 38
red_ghost = [4,9] # 70
pink_ghost = [6,9] # 88
blue_ghost = [6,9] # 184

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

    state = copy.deepcopy(game_board)
    for row in range(n_rows): # 14 rows of blocks
        for col in range(n_cols): # 20 col of blocks
            p_occ = {} # record occurance of pixels
            # in each block, collect information of diff color pixel
            for i in range(block_size[0]):
                for j in range(block_size[1]):
                    pixel = screen[row*block_size[0]+i+2][col*block_size[1]+j] # ???
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
                if row != pacman_position[0] or col != pacman_position[1]: # if the position of pacman has changed
                    pacman_position = [row,col] # update his position
            elif 38 in p_occ:
                if row != yellow_ghost[0] or col != yellow_ghost[1]: # if the position of yellow ghost has changed
                    yellow_ghost = [row,col]
            elif 70 in p_occ:
                if row != red_ghost[0] or col != red_ghost[1]: # if the position of red ghost has changed
                    red_ghost = [row,col]
            elif 88 in p_occ:
                if row != pink_ghost[0] or col != pink_ghost[1]: # if the position of pink ghost has changed
                    pink_ghost = [row,col]
            elif 184 in p_occ:
                if row != blue_ghost[0] or col != blue_ghost[1]: # if the position of blue ghost has changed
                    blue_ghost = [row,col]
            state[row][col] = block_code
    state[pacman_position[0]][pacman_position[1]] = 3
    state[yellow_ghost[0]][yellow_ghost[1]] = 4
    state[red_ghost[0]][red_ghost[1]] = 5
    state[pink_ghost[0]][pink_ghost[1]] = 6
    state[blue_ghost[0]][blue_ghost[1]] = 7
    return state

''' Helper function for testing '''
def print_state_board(board):
    for i in range(len(board)):
        print(board[i])
    print('##################################')



###########################################################

'''Q-Learning'''
legal_actions = [2,3,4,5] #up, right, left, down

def valueQ(state, action):
    return 0


def maxQ(state):
    '''
    given the current state find the best action
    '''
    global legal_actions
    global pacman_position
    
    Qmax = float('-inf')
    action = -1
    for act in legal_actions:
        Q = valueQ(state, act)
        if (Q > Qmax):
            try:
                # if the next move is toward a wall, choose another one, to avoid stuck
                if(
                    (act == 2 and state[pacman_position[0]-1][pacman_position[1]] == 6) or  # up
                    (act == 3 and state[pacman_position[0]-1][pacman_position[1]+1] == 6) or  # right
                    (act == 4 and state[pacman_position[0]][pacman_position[1]-1] == 6) or  # left
                    (act == 5 and state[pacman_position[0]+1][pacman_position[1]] == 6)     # down
                ):
                    continue
                else:
                    action = act
                    Qmax = Q
            except IndexError:
                continue
    return action, Qmax



# Get the list of legal actions
# 2: up; 3:right; 4:left; 5:down


# Q-learning parameters
alpha = 0.01  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.5  # Exploration-exploitation trade-off 


# Initialize Q-table
state_space_size = ale.getScreenDims()
action_space_size = len(legal_actions)
q_table = np.zeros((state_space_size[0], state_space_size[1], action_space_size))

# Play 1000 episodes for training
for episode in range(1000):
    total_reward = 0
    ale.reset_game()
    state = ale.getScreen()
    state = np.asanyarray(state)
    np.set_printoptions(threshold=np.inf)

    while not ale.game_over():
        print_state_board(read_state(ale.getScreen()))
        # Convert state to indices for indexing the Q-table
        state_indices = state[0], state[1]

        # Exploration-exploitation trade-off
        if np.random.uniform(0, 1) < epsilon:
            action = legal_actions[randrange(len(legal_actions))]
        else:
            action = np.argmax(q_table[state_indices])
        action = Action.DOWNRIGHT
        # Apply the chosen action
        reward = ale.act(action)
        # Get the next state
        next_state = ale.getScreen()

        # Convert next state to indices for indexing the Q-table
        next_state_indices = next_state[0], next_state[1]

        # Update Q-value based on the Bellman equation
        q_table[state_indices][action] = q_table[state_indices][action] + alpha * (
                reward + gamma * np.max(q_table[next_state_indices]) - q_table[state_indices][action]
        )

        total_reward += reward
        state = next_state

    print("Episode %d ended with score: %d" % (episode, total_reward))

# testing
for episode in range(10):
    total_reward = 0
    ale.reset_game()
    state = ale.getScreen()

    while not ale.game_over():
        # Convert state to indices for indexing the Q-table
        state_indices = state[0], state[1]
        action = np.argmax(q_table[state_indices])
        reward = ale.act(action)
        total_reward += reward
        state = ale.getScreen()

    print("Test episode %d ended with score: %d" % (episode, total_reward))