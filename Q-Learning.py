import numpy as np
import socket

from Agent_Client_Cognition import interpreting, create_msg

host_name, _, ips = socket.gethostbyname_ex(socket.gethostname())
host_IP = None

for ip in ips:
    if not ip.startswith("172.") and not ip.startswith("::1"):
        host_IP = ip

IPC_port = 15051
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (host_IP, IPC_port)

sock.connect(server_address)
print('Conectado ao Servidor: %s >> porta: %s' % server_address)

def restart():
    msg = '{"request":["restart",0]}'
    sock.sendall(msg.encode("utf-8"))
    answES = sock.recv(256)

def request_forward():
    msg = '{"request":["forward",1]}'
    sock.sendall(msg.encode("utf-8"))
    answES = sock.recv(256)
    answES = answES.decode("utf-8")
    sttMM, strCode, idxInpSensor, CurrentSensBits = interpreting(answES)

    return idxInpSensor

def send_command(msg):
    sock.sendall(msg.encode("utf-8"))
    answES = sock.recv(256)
    return interpret(answES)

def is_initial():
    msg = '{"request":["forward",0]}'
    sock.sendall(msg.encode("utf-8"))
    answES = sock.recv(256)
    answES = answES.decode("utf-8")

    if (answES == '{"sense":["initial"]}'):
        return True
    return False

def interpret(answES):
    global carryRWD

    sttMM, strCode, idxInpSensor, CurrentSensBits = interpreting(answES)

    request_sensors = [19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]

    if idxInpSensor == 17 and carryRWD == 0:
        carryRWD = 1

    if idxInpSensor in request_sensors:
        if (is_initial()):
            return 5, carryRWD
        else:
            return request_forward(), carryRWD
    
    return idxInpSensor, carryRWD

def map_output_neurons(action):
    outy = action

    if action == 4:
        outy = 11
    elif action == 5:
        outy = 12
    elif action == 6:
        outy = 13
    
    return outy

def get_next_state(outy):
    msg = create_msg(outy, 1)
    idx_inp_sensor, carryRWD = send_command(msg)

    next_state = idx_inp_sensor

    if idx_inp_sensor == 0:
        next_state = 0
    elif idx_inp_sensor == 1:
        next_state = 1
    elif idx_inp_sensor == 2:
        next_state = 2
    elif idx_inp_sensor == 3 and carryRWD == 0:
        next_state = 4
    elif idx_inp_sensor == 3 and carryRWD == 1:
        next_state = 3
    elif idx_inp_sensor == 4 and carryRWD == 0:
        next_state = 6
    elif idx_inp_sensor == 4 and carryRWD == 1:
        next_state = 5
    elif idx_inp_sensor == 5 and carryRWD == 0:
        next_state = 8
    elif idx_inp_sensor == 5 and carryRWD == 1:
        next_state = 7
    elif idx_inp_sensor == 6:
        next_state = 9
    elif idx_inp_sensor == 7:
        next_state = 10
    elif idx_inp_sensor == 8 and carryRWD == 0:
        next_state = 12
    elif idx_inp_sensor == 8 and carryRWD == 1:
        next_state = 11
    elif idx_inp_sensor == 9 and carryRWD == 0:
        next_state = 14
    elif idx_inp_sensor == 9 and carryRWD == 1:
        next_state = 13
    elif idx_inp_sensor == 10:
        next_state = 15
    elif idx_inp_sensor == 11 and carryRWD == 0:
        next_state = 17
    elif idx_inp_sensor == 11 and carryRWD == 1:
        next_state = 16
    elif idx_inp_sensor == 12:
        next_state = 18
    elif idx_inp_sensor == 15:
        next_state = 19
    elif idx_inp_sensor == 17:
        next_state = 20
    elif idx_inp_sensor == 20 and carryRWD == 0:
        next_state = 22
    elif idx_inp_sensor == 20 and carryRWD == 1:
        next_state = 21
    elif idx_inp_sensor == 16:
        next_state = 23
    else:
        next_state = 0
    
    return next_state

# possible states of the agent
states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

# rewards for each state
rewards = [0, -5, -10, 0, 1, 0, 10, 50, -3, -5, -5, -1, 3, -1, 3, -10, -5, 3, -3, -10, 30, 100, -1, -100]

# possible actions the agent can take
actions = [0, 1, 2, 3, 4, 5, 6]

# initial Q table
Q = np.zeros((len(states), len(actions)))

# learning rate
alpha = 0.1

# discount factor
gamma = 0.9

# exploration function
epsilon = 1.0
final_epsilon = 0.05
epsilon_decay_rate = 0.95

# choose an action using epsilon-greedy policy
def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(actions)
    else:
        return np.argmax(Q[state])
    
num_episodes = 50

for episode in range(num_episodes):
    restart()
    state = np.random.choice(states)
    done = False
    carryRWD = 0

    while not done:
        action = choose_action(state)
        outy = map_output_neurons(action)

        next_state = get_next_state(outy)
        reward = rewards[next_state]

        Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]))

        state = next_state

        if state == 21:
            done = True
    
    epsilon = max(final_epsilon, epsilon * epsilon_decay_rate)
    print(episode)
    
    print(Q)

print(Q)

for state in range(len(states)):
    total_prob = np.sum(Q[state])
    if total_prob > 0:
        Q[state] /= total_prob

print(Q)

# input sensors
    # [ 0] inp_nothing
    # [ 1] inp_breeze
    # [ 2] inp_danger
    # [ 3] inp_flash_cr
    # [ 4] inp_flash_ncr
    # [ 5] inp_goal_cr
    # [ 6] inp_goal_ncr
    # [ 7] inp_initial_cr
    # [ 8] inp_initial_ncr
    # [ 9] inp_obstruction
    # [10] inp_stench
    # [11] inp_bf_cr
    # [12] inp_bf_ncr
    # [13] inp_bfs_cr
    # [14] inp_bfs_ncr
    # [15] inp_bs
    # [16] inp_fs_cr
    # [17] inp_fs_ncr
    # [18] inp_boundary
    # [19] inp_cannot
    # [20] inp_grabbed
    # [21] inp_success_cr
    # [22] inp_success_ncr
    # [23] inp_died

# output neurons
    # [ 0] out_act_grab
    # [ 1] out_act_leave
    # [ 2] out_act_nill
    # [ 3] out_mov_forward
    # [ 4] out_rot_left
    # [ 5] out_rot_right
    # [ 6] out_rot_back