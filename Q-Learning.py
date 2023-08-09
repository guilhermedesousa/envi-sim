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

# restart the EnviSim
def restart():
    msg = '{"request":["restart",1]}'
    sock.sendall(msg.encode("utf-8"))
    sock.recv(256)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# choose an action using epsilon-greedy policy
def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        if state == 0:
            return np.random.choice([1,3])
        elif state == 4:
            return np.random.choice([1,2])
        elif state == 20:
            return np.random.choice([0,3])
        elif state == 24:
            return np.random.choice([0,2])
        elif state in [1,2,3]:
            return np.random.choice([1,2,3])
        elif state in [21,22,23]:
            return np.random.choice([0,2,3])
        elif state in [5,10,15]:
            return np.random.choice([0,1,3])
        elif state in [9,14,19]:
            return np.random.choice([0,1,2,])
        else:
            return np.random.choice(actions)
    else:
        action_logits = Q[state]
        action_probs = softmax(action_logits)
        return np.random.choice(len(action_probs), p=action_probs)

# map the chosen action with the correct for EnviSim
def map_output_neurons(action):

    outy = action

    if action == 1:
        outy = 3
    elif action == 2:
        outy = 11
    elif action == 3:
        outy = 12
    elif action == 4:
        outy = 13
    
    return outy

# request forward for information
def request_forward():
    msg = '{"request":["forward",0]}'
    sock.sendall(msg.encode("utf-8"))
    answES = sock.recv(256)
    answES = answES.decode("utf-8")
    sttMM, strCode, idxInpSensor, CurrentSensBits = interpreting(answES)

    return idxInpSensor

# send the chosen command to EnviSim
def send_cmd(msg):
    sock.sendall(msg.encode("utf-8"))
    answES = sock.recv(256)
    return interpret(answES)

# interpret the message sent by EnviSim
def interpret(answES):
    sttMM, strCode, idxInpSensor, CurrentSensBits = interpreting(answES)

    # request_sensors = [19, 22, 23, 24, 25, 26, 27, 28, 29]

    if idxInpSensor in [12, 16, 15, 17]:
        return idxInpSensor
    
    return request_forward()

next_state_matrix = [
    [0, 5, 0, 1],
    [0, 6, 0, 2],
    [0, 7, 1, 3],
    [0, 8, 2, 4],
    [0, 9, 3, 0],
    [0, 10, 0, 6],
    [1, 11, 5, 7],
    [2, 12, 6, 8],
    [3, 13, 7, 9],
    [4, 14, 8, 0],
    [5, 15, 0, 11],
    [6, 16, 10, 12],
    [7, 17, 11, 13],
    [8, 18, 12, 14],
    [9, 19, 13, 0],
    [10, 20, 0, 16],
    [11, 21, 15, 17],
    [12, 22, 16, 18],
    [13, 23, 17, 19],
    [14, 24, 18, 0],
    [15, 0, 0, 21],
    [16, 0, 20, 22],
    [17, 0, 21, 23],
    [18, 0, 22, 24],
    [19, 0, 23, 0]
]

# possible states of the agent
states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

# possible actions the agent can take (cima, baixo, esq, dir)
actions = [0, 1, 2, 3]

# rewards for each state
rewards = [-1, -5, 1, -1, -5, 0, 1, 100, 2, -1, 0, -1, -5, -1, 0, 0, 0, -1, -1, 0, 0, 0, -1, -5, -1]

# initial Q table
Q = np.zeros((len(states), len(actions)))

# learning rate
alpha = 0.1

# discount factor
gamma = 0.3

# exploration function
epsilon = 1.0
final_epsilon = 0.05
epsilon_decay_rate = 0.95

num_episodes = 50

for episode in range(num_episodes):
    restart()
    state = np.random.choice(states)
    done = False

    while not done:
        action = choose_action(state)
        next_state = next_state_matrix[state][action]
        reward = rewards[next_state]

        Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]))

        state = next_state

        if state == 7:
            done = True
    
    epsilon = max(final_epsilon, epsilon * epsilon_decay_rate)

    print(episode, epsilon)
    
    print(Q)

print(Q)

for row in Q:
    row_str = ', '.join([str(value) for value in row])
    print(row_str)