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
    msg = '{"request":["restart",0]}'
    sock.sendall(msg.encode("utf-8"))
    answES = sock.recv(256)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# choose an action using epsilon-greedy policy
def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
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
    msg = '{"request":["forward",1]}'
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

# get the next state
def get_next_state(outy):
    msg = create_msg(outy, 1)
    idx_inp_sensor = send_cmd(msg)

    next_state = idx_inp_sensor

    if idx_inp_sensor == 15:
        next_state = 13
    elif idx_inp_sensor == 16:
        next_state = 14
    elif idx_inp_sensor == 17:
        next_state = 15
    
    return next_state

# possible states of the agent
states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# rewards for each state
rewards = [0, -5, -5, 1, 10, 0, -1, -5, 3, 3, -5, 3, -3, -5, -100, 200]

# possible actions the agent can take
actions = [0, 1, 2, 3, 4]

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

num_episodes = 100

for episode in range(num_episodes):
    restart()
    state = np.random.choice(states)
    done = False

    while not done:
        action = choose_action(state)
        outy = map_output_neurons(action)

        next_state = get_next_state(outy)
        reward = rewards[next_state]

        Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]))

        state = next_state

        if state == 15:
            done = True
    
    epsilon = max(final_epsilon, epsilon * epsilon_decay_rate)

    print(episode, epsilon)
    
    print(Q)

print(Q)

for row in Q:
    row_str = ', '.join([str(value) for value in row])
    print(row_str)