from enum import Enum
import json
import numpy as np

# defines different FSM states of the agent
class Stt(Enum):
    BEGIN = 1
    RECEIVING = 2
    INTERPRETING = 3
    SENDING = 4
    DECIDING = 5
    REQUESTING = 6
    RESTARTING = 7
    EXCEPTIONS = 8
    ERRORS = 9
    FOR_TESTS = 10

# defines different sub-FSM states for the agent
class SubStt(Enum):
    RES = 1
    START = 2
    ASK = 3
    SAVE = 4
    CMD = 5
    CNT = 6
    WAITRQ = 7
    WAITCM = 8
    WAITCM2 = 9

sttMM = Stt.BEGIN
sttSUBfsm = SubStt.RES

# list which contains a sequence of REQUESTS to EnviSim before DECIDES
InfoReqSeq = [["fwd", 1]]
nofInfoRequest = len(InfoReqSeq)

CurrentSensBits = np.zeros((nofInfoRequest, 32), dtype=np.int32)

# input sensors
InpSensors = ["inp_nothing",
              "inp_breeze",
              "inp_danger",
              "inp_flash",
              "inp_goal",
              "inp_initial",
              "inp_obstruction",
              "inp_stench",
              "inp_bf",
              "inp_bfs",
              "inp_bs",
              "inp_fs",
              "inp_boundary",
              "inp_obstacle",
              "inp_wall",
              "inp_cannot",
              "inp_died",
              "inp_grabbed",
              "inp_none",
              "inp_restarted",
              "inp_success",
              "inp_pheromone",
              "inp_dir_n",
              "inp_dir_ne",
              "inp_dir_e",
              "inp_dir_se",
              "inp_dir_s",
              "inp_dir_sw",
              "inp_dir_w",
              "inp_dir_nw",
              "inp_deviation",
              "go",
              "_0",
              "_1",
              "mGB",
              "mFLSH"
            ]

nofInpSensors = len(InpSensors) # 36

# output neurons
OutNeurons = ["out_act_grab",
              "out_act_leave",
              "out_act_nill",
              "out_mov_forward",
              "out_req_forward",
              "out_req_left",
              "out_req_left45",
              "out_req_orientation",
              "out_req_restart",
              "out_req_right",
              "out_req_right45",
              "out_rot_left",
              "out_rot_right",
              "out_rot_back"
              ]

# exchanged messages from AGENT to EnviSim
LstMsgsAGtoES = [[['act'], ['grab', 'leave', 'nill']],
                 [['move'], ['forward']],
                 [['request'], ['forward', 'left', 'left45', 'orientation', 'restart', 'right', 'right45']],
                 [['rotate'], ['left', 'right', 'back']]]

keyMagACT: str = LstMsgsAGtoES[0][0][0]
keyMagMOV: str = LstMsgsAGtoES[1][0][0]
keyMagREQ: str = LstMsgsAGtoES[2][0][0]
keyMagROT: str = LstMsgsAGtoES[3][0][0]

ACTgrb: str = LstMsgsAGtoES[0][1][0]
ACTlev: str = LstMsgsAGtoES[0][1][1]
ACTnil: str = LstMsgsAGtoES[0][1][2]

MOVfor: str = LstMsgsAGtoES[1][1][0]

REQfwd: str = LstMsgsAGtoES[2][1][0]
REQlft: str = LstMsgsAGtoES[2][1][1]
REQl45: str = LstMsgsAGtoES[2][1][2]
REQori: str = LstMsgsAGtoES[2][1][3]
REQrst: str = LstMsgsAGtoES[2][1][4]
REQrgt: str = LstMsgsAGtoES[2][1][5]
REQr45: str = LstMsgsAGtoES[2][1][6]

ROTlft: str = LstMsgsAGtoES[3][1][0]
ROTrgt: str = LstMsgsAGtoES[3][1][1]
ROTbck: str = LstMsgsAGtoES[3][1][2]

# exchanged messages from Envisim to AGENT
LstMsgEStoAG = [[['sense'], ['breeze', 'danger', 'flash', 'goal', 'initial', 'obstruction', 'stench']],
                [['collision'], ['boundary', 'obstacle', 'wall']],
                [['outcome'], ['cannot', 'died', 'grabbed', 'none', 'restarted', 'success']],
                [['server'], ['connected', 'invalid', 'normal', 'paused', 'ready', 'stopped']],
                [['pheromone'], []],
                [['position'], []],
                [['direction'], ['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw']],
                [['deviation'], []]]

keyMwpSNS: str = LstMsgEStoAG[0][0][0]
keyMwpCOL: str = LstMsgEStoAG[1][0][0]
keyMwpOUT: str = LstMsgEStoAG[2][0][0]
keyMwpSRV: str = LstMsgEStoAG[3][0][0]
keyMwpPHR: str = LstMsgEStoAG[4][0][0]
keyMwpPOS: str = LstMsgEStoAG[5][0][0]
keyMwpDIR: str = LstMsgEStoAG[6][0][0]
keyMwpDVA: str = LstMsgEStoAG[7][0][0]

SNSbrz: str = LstMsgEStoAG[0][1][0]
SNSdng: str = LstMsgEStoAG[0][1][1]
SNSfsh: str = LstMsgEStoAG[0][1][2]
SNSgol: str = LstMsgEStoAG[0][1][3]
SNSini: str = LstMsgEStoAG[0][1][4]
SNSobs: str = LstMsgEStoAG[0][1][5]
SNStch: str = LstMsgEStoAG[0][1][6]
SNSnth: str = 'nothing'

CLDbnd: str = LstMsgEStoAG[1][1][0]
CLDobs: str = LstMsgEStoAG[1][1][1]
CLDwll: str = LstMsgEStoAG[1][1][2]

OUTcnt: str = LstMsgEStoAG[2][1][0]
OUTdie: str = LstMsgEStoAG[2][1][1]
OUTgrb: str = LstMsgEStoAG[2][1][2]
OUTnon: str = LstMsgEStoAG[2][1][3]
OUTrst: str = LstMsgEStoAG[2][1][4]
OUTsuc: str = LstMsgEStoAG[2][1][5]

SRVcnn: str = LstMsgEStoAG[3][1][0]
SRVinv: str = LstMsgEStoAG[3][1][1]
SRVnor: str = LstMsgEStoAG[3][1][2]
SRVpsd: str = LstMsgEStoAG[3][1][3]
SRVrdy: str = LstMsgEStoAG[3][1][4]
SRVstp: str = LstMsgEStoAG[3][1][5]

DIRn: str = LstMsgEStoAG[6][1][0]
DIRne: str = LstMsgEStoAG[6][1][1]
DIRe: str = LstMsgEStoAG[6][1][2]
DIRse: str = LstMsgEStoAG[6][1][3]
DIRs: str = LstMsgEStoAG[6][1][4]
DIRsw: str = LstMsgEStoAG[6][1][5]
DIRw: str = LstMsgEStoAG[6][1][6]
DIRnw: str = LstMsgEStoAG[6][1][7]

iterNum = 0
energy = 500
carryRWD: int = 0
cntNofReqs: int = 0
msg = ''
answES = ''
posX = 0
posY = 0
direction: str = 'e'
pherom = 1
devAngle = 0

strCode = ''
idxInpSensor: int = 0
decision: int = 0
nextReturnOutIdx = 0

delaySec = 0.25