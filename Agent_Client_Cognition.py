import json
from typing import List

import numpy as np
import sys

from Agent_Client_Setup import Stt, InpSensors, OutNeurons

from Agent_Client_Setup import keyMagACT, keyMagMOV, keyMagREQ, keyMagROT, ACTgrb, ACTlev, ACTnil, \
    MOVfor, REQfwd, REQlft, REQl45, REQori, REQrst, REQrgt, REQr45, ROTlft, ROTrgt, ROTbck, \
    keyMwpSNS, keyMwpCOL, keyMwpOUT, keyMwpSRV, keyMwpPHR, keyMwpPOS, keyMwpDIR, keyMwpDVA, \
    SNSbrz, SNSdng, SNSfsh, SNSgol, SNSini, SNSobs, SNStch, SNSnth, CLDbnd, CLDobs, CLDwll, \
    OUTcnt, OUTdie, OUTgrb, OUTnon, OUTrst, OUTsuc, SRVcnn, SRVinv, SRVnor, SRVpsd, \
    DIRn, DIRne, DIRe, DIRse, DIRs, DIRsw, DIRw, DIRnw

# analyze the response/feedback received from EnviSim
def feedback_analysis(vecInpSens: np.int32, carryRWD: int) -> int:
    outy = -1

    if np.sum(vecInpSens) != 1:
        return outy
    
    else:
        inx = np.argmax(vecInpSens)
        tmpStr: str = InpSensors[inx]

        if tmpStr == 'inp_' + SNSgol and carryRWD == 0:
            outy = OutNeurons.index("out_act_grab")
        elif tmpStr == 'inp_' + SNSini and carryRWD == 1:
            outy = OutNeurons.index("out_act_leave")
        elif tmpStr == 'inp_' + OUTgrb:
            outy = 50
        elif tmpStr == 'inp_' + OUTsuc and carryRWD == 1:
            outy = 100
        elif tmpStr == 'inp_' + OUTdie:
            outy = -100
        else:
            outy = OutNeurons.index("out_act_nill")

    return outy

# Q = np.array([
#     [0.08180034400221003, 0.266033515193871, 0.12799444282201322, 0.18983362466403386, 0.33433807331787196],
#     [0.008181133633713976, 0.013890189971975822, 0.2086000514991492, 0.020326629867513827, 0.7490019950276472],
#     [0.004422226711502339, 1.2965774867854421e-42, 0.8420393110825323, 0.10160128353911137, 0.051937178666853816],
#     [8.529481162998041e-05, 0.9987070151027806, 0.0007279171357575584, 0.00025544225345745233, 0.00022433069637419026],
#     [0.9999999992128052, 7.870590148910355e-10, 1.420880113411678e-16, 3.3671640503473824e-18, 1.354776183202348e-13],
#     [0.03074346872455458, 0.04824808041750167, 0.21630540291014458, 0.7018147434639427, 0.0028883044838564874],
#     [0.031653309211899835, 0.025298561521541228, 0.3058943880434051, 0.0847389763102941, 0.5524147649128598],
#     [0.0005266551564234157, 0.000995591440440786, 0.01780788860869265, 0.9742766566512263, 0.0063932081432167874],
#     [0.17770354161378743, 0.17770354161378743, 0.15375312252840848, 0.17770354161378743, 0.3131362526302292],
#     [0.13802874848911956, 0.09899047158929238, 0.48692328294334897, 0.13802874848911956, 0.13802874848911956],
#     [0.07866313005272887, 0.07698011908315296, 0.5255394759118572, 0.11151809082932468, 0.20729918412293621],
#     [0.004686208265746284, 0.0005166552038717102, 0.004528222383582947, 0.004229389876167994, 0.9860395242706311],
#     [0.0016198039924556623, 0.005193600241928922, 0.0407213903060938, 0.061352950921166084, 0.8911122545383556],
#     [0.0016267781897223686, 0.00016561169091898446, 0.0032369616667738527, 0.0006894635542839868, 0.9942811848983008],
#     [0.0044795549748038515, 0.053164184317425034, 0.08061752815198771, 0.007727010257526148, 0.8540117222982573],
#     [0.09504240191348279, 0.18934420260095214, 0.4094352023602059, 0.1405229160620524, 0.1656552770633068]
# ])
Q = np.array([
[0.006941107646815398, 0.19449600677063456, 0.5490227727668615, 0.08835724460601517, 0.16118286820967337],
[0.007463816952670078, 0.03559509308948595, 0.3424541112065599, 0.08626915739190076, 0.5282178213593833],
[0.017023185804342006, 6.486031358969777e-44, 0.20071473525726968, 0.6237906428788383, 0.15847143605955005],
[1.0, 3.3651098650972725e-44, 3.0119932244259276e-45, 1.4683367136011848e-45, 5.627489096660387e-45],
[5.112623518111543e-09, 0.9999999297694742, 6.538900352543691e-09, 7.031709448990578e-09, 5.1547292689726454e-08],
[0.004724635419903378, 0.277310886620949, 0.28533436359472586, 0.4093845176333369, 0.023245596731084926],
[0.004870723453836984, 0.02797766947667226, 0.1298348553099361, 0.7637656750597372, 0.07355107669981738],
[0.0003625453518802473, 0.005481879200681162, 0.016861242495062236, 0.949097013987578, 0.0281973189647986],
[0.2046426016771879, 0.23396245076182798, 0.17414193322056257, 0.2046426016771879, 0.18261041266323377],
[0.0627073781514589, 0.19049982489553882, 0.2619465239482941, 0.2316214765072389, 0.25322479649746926],
[0.21178881709687314, 0.1528447316125074, 0.21178881709687314, 0.21178881709687314, 0.21178881709687314],
[3.7425442316916324e-05, 0.9995202948596346, 0.00010910346582975826, 0.00029891703051952516, 3.425920169910054e-05],
[0.01698256763735707, 0.08391724451810646, 0.3860108157264352, 0.38004244506406504, 0.13304692705403612],
[0.009977086935029036, 0.02366377889918398, 0.2352209281276182, 0.11684682059099431, 0.6142913854471744],
[0.030473746682773868, 0.3633125894890824, 0.05720794484206974, 0.3343545210007029, 0.21465119798537105],
[0.15186234488833072, 0.09588470857787242, 0.263336677047496, 0.23505465754473037, 0.2538616119415706]])

def map_outy(action):
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

# it is the agent intelligence
def infer(vecInpSens: np.int32, carryRWD: int) -> int:
    outy = -1

    print('infer: ', len(vecInpSens), ' ', vecInpSens) # type: ignore
    
    if len(vecInpSens) == 1: # type: ignore
        if np.sum(vecInpSens) == 0:
            return outy
        else:
            state = np.where(vecInpSens[0] == 1)[0][0] # type: ignore
            action_prob = Q[state]
            action = np.random.choice(len(action_prob), p=action_prob )
            outy = map_outy(action)
            print('out: ', OutNeurons[outy])

    return outy

# create a message to the EnviSim requesting infos from Wumpus World
def create_msg(indx_out: int, dist: int) -> str:
    rasc: str = OutNeurons[indx_out]
    msg = ''

    if rasc == 'out_req_' + REQfwd:
        msg = '{\"' + keyMagREQ + '\":[\"' + REQfwd + '\",' + str(dist) + ']}'
    elif rasc == 'out_req_' + REQlft:
        msg = '{\"' + keyMagREQ + '\":[\"' + REQlft + '\",' + str(dist) + ']}'
    elif rasc == 'out_req_' + REQl45:
        msg = '{\"' + keyMagREQ + '\":[\"' + REQl45 + '\",' + str(dist) + ']}'
    elif rasc == 'out_req_' + REQori:
        msg = '{\"' + keyMagREQ + '\":[\"' + REQori + '\",' + str(dist) + ']}'
    elif rasc == 'out_req_' + REQrst:
        msg = '{\"' + keyMagREQ + '\":[\"' + REQrst + '\",' + str(0) + ']}'
    elif rasc == 'out_req_' + REQrgt:
        msg = '{\"' + keyMagREQ + '\":[\"' + REQrgt + '\",' + str(dist) + ']}'
    elif rasc == 'out_req_' + REQr45:
        msg = '{\"' + keyMagREQ + '\":[\"' + REQr45 + '\",' + str(dist) + ']}'
    elif rasc == 'out_act_' + ACTgrb:
        msg = '{\"' + keyMagACT + '\":[\"' + ACTgrb + '\",' + str(dist) + ']}'
    elif rasc == 'out_act_' + ACTlev:
        msg = '{\"' + keyMagACT + '\":[\"' + ACTlev + '\",' + str(dist) + ']}'
    elif rasc == 'out_act_' + ACTnil:
        msg = '{\"' + keyMagACT + '\":[\"' + ACTnil + '\",' + str(dist) + ']}'
    elif rasc == 'out_mov_' + MOVfor:
        msg = '{\"' + keyMagMOV + '\":[\"' + MOVfor + '\",' + str(dist) + ']}'
    elif rasc == 'out_rot_' + ROTlft:
        msg = '{\"' + keyMagROT + '\":[\"' + ROTlft + '\",' + str(2) + ']}'
    elif rasc == 'out_rot_' + ROTrgt:
        msg = '{\"' + keyMagROT + '\":[\"' + ROTrgt + '\",' + str(2) + ']}'
    elif rasc == 'out_rot_' + ROTbck:
        msg = '{\"' + keyMagROT + '\":[\"' + ROTbck + '\",' + str(4) + ']}'

    return msg

# interpret the message sent by EnviSim
def interpreting(envisim_answ: str) -> tuple[Stt, str, int, np.int32]:
    jobj = json.loads(envisim_answ)
    str_code = ''
    stt_mm = Stt.DECIDING
    idx_inp_sns: int = 0
    CurrSensBits = np.zeros(32, dtype=np.int32)

    # if the message contains the key "server"
    if keyMwpSRV in jobj:
        jrasc = jobj[keyMwpSRV]
        if SRVcnn in jrasc:
            str_code = SRVcnn
            stt_mm = Stt.RESTARTING

        elif SRVinv in jrasc:
            str_code = 'msg_invalid'
            stt_mm = Stt.ERRORS

        elif SRVpsd in jrasc:
            str_code = 'server_paused'
            stt_mm = Stt.ERRORS

        elif SRVnor in jrasc:
            str_code = 'server_normal'
            stt_mm = Stt.ERRORS

    # if the message contains the key "outcome"
    elif keyMwpOUT in jobj:
        jrasc = jobj[keyMwpOUT]

        if OUTrst in jrasc:
            idx_inp_sns = InpSensors.index('inp_' + OUTrst)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_' + OUTrst

        elif OUTgrb in jrasc:
            idx_inp_sns = InpSensors.index('inp_' + OUTgrb)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_' + OUTgrb

        elif OUTdie in jrasc:
            idx_inp_sns = InpSensors.index('inp_' + OUTdie)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_' + OUTdie
            stt_mm = Stt.EXCEPTIONS

        elif OUTsuc in jrasc:
            idx_inp_sns = InpSensors.index('inp_' + OUTsuc)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_' + OUTsuc
            stt_mm = Stt.EXCEPTIONS

        elif OUTcnt in jrasc:
            idx_inp_sns = InpSensors.index('inp_' + OUTcnt)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_' + OUTcnt

        elif OUTnon in jrasc:
            idx_inp_sns = InpSensors.index('inp_' + OUTnon)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_' + OUTnon

        else:
            str_code = 'undefined_outcome'
            stt_mm = Stt.ERRORS

    # if the message contains the key "collision"
    elif keyMwpCOL in jobj:
        jrasc = jobj[keyMwpCOL]

        if CLDbnd in jrasc:
            idx_inp_sns = InpSensors.index('inp_' + CLDbnd)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_' + CLDbnd

        elif CLDobs in jrasc:
            idx_inp_sns = InpSensors.index('inp_' + CLDobs)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_' + CLDobs

        elif CLDwll in jrasc:
            idx_inp_sns = InpSensors.index('inp_' + CLDwll)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_' + CLDwll

        else:
            print('Attention: collision came - undefined - ?!')
            str_code = 'undefined_collision'
            stt_mm = Stt.ERRORS

    # if the message contains the key "sense"
    elif keyMwpSNS in jobj:
        jrasc = jobj[keyMwpSNS]

        if len(jrasc) == 3:
            if (SNSbrz in jrasc) and (SNSfsh in jrasc) and (SNStch in jrasc):
                idx_inp_sns = InpSensors.index('inp_bfs')
                CurrSensBits[idx_inp_sns] |= 0b1
                str_code = 'inp_bfs'

        elif len(jrasc) == 2:
            if (SNSbrz in jrasc) and (SNSfsh in jrasc):
                idx_inp_sns = InpSensors.index('inp_bf')
                CurrSensBits[idx_inp_sns] |= 0b1
                str_code = 'inp_bf'

            elif (SNSbrz in jrasc) and (SNStch in jrasc):
                idx_inp_sns = InpSensors.index('inp_bs')
                CurrSensBits[idx_inp_sns] |= 0b1
                str_code = 'inp_bs'

            elif (SNSfsh in jrasc) and (SNStch in jrasc):
                idx_inp_sns = InpSensors.index('inp_fs')
                CurrSensBits[idx_inp_sns] |= 0b1
                str_code = 'inp_fs'

        elif len(jrasc) > 0:
            for item in jrasc:
                if SNSfsh in item:
                    idx_inp_sns = InpSensors.index('inp_' + SNSfsh)
                    CurrSensBits[idx_inp_sns] |= 0b1
                    str_code = 'inp_' + SNSfsh

                elif SNSdng in item:
                    idx_inp_sns = InpSensors.index('inp_' + SNSdng)
                    CurrSensBits[idx_inp_sns] |= 0b1
                    str_code = 'inp_' + SNSdng

                elif SNSobs in item:
                    idx_inp_sns = InpSensors.index('inp_' + SNSobs)
                    CurrSensBits[idx_inp_sns] |= 0b1
                    str_code = 'inp_' + SNSobs
                    
                elif SNSgol in item:
                    idx_inp_sns = InpSensors.index('inp_' + SNSgol)
                    CurrSensBits[idx_inp_sns] |= 0b1
                    str_code = 'inp_' + SNSgol

                elif SNSini in item:
                    idx_inp_sns = InpSensors.index('inp_' + SNSini)
                    CurrSensBits[idx_inp_sns] |= 0b1
                    str_code = 'inp_' + SNSini

                elif SNSbrz in item:
                    idx_inp_sns = InpSensors.index('inp_' + SNSbrz)
                    CurrSensBits[idx_inp_sns] |= 0b1
                    str_code = 'inp_' + SNSbrz

                elif SNStch in item:
                    idx_inp_sns = InpSensors.index('inp_' + SNStch)
                    CurrSensBits[idx_inp_sns] |= 0b1
                    str_code = 'inp_' + SNStch
        else:
            idx_inp_sns = InpSensors.index('inp_' + SNSnth)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_' + SNSnth

    # if the message contains the key "direction"
    if keyMwpDIR in jobj:
        jrasc = jobj[keyMwpDIR]

        if DIRn in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRn)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_dir_' + DIRn

        elif DIRne in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRne)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_dir_' + DIRne

        elif DIRe in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRe)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_dir_' + DIRe

        elif DIRse in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRse)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_dir_' + DIRse

        elif DIRs in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRs)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_dir_' + DIRs

        elif DIRsw in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRsw)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_dir_' + DIRsw

        elif DIRw in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRw)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_dir_' + DIRw

        elif DIRnw in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRnw)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_dir_' + DIRnw

        else:
            print('Atenção: DIRECTION veio - indefinido - ?!')
            str_code = 'direcao_indefinida'
            stt_mm = Stt.ERRORS

    # if the message contains the key "pheromone" (optional)
    if keyMwpPHR in jobj:
        jrasc = jobj[keyMwpPHR]

        if len(jrasc) != 1:
            print('Atenção: PHEROMONE chegou - indefinido - ?!')
            str_code = 'feromônio_indefinido'
            stt_mm = Stt.ERRORS

        else:
            pherom = jrasc[0]
            idx_inp_sns = InpSensors.index('inp_' + keyMwpPHR)
            CurrSensBits[idx_inp_sns] |= 0b1

    # if the message contains the key "devitation" (optional)
    if keyMwpDVA in jobj:
        jrasc = jobj[keyMwpDVA]

        if len(jrasc) != 1:
            print('Atenção: ângulo DEVIATION veio - indefinido - ?!')
            str_code = 'desvio_indefinido'
            stt_mm = Stt.ERRORS

        else:
            devAngle = jrasc[0]
            idx_inp_sns = InpSensors.index('inp_' + keyMwpDVA)
            CurrSensBits[idx_inp_sns] |= 0b1

    # if the message contains the key "position" (optional)
    if keyMwpPOS in jobj:
        stt_mm = Stt.EXCEPTIONS
        jrasc = jobj[keyMwpPOS]

        if len(jrasc) != 2:
            print('Atenção: POSIÇÃO recebida - indefinida - ?!')
            str_code = 'posição_indefinida'
            stt_mm = Stt.ERRORS

        else:
            posX = jrasc[0]
            posY = jrasc[1]
            str_code = keyMwpPOS
            CurrSensBits[idx_inp_sns] |= 0b1
    
    return stt_mm, str_code, idx_inp_sns, CurrSensBits # type: ignore
