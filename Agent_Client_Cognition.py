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

Q = np.array([
    [0.13972651, 0.15275881, 0.13347362, 0.13584955, 0.14697476, 0.15375406, 0.13746268],
    [0.13604406, 0.18227582, 0.13376651, 0.14214774, 0.13757252, 0.13623029, 0.13196306],
    [0.1289185, 0.14392932, 0.12237831, 0.19241407, 0.13595801, 0.14465369, 0.13174808],
    [0.15099224, 0.1444561, 0.13925211, 0.13766547, 0.14090492, 0.1472244, 0.13950476],
    [0.11587224, 0.05545833, 0.12592705, 0.13800077, 0.30525551, 0.14042072, 0.11906539],
    [0.1330861, 0.13472295, 0.13488372, 0.13785871, 0.13383965, 0.19165211, 0.13395676],
    [0.51557866, 0.06708847, 0.11345559, 0.08225574, 0.06786297, 0.09914788, 0.05461069],
    [0.13093778, 0.12566389, 0.1089043, 0.10799773, 0.17519987, 0.17619644, 0.17509998],
    [0.14236679, 0.14288959, 0.14292098, 0.14333099, 0.14393267, 0.13924707, 0.14531191],
    [0.14031057, 0.15780257, 0.1275076, 0.1191475, 0.13003283, 0.14271858, 0.18248035],
    [0.1478532, 0.13988281, 0.14507155, 0.13871054, 0.15149199, 0.14340421, 0.13358569],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, -0.3, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.13707521, 0.14042032, 0.13862324, 0.14106565, 0.15886825, 0.14201589, 0.14193144],
    [0.14165485, 0.14477003, 0.14423204, 0.14495013, 0.14513241, 0.13678523, 0.14247531],
    [0.16466088, 0.09054349, 0.09003812, 0.13699475, 0.08609546, 0.19430515, 0.23736216],
    [0.12367087, 0.11770681, 0.11671571, 0.12740338, 0.20361944, 0.14233358, 0.16855021],
    [0.13708399, 0.13098162, 0.13214198, 0.12797235, 0.18905153, 0.13157175, 0.15119679],
    [0.15455636, 0.1146388, 0.06914152, 0.0611106, 0.21260017, 0.18932639, 0.19862616],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.09246354, 0.06095317, 0.40495383, 0.10352869, 0.07461988, 0.06775176, 0.19572914],
    [0.12159479, 0.13307521, 0.10711267, 0.10343907, 0.21023668, 0.16351323, 0.16102836]
])

def map_input_sensor(enabled_sensor: int, carryRWD: int) -> int:
    outy = enabled_sensor

    if enabled_sensor == 3 and carryRWD == 0:
        outy = 4
    elif enabled_sensor == 3 and carryRWD == 1:
        outy = 3
    elif enabled_sensor == 4 and carryRWD == 0:
        outy = 6
    elif enabled_sensor == 4 and carryRWD == 1:
        outy = 5
    elif enabled_sensor == 5 and carryRWD == 0:
        outy = 8
    elif enabled_sensor == 5 and carryRWD == 1:
        outy = 7
    elif enabled_sensor == 6:
        outy = 9
    elif enabled_sensor == 7:
        outy = 10
    elif enabled_sensor == 8 and carryRWD == 0:
        outy = 12
    elif enabled_sensor == 8 and carryRWD == 1:
        outy = 11
    elif enabled_sensor == 9 and carryRWD == 0:
        outy = 14
    elif enabled_sensor == 9 and carryRWD == 1:
        outy = 13
    elif enabled_sensor == 10:
        outy = 15
    elif enabled_sensor == 11 and carryRWD == 0:
        outy = 17
    elif enabled_sensor == 11 and carryRWD == 1:
        outy = 16
    elif enabled_sensor == 12:
        outy = 18
    elif enabled_sensor == 15:
        outy = 19
    elif enabled_sensor == 16:
        outy = 23
    elif enabled_sensor == 17:
        outy = 20
    elif enabled_sensor == 20 and carryRWD == 0:
        outy = 22
    elif enabled_sensor == 20 and carryRWD == 1:
        outy = 21

    return outy

# it is the agent intelligence
def infer(vecInpSens: np.int32, carryRWD: int) -> int:
    outy = -1

    print('infer: ', len(vecInpSens), ' ', vecInpSens) # type: ignore
    
    if len(vecInpSens) == 1: # type: ignore
        if np.sum(vecInpSens) == 0:
            return outy
        else:
            enabled_sensor_i = np.where(vecInpSens[0] == 1)[0][0] # type: ignore
            state = map_input_sensor(enabled_sensor_i, carryRWD)
            action_prob = Q[state]
            outy = np.random.choice(len(action_prob), p=action_prob )
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
