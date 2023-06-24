import random
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

def infer(vecInpSens: np.int32) -> int:
    # define transition matrix
    transition_matrix = [
        [0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.15, 0.0],
        [0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.15, 0.05],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.4],
        [0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.0],
        [0.8, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.3, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.8],
        [0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.15, 0.05],
        [0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.0],
        [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.1],
        [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.1],
        [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.1],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]

    print('infer: ', len(vecInpSens), ' ', vecInpSens) # type: ignore
    outy = -1

    if len(vecInpSens) == 1: # type: ignore
        if np.sum(vecInpSens) == 0:
            return outy
        else:
            enabled_sensor_i = np.where(vecInpSens[0] == 1)[0][0] # type: ignore
            outcomes = transition_matrix[enabled_sensor_i]
            outy = np.random.choice(len(outcomes), p=outcomes)
            print('out: ', OutNeurons[outy])

    elif len(vecInpSens) > 1: # type: ignore
        for k in range(len(vecInpSens)): # type: ignore
            if np.sum(vecInpSens[k]) != 1: # type: ignore
                return outy
            else:
                enabled_sensor_i = np.where(vecInpSens[0] == 1)[0][0] # type: ignore
                outcomes = transition_matrix[enabled_sensor_i]
                outy = np.random.choice(len(outcomes), p=outcomes)
                print('out: ', OutNeurons[outy])

    else:
        return outy
    return outy

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

def interpreting(envisim_answ: str) -> tuple[Stt, str, int, np.int32]:
    jobj = json.loads(envisim_answ)
    str_code = ''
    stt_mm = Stt.DECIDING
    idx_inp_sns: int = 0
    CurrSensBits = np.zeros(32, dtype=np.int32)

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

    elif keyMwpOUT in jobj:
        jrasc = jobj[keyMwpOUT]
        if OUTrst in jrasc:
            idx_inp_sns = InpSensors.index('inp_' + OUTrst)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_' + OUTrst
        elif OUTgrb in jrasc:
            idx_inp_sns = InpSensors.index('inp_' + OUTgrb)
            CurrSensBits[idx_inp_sns] |= 0b1
        elif OUTdie in jrasc:
            idx_inp_sns = InpSensors.index('inp_' + OUTdie)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = OUTdie
            stt_mm = Stt.EXCEPTIONS
        elif OUTsuc in jrasc:
            idx_inp_sns = InpSensors.index('inp_' + OUTsuc)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = OUTsuc
            stt_mm = Stt.EXCEPTIONS
        elif OUTcnt in jrasc:
            idx_inp_sns = InpSensors.index('inp_' + OUTcnt)
            CurrSensBits[idx_inp_sns] |= 0b1
        elif OUTnon in jrasc:
            idx_inp_sns = InpSensors.index('inp_' + OUTnon)
            CurrSensBits[idx_inp_sns] |= 0b1
        else:
            str_code = 'undefined_outcome'
            stt_mm = Stt.ERRORS

    elif keyMwpCOL in jobj:
        jrasc = jobj[keyMwpCOL]
        if CLDbnd in jrasc:
            idx_inp_sns = InpSensors.index('inp_' + CLDbnd)
            CurrSensBits[idx_inp_sns] |= 0b1
        elif CLDobs in jrasc:
            idx_inp_sns = InpSensors.index('inp_' + CLDobs)
            CurrSensBits[idx_inp_sns] |= 0b1
        elif CLDwll in jrasc:
            idx_inp_sns = InpSensors.index('inp_' + CLDwll)
            CurrSensBits[idx_inp_sns] |= 0b1
        else:
            print('Attention: collision came - undefined - ?!')
            str_code = 'undefined_collision'
            stt_mm = Stt.ERRORS

    elif keyMwpSNS in jobj:
        jrasc = jobj[keyMwpSNS]
        if len(jrasc) == 3:
            if (SNSbrz in jrasc) and (SNSfsh in jrasc) and (SNStch in jrasc):
                idx_inp_sns = InpSensors.index('inp_bfs')
                CurrSensBits[idx_inp_sns] |= 0b1
        elif len(jrasc) == 2:
            if (SNSbrz in jrasc) and (SNSfsh in jrasc):
                idx_inp_sns = InpSensors.index('inp_bf')
                CurrSensBits[idx_inp_sns] |= 0b1
            elif (SNSbrz in jrasc) and (SNStch in jrasc):
                idx_inp_sns = InpSensors.index('inp_bs')
                CurrSensBits[idx_inp_sns] |= 0b1
            elif (SNSfsh in jrasc) and (SNStch in jrasc):
                idx_inp_sns = InpSensors.index('inp_fs')
                CurrSensBits[idx_inp_sns] |= 0b1
        elif len(jrasc) > 0:
            for item in jrasc:
                if SNSfsh in item:
                    idx_inp_sns = InpSensors.index('inp_' + SNSfsh)
                    CurrSensBits[idx_inp_sns] |= 0b1
                elif SNSdng in item:
                    idx_inp_sns = InpSensors.index('inp_' + SNSdng)
                    CurrSensBits[idx_inp_sns] |= 0b1
                elif SNSobs in item:
                    idx_inp_sns = InpSensors.index('inp_' + SNSobs)
                    CurrSensBits[idx_inp_sns] |= 0b1
                elif SNSgol in item:
                    idx_inp_sns = InpSensors.index('inp_' + SNSgol)
                    CurrSensBits[idx_inp_sns] |= 0b1
                elif SNSini in item:
                    idx_inp_sns = InpSensors.index('inp_' + SNSini)
                    CurrSensBits[idx_inp_sns] |= 0b1
                elif SNSbrz in item:
                    idx_inp_sns = InpSensors.index('inp_' + SNSbrz)
                    CurrSensBits[idx_inp_sns] |= 0b1
                elif SNStch in item:
                    idx_inp_sns = InpSensors.index('inp_' + SNStch)
                    CurrSensBits[idx_inp_sns] |= 0b1
        else:
            idx_inp_sns = InpSensors.index('inp_' + SNSnth)
            CurrSensBits[idx_inp_sns] |= 0b1

    if keyMwpDIR in jobj:
        jrasc = jobj[keyMwpDIR]
        if DIRn in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRn)
            CurrSensBits[idx_inp_sns] |= 0b1
        elif DIRne in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRne)
            CurrSensBits[idx_inp_sns] |= 0b1
        elif DIRe in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRe)
            CurrSensBits[idx_inp_sns] |= 0b1
        elif DIRse in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRse)
            CurrSensBits[idx_inp_sns] |= 0b1
        elif DIRs in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRs)
            CurrSensBits[idx_inp_sns] |= 0b1
        elif DIRsw in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRsw)
            CurrSensBits[idx_inp_sns] |= 0b1
        elif DIRw in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRw)
            CurrSensBits[idx_inp_sns] |= 0b1
        elif DIRnw in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRnw)
            CurrSensBits[idx_inp_sns] |= 0b1
        else:
            print('Atenção: DIRECTION veio - indefinido - ?!')
            str_code = 'direcao_indefinida'
            stt_mm = Stt.ERRORS

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
