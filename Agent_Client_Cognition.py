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

direction = ''

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
    [0.011261525770126015, 0.9778224329345844, 0.010371035941550047, 0.0005450053537398026],
    [1.1146567193359942e-05, 0.9999467961942368, 6.610488388882947e-06, 3.5446750181231926e-05],
    [1.666825827863722e-12, 0.9999999999961551, 6.699053239794713e-13, 1.5082063783998588e-12],
    [0.00432734198801146, 0.9848896803711217, 0.006455635652855574, 0.00432734198801146],
    [0.25968181234769233, 0.22095456295692295, 0.25968181234769233, 0.25968181234769233],
    [2.3400481089547084e-11, 3.546262190911295e-11, 3.5230535513964624e-11, 0.9999999999059064],
    [5.124002834145484e-44, 1.7964832768555884e-43, 6.090009468261404e-43, 1.0],
    [0.008151500924936425, 0.004171192945124904, 0.9835061131848136, 0.004171192945124904],
    [4.766703531315274e-32, 4.3468397308188225e-32, 1.0, 4.94318115480553e-32],
    [0.02406074221230222, 0.039669457474255126, 0.8976574695073416, 0.03861233080610097],
    [0.41885300202065706, 0.17878445978263074, 0.15354873097971689, 0.24881380721699523],
    [0.9999951332304594, 2.039085721081367e-06, 2.039085721081367e-06, 7.885980986824679e-07],
    [0.9998993656061036, 3.186544727045929e-05, 3.8533279290971235e-05, 3.0235667334951077e-05],
    [0.9996832098223232, 0.00011413842875836803, 7.650927681974476e-05, 0.00012614247209859742],
    [0.23487211473441744, 0.2680193189100578, 0.2565951449626405, 0.24051342139288423],
    [0.27026751959684714, 0.2537733105320731, 0.2221790957899643, 0.2537800740811155],
    [0.22136656347054243, 0.28715456496066966, 0.2878866693156816, 0.20359220225310629],
    [0.20053182846768985, 0.23441037419443148, 0.3306475018137798, 0.23441029552409884],
    [0.3276715120552211, 0.08018611998082635, 0.28127994875838114, 0.3108624192055712],
    [0.30642061330495796, 0.16606498711725112, 0.25339732500106193, 0.274117074576729],
    [0.2531522609311401, 0.25305753273708836, 0.24073189234277423, 0.25305831398899736],
    [0.28503598549317605, 0.26869441188389004, 0.285015124939584, 0.16125447768334994],
    [0.2181084202904254, 0.2541631563001414, 0.32849057930069486, 0.1992378441087383],
    [0.23004721129932262, 0.32446723307124775, 0.23004725633314552, 0.21543829929628416],
    [0.3486516430651403, 0.2770206865798252, 0.025676027289894217, 0.3486516430651403]
])

Q_return = np.array([
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0]
])

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

actual_state = 20

def map_outy(action, direction):
    outy = 11

    if action == 0 and direction == 'e':
        outy = 11
    elif action == 0 and direction == 's':
        outy = 11
    elif action == 0 and direction == 'w':
        outy = 12

    elif action == 1 and direction == 'e':
        outy = 12
    elif action == 1 and direction == 'n':
        outy = 11
    elif action == 1 and direction == 'w':
        outy = 11

    elif action == 2 and direction == 'e':
        outy = 12
    elif action == 2 and direction == 'n':
        outy = 11
    elif action == 2 and direction == 's':
        outy = 12

    elif action == 3 and direction == 'n':
        outy = 12
    elif action == 3 and direction == 's':
        outy = 11
    elif action == 3 and direction == 'w':
        outy = 12

    else:
        outy = 3

    return outy

def get_state(posX: int, posY: int) -> int:
    if posX == 0 and posY == 0:
        return 0
    elif posX == 1 and posY == 0:
        return 1
    elif posX == 2 and posY == 0:
        return 2
    elif posX == 3 and posY == 0:
        return 3
    elif posX == 4 and posY == 0:
        return 4
    elif posX == 0 and posY == 1:
        return 5
    elif posX == 1 and posY == 1:
        return 6
    elif posX == 2 and posY == 1:
        return 7
    elif posX == 3 and posY == 1:
        return 8
    elif posX == 4 and posY == 1:
        return 9
    elif posX == 0 and posY == 2:
        return 10
    elif posX == 1 and posY == 2:
        return 11
    elif posX == 2 and posY == 2:
        return 12
    elif posX == 3 and posY == 2:
        return 13
    elif posX == 4 and posY == 2:
        return 14
    elif posX == 0 and posY == 3:
        return 15
    elif posX == 1 and posY == 3:
        return 16
    elif posX == 2 and posY == 3:
        return 17
    elif posX == 3 and posY == 3:
        return 18
    elif posX == 4 and posY == 3:
        return 19
    elif posX == 0 and posY == 4:
        return 20
    elif posX == 1 and posY == 4:
        return 21
    elif posX == 2 and posY == 4:
        return 22
    elif posX == 3 and posY == 4:
        return 23
    
    return 24

# it is the agent intelligence
def infer(vecInpSens: np.int32, carryRWD: int) -> int:
    global actual_state
    global direction
    outy = -1

    print('infer: ', len(vecInpSens), ' ', vecInpSens) # type: ignore
    
    if len(vecInpSens) == 1: # type: ignore
        if np.sum(vecInpSens) == 0:
            return outy
        else:
            enabledSensorIdx = np.where(vecInpSens[0] == 1)[0][0] # type: ignore

            if carryRWD == 0:
                # to know the direction
                if enabledSensorIdx in [6, 12]:
                    return 12
                
                if enabledSensorIdx == 0 and direction == '':
                    return 11
                
                if actual_state == 7:
                    return 0
                
                action = np.argmax(Q[actual_state])
                outy = map_outy(action, direction)

                if outy == 3:
                    actual_state = next_state_matrix[actual_state][action]
            else:
                if actual_state == 20:
                    return 1
                
                action = np.argmax(Q_return[actual_state])
                outy = map_outy(action, direction)

                if outy == 3:
                    actual_state = next_state_matrix[actual_state][action]

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
    global direction
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
            direction = DIRn

        elif DIRne in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRne)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_dir_' + DIRne
            direction = DIRne

        elif DIRe in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRe)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_dir_' + DIRe
            direction = DIRe

        elif DIRse in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRse)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_dir_' + DIRse
            direction = DIRse

        elif DIRs in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRs)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_dir_' + DIRs
            direction = DIRs

        elif DIRsw in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRsw)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_dir_' + DIRsw
            direction = DIRsw

        elif DIRw in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRw)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_dir_' + DIRw
            direction = DIRw

        elif DIRnw in jrasc:
            idx_inp_sns = InpSensors.index('inp_dir_' + DIRnw)
            CurrSensBits[idx_inp_sns] |= 0b1
            str_code = 'inp_dir_' + DIRnw
            direction = DIRnw

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

        if len(jrasc) != 3:
            print('Atenção: POSIÇÃO recebida - indefinida - ?!')
            str_code = 'posição_indefinida'
            stt_mm = Stt.ERRORS

        else:
            posX = jrasc[0]
            posY = jrasc[1]
            orientation = jrasc[2]
            str_code = keyMwpPOS
            CurrSensBits[idx_inp_sns] |= 0b1
    
    return stt_mm, str_code, idx_inp_sns, CurrSensBits # type: ignore
