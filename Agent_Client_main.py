import socket
import time
import keyboard as kb

from Agent_Client_Cognition import *

from Agent_Client_Setup import Stt, SubStt, InfoReqSeq, sttMM, sttSUBfsm, msg, answES, \
    energy, carryRWD, iterNum, strCode, InpSensors, idxInpSensor, nofInfoRequest, cntNofReqs,\
    delaySec, keyMagREQ, REQfwd, REQrst, keyMwpPOS, OUTdie, OUTrst, OUTsuc, posX, posY

# host_name = socket.gethostname()
# host_IP = socket.gethostbyname(host_name)

# getting the right ip address
host_name, _, ips = socket.gethostbyname_ex(socket.gethostname())
host_IP = None
for ip in ips:
    if not ip.startswith("172.") and not ip.startswith("::1"):
        host_IP = ip

IPC_port = 15051
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (host_IP, IPC_port)

while msg != 'esc':
    if kb.is_pressed('esc'):
        msg = 'esc'
        break

    while sttMM == Stt.BEGIN:
        try:
            sock.connect(server_address)
            print('Conectado ao Servidor: %s >> porta: %s' % server_address)
            sttMM = Stt.RECEIVING
        except socket.timeout:
            strCode = 'tempo_limite_socket'
            sttMM = Stt.ERRORS
        except socket.error as e:
            print(e)
            strCode = 'conexao_servidor'
            sttMM = Stt.ERRORS
        break

    while sttMM == Stt.RESTARTING:
        msg = '{\"' + keyMagREQ + '\":[\"' + REQrst + '\",0]}'
        sttSUBfsm = SubStt.RES
        sttMM = Stt.SENDING
        break

    while sttMM == Stt.INTERPRETING:
        sttMM, strCode, idxInpSensor, CurrentSensBits = interpreting(answES) # type: ignore

    while sttMM == Stt.DECIDING:
        print('<< decidindo >> ', (energy - iterNum))

        if iterNum >= energy:
            print('O agente não tem mais ENERGIA!')
            strCode = 'semEnergia'
            sttMM = Stt.EXCEPTIONS
            break
        else:
            iterNum = iterNum + 1

        while sttSUBfsm == SubStt.RES:
            if InpSensors[idxInpSensor] == 'inp_' + OUTrst:
                nofIter = 0
                sttSUBfsm = SubStt.START
            else:
                strCode = 'erro => esperava reiniciado...'
                sttMM = Stt.ERRORS
            break

        while sttSUBfsm == SubStt.START:
            cntNofReqs = 0
            sensInpBits = np.zeros((nofInfoRequest, 32), dtype=np.int32)
            sttSUBfsm = SubStt.ASK
            break

        while sttSUBfsm == SubStt.ASK:
            if cntNofReqs < nofInfoRequest:
                d = str(InfoReqSeq[cntNofReqs][1])
                msg = '{\"' + keyMagREQ + '\":[\"'
                if InfoReqSeq[cntNofReqs][0] == 'fwd':
                    msg = msg + REQfwd + '\",' + d + ']}'
                elif InfoReqSeq[cntNofReqs][0] == 'r90':
                    msg = msg + REQrgt + '\",' + d + ']}'
                elif InfoReqSeq[cntNofReqs][0] == 'l90':
                    msg = msg + REQlft + '\",' + d + ']}'
                elif InfoReqSeq[cntNofReqs][0] == 'r45':
                    msg = msg + REQr45 + '\",' + d + ']}'
                elif InfoReqSeq[cntNofReqs][0] == 'l45':
                    msg = msg + REQl45 + '\",' + d + ']}'
                sttMM = Stt.SENDING
                sttSUBfsm = SubStt.WAITRQ
                break
            break

        while sttSUBfsm == SubStt.SAVE:
            sensInpBits[cntNofReqs] = CurrentSensBits
            cntNofReqs = cntNofReqs + 1
            if cntNofReqs == nofInfoRequest:
                sttSUBfsm = SubStt.CMD
            else:
                sttSUBfsm = SubStt.ASK
            break

        while sttSUBfsm == SubStt.CMD:
            print("<< cmd >>")
            cntNofReqs = 0
            print(sensInpBits)
            decision = infer(sensInpBits)
            msg = create_msg(decision, 1)
            sttMM = Stt.SENDING
            sttSUBfsm = SubStt.WAITCM
            break

        while sttSUBfsm == SubStt.CNT:
            fdbkcode = feedback_analysis(sensInpBits, carryRWD)
            if fdbkcode == -1:
                strCode = 'erro => reiniciando...'
                sttMM = Stt.ERRORS
            elif fdbkcode == 50:
                print('-> a RECOMPENSA foi coletada <-')
                carryRWD = 1
                nofIter = 0
            elif fdbkcode == 100:
                print('O Agente GANHOU - sucesso!')
                strCode = OUTsuc
                sttMM = Stt.EXCEPTIONS
                sttSUBfsm = SubStt.ASK
                break
            elif fdbkcode == -100:
                print('O Agente MORREU ')
                strCode = OUTdie
                sttMM = Stt.EXCEPTIONS
                sttSUBfsm = SubStt.ASK
                break
            msg = create_msg(fdbkcode, 0)
            sttMM = Stt.SENDING
            sttSUBfsm = SubStt.ASK
            break

        while sttSUBfsm == SubStt.WAITRQ:
            sttSUBfsm = SubStt.SAVE
            break

        while sttSUBfsm == SubStt.WAITCM:
            if delaySec > 0:
                time.sleep(delaySec)
            sttSUBfsm = SubStt.CNT
            break

    while sttMM == Stt.EXCEPTIONS:
        if strCode == OUTrst:
            msg = '{\"' + keyMagREQ + '\":[\"' + REQfwd + '\",1]}'
            sttMM = Stt.SENDING
            break

        elif strCode == keyMwpPOS:
            posX = posX
            posY = posY
            msg = '{\"' + keyMagREQ + '\":[\"' + REQfwd + '\",1]}'
            sttMM = Stt.SENDING
            break

        elif strCode == 'noEnergy':
            msg = '{\"' + keyMagREQ + '\":[\"' + REQrst + '\",0]}'
            nofIter = 0
            sttMM = Stt.SENDING
            break

        elif strCode == OUTdie:
            msg = '{\"' + keyMagREQ + '\":[\"' + REQrst + '\",0]}'
            nofIter = 0
            sttMM = Stt.SENDING
            break

        elif strCode == OUTsuc:
            msg = '{\"' + keyMagREQ + '\":[\"' + REQrst + '\",0]}'
            nofIter = 0
            carryRWD = 0

            sttMM = Stt.SENDING
            break
        sys.exit(-2)

    while sttMM == Stt.ERRORS:
        print('--> estado ERRORS::')
        print(strCode)
        sys.exit(-1)

    while sttMM == Stt.RECEIVING:
        try:
            answES = sock.recv(256)
            print('resposta_conn: %s' % answES)
            sttMM = Stt.INTERPRETING
        except socket.error as e:
            print('Erro de Socket: ', str(e))
            strCode = 'socket_error'
            sttMM = Stt.ERRORS
        break

    while sttMM == Stt.SENDING:
        print('enviando = ', msg)
        if msg != '':
            try:
                sock.sendall(msg.encode('utf-8'))
                msg = ''
                sttMM = Stt.RECEIVING
            except socket.error as e:
                print('Erro de Socket: ', str(e))
                strCode = 'socket_error'
                sttMM = Stt.ERRORS
            break
        else:
            print('Atenção: tentando enviar uma mensagem vazia')
            strCode = 'empty_msg'
            sttMM = Stt.ERRORS
            break

    while sttMM == Stt.FOR_TESTS:
        print('>> state FOR_TESTS <<')
        break

else:
    sock.close()
    print('<< END of process >>')

sys.exit(0)