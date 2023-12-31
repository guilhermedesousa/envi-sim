import socket
import time
import keyboard as kb

from Agent_Client_Cognition import *

from Agent_Client_Setup import Stt, SubStt, InfoReqSeq, sttMM, sttSUBfsm, msg, answES, \
    energy, carryRWD, iterNum, strCode, InpSensors, idxInpSensor, nofInfoRequest, cntNofReqs,\
    delaySec, keyMagREQ, REQfwd, REQrst, keyMwpPOS, OUTdie, OUTrst, OUTsuc, posX, posY

# host_name = socket.gethostname()
# host_IP = socket.gethostbyname(host_name)

# make the IPC conection with the EnviSim process
host_name, _, ips = socket.gethostbyname_ex(socket.gethostname())
host_IP = None

for ip in ips:
    if not ip.startswith("172.") and not ip.startswith("::1"):
        host_IP = ip

IPC_port = 15051 # EnviSim process port
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (host_IP, IPC_port)

while msg != 'esc':
    if kb.is_pressed('esc'):
        msg = 'esc'
        break

    # BEGIN FSM state - create a socket and try to connect to the EnviSim process
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
    
    # RESTARTING FSM state - send the restart request to the EnviSim
    while sttMM == Stt.RESTARTING:
        msg = '{\"' + keyMagREQ + '\":[\"' + REQrst + '\",0]}'
        sttSUBfsm = SubStt.RES
        sttMM = Stt.SENDING
        break
    
    # INTERPRETING FSM state - interpret the response sent by EnviSim
    while sttMM == Stt.INTERPRETING:
        sttMM, strCode, idxInpSensor, CurrentSensBits = interpreting(answES) # type: ignore

    # DECIDING FSM state - take decisions and generate: command-action-request
    while sttMM == Stt.DECIDING:
        print('<< decidindo >> ', (energy - iterNum))

        if iterNum >= energy:
            print('O agente não tem mais ENERGIA!')
            strCode = 'noEnergy'
            sttMM = Stt.EXCEPTIONS
            break
        else:
            iterNum = iterNum + 1

        # RES sub-FSM state - control how many times the agent requests infos from EnviSim
        while sttSUBfsm == SubStt.RES:
            if InpSensors[idxInpSensor] == 'inp_' + OUTrst:
                nofIter = 0
                sttSUBfsm = SubStt.START
            else:
                strCode = 'erro => esperava reiniciado...'
                sttMM = Stt.ERRORS
            break
        
        # START sub-FSM state - request information only
        while sttSUBfsm == SubStt.START:
            cntNofReqs = 0
            sensInpBits = np.zeros((nofInfoRequest, 32), dtype=np.int32)
            sttSUBfsm = SubStt.ASK
            break
        
        # ASK sub-FSM state - ask for infos through requests
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
        
        # SAVE sub-FSM state - save the response of a request
        while sttSUBfsm == SubStt.SAVE:
            sensInpBits[cntNofReqs] = CurrentSensBits # type: ignore
            cntNofReqs = cntNofReqs + 1

            if cntNofReqs == nofInfoRequest:
                sttSUBfsm = SubStt.CMD
            else:
                sttSUBfsm = SubStt.ASK
            break
        
        # CMD sub-FSM state - send a command to the EnviSim
        while sttSUBfsm == SubStt.CMD:
            cntNofReqs = 0
            
            decision = infer(sensInpBits, carryRWD) # type: ignore
            msg = create_msg(decision, 1)
            sttMM = Stt.SENDING
            sttSUBfsm = SubStt.WAITCM
            break
        
        # CNT sub-FSM state - decide if the session is ended or keep it
        while sttSUBfsm == SubStt.CNT:
            fdbkcode = feedback_analysis(CurrentSensBits, carryRWD) # type: ignore
            
            if fdbkcode == -1:
                strCode = 'erro => reiniciando...'
                sttMM = Stt.ERRORS
            elif fdbkcode == 50:
                print('-> a RECOMPENSA foi coletada <-')
                carryRWD = 1
                nofIter = 0
                msg = create_msg(0, 1)
                sttMM = Stt.SENDING
                sttSUBfsm = SubStt.ASK
                break
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
        
        # WAITRQ sub-FSM state - wait for all requests be completed
        while sttSUBfsm == SubStt.WAITRQ:
            sttSUBfsm = SubStt.SAVE
            break
        
        # WAITCM sub-FSM state - wait for the feedback response after a command
        while sttSUBfsm == SubStt.WAITCM:
            if delaySec > 0:
                time.sleep(delaySec)
            sttSUBfsm = SubStt.CNT
            break

    # EXCEPTIONS FSM state - a message has been received from EnviSim
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

    # ERROR FSM state - only for tests
    while sttMM == Stt.ERRORS:
        print('--> estado ERRORS::')
        print(strCode)
        sys.exit(-1)

    # RECEIVING FSM state - wait for a response from EnviSim
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
    
    # SENDING FSM state - send the content to the EnviSim
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

    # FOR TESTS FSM state - only for tests
    while sttMM == Stt.FOR_TESTS:
        print('>> state FOR_TESTS <<')
        break

else:
    sock.close()
    print('<< END of process >>')

sys.exit(0)