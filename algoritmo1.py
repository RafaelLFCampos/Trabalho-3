import cv2
import numpy as np

#carregar o vídeo
captura = cv2.VideoCapture('video.MTS')
if not captura.isOpened:
    print("Não foi possível abrir o vídeo")
    exit(0)

#pegar um frame do vídeo 
ret, plano = captura.read()
#converter para escalas de cinza
plano = cv2.cvtColor(plano, cv2.COLOR_BGR2GRAY)
#Aplicando filtro gaussiano
plano_blur = cv2.GaussianBlur(plano,(7,7),0)
#kernel utilizado na etapa de erosão
kernel = np.ones((5, 5), np.uint8)

while True:
    #mesmos passos do frame capturado acima
    ret, frames = captura.read()
    if frames is None:
        break
    
    #conversão para escala de cinza
    frame = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    #aplicação do filtro gaussiano
    blur = cv2.GaussianBlur(frame,(7,7),0)

    #subtração do plano de fundo com o frame atual para se obter apenas os elementos que não são do plano de fundo
    fr = blur - plano_blur

    #aplicação do threshold binário com OTSU
    ret,th = cv2.threshold(fr,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #aplicação da erosão para diminuir ruídos
    eros = cv2.erode(th, kernel, iterations = 2)
    #aplicação da máscara binária obtida no frame do vídeo
    result = cv2.bitwise_and(frames, frames, mask = eros)

    #mostrar contador na tela
    cv2.rectangle(result, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(result, str(captura.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    #exibição de cada frame
    cv2.imshow('Plano de Fundo Fixo', result)

    #utilizado para encerrar a execução do vídeo
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
 
captura.release()
