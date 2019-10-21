import cv2 as cv2
import numpy as np

#carregar o vídeo
captura = cv2.VideoCapture('video.MTS')
if not captura.isOpened:
    print("Não foi possível abrir o vídeo")
    exit(0)
    
#kernel utilizado na etapa de erosão
kernel = np.ones((5, 5), np.uint8)
#Usado para gerar a máscara de foreground
backSub = cv2.createBackgroundSubtractorKNN()

while True:
    #pegar um frame do vídeo 
    ret, frame = captura.read()
    if frame is None:
        break
    
    #Aplicando filtro gaussiano
    blur = cv2.GaussianBlur(frame,(7,7),0)
    #usado para atualizar o plano de fundo
    fgMask = backSub.apply(blur, 1)

    #aplicação do threshold binário com OTSU
    ret,th = cv2.threshold(fgMask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #aplicação da erosão para diminuir ruídos
    eros = cv2.erode(th, kernel, iterations = 1)
    ##aplicação da dilatção para preencher espaços causados por ruídos
    dil = cv2.dilate(eros, kernel, iterations = 1)
    #aplicação da máscara binária obtida no frame do vídeo
    result = cv2.bitwise_and(frame, frame, mask = dil)

    #mostrar contador na tela
    cv2.rectangle(result, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(result, str(captura.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    #exibição de cada frame
    #cv2.imshow('Plano de Fundo Adaptativo', result)
    cv2.imshow('Plano de Fundo Adaptativo', result)
    #utilizado para encerrar a execução do vídeo
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

captura.release()
