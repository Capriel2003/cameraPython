import cv2 as cv
import numpy as np

def extrairMaiorCtn(img):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgTh = cv.adaptiveThreshold(imgGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 12)
    kernel = np.ones((2,2), np.uint8)
    imgDil = cv.dilate(imgTh, kernel)
    contours, hi = cv.findContours(imgDil, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    maiorCtn = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(maiorCtn)
    bbox = [x, y, w, h]
    recorte = img[y:y+h, x:x+w]
    recorte = cv.resize(recorte, (400, 750))
    cv.imshow('Recorte', recorte)

    return recorte, bbox

# Defina as coordenadas e respostas diretamente no código
campos = [
    [50, 100, 30, 30],  # Exemplo de coordenadas [x, y, w, h]
    [90, 100, 30, 30],
    [130, 100, 30, 30],
    # Adicione mais campos conforme necessário
]

resp = [
    'A',  # Respostas correspondentes aos campos acima
    'B',
    'C',
    # Adicione mais respostas conforme necessário
]

respostasCorretas = ["1-B", "2-A", "3-C", "4-C", "5-A", "6-A", "7-C", "8-B", "9-D", "10-A"]

img = cv.imread("D:\\BKP_D\\Visual Studio Code\\Scanner_DC\\gab2_10_marcado.jpg")

img = cv.resize(img, (600, 700))
imgContours = img.copy()
gabarito, bbox = extrairMaiorCtn(img)
imgGray2 = cv.cvtColor(gabarito, cv.COLOR_BGR2GRAY)
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv.Canny(imgBlur, 10, 50)

ret, imgTh = cv.threshold(imgGray2, 70, 255, cv.THRESH_BINARY_INV)
contours, h = cv.findContours(imgCanny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cv.drawContours(imgContours, contours, -1, (0, 255, 0), 2)

cv.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 3)

respostas = []
for id, vg in enumerate(campos):
    x = int(vg[0])
    y = int(vg[1])
    w = int(vg[2])
    h = int(vg[3])
    cv.rectangle(gabarito, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv.rectangle(imgTh, (x, y), (x + w, y + h), (255, 255, 255), 1)
    campo = imgTh[y:y + h, x:x + w]
    height, width = campo.shape[:2]
    tamanho = height * width
    brancas = cv.countNonZero(campo)
    percentual = round((brancas / tamanho) * 100, 2)
    if percentual >= 18:
        cv.rectangle(gabarito, (x, y), (x + w, y + h), (255, 0, 0), 2)
        respostas.append(resp[id])

print("Respostas:", respostas)

erros = 0
acertos = 0
if len(respostas) == len(respostasCorretas):
    for num, res in enumerate(respostas):
        if res == respostasCorretas[num]:
            acertos += 1
        else:
            erros += 1

pontuacao = int(acertos * 10)
cv.putText(img, f'ACERTOS: {acertos}, PORCENTAGEM: {pontuacao}%', (30, 140), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

cv.imshow('img', img)
cv.imshow('Gabarito', gabarito)
cv.imshow('IMG TH', imgTh)
cv.waitKey(0)
