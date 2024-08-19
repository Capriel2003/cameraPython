from flask import Flask, request, jsonify
import cv2 as cv
import numpy as np

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    npimg = np.fromfile(file, np.uint8)
    img = cv.imdecode(npimg, cv.IMREAD_COLOR)
    
    # Processamento da imagem com OpenCV
    gabarito, bbox = extrairMaiorCtn(img)
    
    # Retorne uma resposta
    return jsonify({"status": "Imagem processada com sucesso!"})

def extrairMaiorCtn(img):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgTh = cv.adaptiveThreshold(imgGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 12)
    kernel = np.ones((2,2), np.uint8)
    imgDil = cv.dilate(imgTh, kernel)
    contours, _ = cv.findContours(imgDil, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    maiorCtn = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(maiorCtn)
    bbox = [x, y, w, h]
    recorte = img[y:y+h, x:x+w]
    recorte = cv.resize(recorte, (400, 750))

    return recorte, bbox

if __name__ == '__main__':
    app.run(debug=True)
