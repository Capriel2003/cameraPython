from flask import Flask, request, jsonify
import cv2 as cv
import numpy as np

app = Flask(__name__)

def extrairMaiorCtn(img):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgTh = cv.adaptiveThreshold(imgGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 12)
    kernel = np.ones((2, 2), np.uint8)
    imgDil = cv.dilate(imgTh, kernel)
    contours, _ = cv.findContours(imgDil, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    maiorCtn = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(maiorCtn)
    bbox = [x, y, w, h]
    recorte = img[y:y+h, x:x+w]
    recorte = cv.resize(recorte, (400, 750))

    return recorte, bbox

@app.route('/upload', methods=['POST'])
def upload_file():
    # Receber a imagem enviada via POST
    file = request.files['image']
    npimg = np.fromfile(file, np.uint8)
    img = cv.imdecode(npimg, cv.IMREAD_COLOR)
    
    # Processamento da imagem
    recorte, bbox = extrairMaiorCtn(img)
    
    # Desenhar contorno e retângulo na imagem original
    imgContours = img.copy()
    cv.rectangle(imgContours, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 3)

    # Converter imagem para JPEG para resposta
    _, img_encoded = cv.imencode('.jpg', imgContours)
    
    # Retornar a imagem processada
    return jsonify({
        "status": "Imagem processada com sucesso!",
        "bounding_box": bbox
    })

if __name__ == '__main__':
    app.run(debug=True)
