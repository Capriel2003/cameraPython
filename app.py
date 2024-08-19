from flask import Flask, Response
import cv2 as cv
import numpy as np

app = Flask(__name__)

def extrairMaiorCtn(img):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgTh = cv.adaptiveThreshold(imgGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 12)
    kernel = np.ones((2,2), np.uint8)
    imgDil = cv.dilate(imgTh, kernel)
    contours, _ = cv.findContours(imgDil, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    if contours:
        maiorCtn = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(maiorCtn)
        bbox = [x, y, w, h]
        recorte = img[y:y+h, x:x+w]
        recorte = cv.resize(recorte, (400, 750))
    else:
        bbox = [0, 0, 0, 0]
        recorte = img

    return recorte, bbox

def generate_frames():
    video = cv.VideoCapture(0)  # Captura a imagem da câmera
    while True:
        success, imagem = video.read()
        if not success:
            break
        
        imagem = cv.resize(imagem, (600, 700))
        imgContours = imagem.copy()
        gabarito, bbox = extrairMaiorCtn(imagem)

        imgGray2 = cv.cvtColor(gabarito, cv.COLOR_BGR2GRAY)
        imgGray = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)
        imgBlur = cv.GaussianBlur(imgGray, (5,5), 1)
        imgCanny = cv.Canny(imgBlur, 10, 50)

        ret, imgTh = cv.threshold(imgGray2, 70, 255, cv.THRESH_BINARY_INV)
        contours, _ = cv.findContours(imgCanny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cv.drawContours(imgContours, contours, -1, (0, 255, 0), 2)
        cv.rectangle(imagem, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 3)

        # Codifica a imagem em formato JPEG e a converte para bytes
        ret, buffer = cv.imencode('.jpg', imagem)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    video.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
