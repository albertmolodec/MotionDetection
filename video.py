import numpy as np
import cv2
import argparse


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='file')
    parser.add_argument('-t', '--type', default='static')
    parser.add_argument('-f', '--file', default='video_static.mp4')

    return parser


def renderAlarm(frame):
    s_img = cv2.imread("alarm.png", -1)

    x_offset = y_offset = 50

    y1, y2 = y_offset, y_offset + s_img.shape[0]
    x1, x2 = x_offset, x_offset + s_img.shape[1]

    alpha_s = s_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        frame[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])

def renderRate(frame, rate):
    x, y, w, h = (50, 200, 120, 50)
    cv2.rectangle(frame, (x, y-30), (x + w, y - 30 + h), (255, 255, 255), cv2.FILLED)
    cv2.putText(img = frame, text = rate, org = (x, y), color = (0, 0, 255), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1)


def analyzeVideo(type):
    # Создаю объект для вычитания смеси гауссиан
    if type == 'static':
        fgbg = cv2.createBackgroundSubtractorMOG2()
    else:
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    # Считаю размер кадра
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_area = int(width * height)

    while(cap.isOpened()):
        ret, input = cap.read()
        # Маска
        mask = fgbg.apply(input)
        # Количество белых пикселей
        mask_area = np.sum(mask) / 255

        rate = mask_area / frame_area * 100

        # Отображаю процент белых пикселей
        renderRate(input, '{0:.2f}%'.format(rate))

        # Сравниваю площадь маски с общим числом пикселей. Если их отношение больше определенного значения, вывожу тревогу
        if type == 'static':
            rate = 0.005
        else: 
            rate = 0.015
        if (mask_area / frame_area > rate):
          renderAlarm(input)
        
        # Ресайзю и двигаю видео
        mask = cv2.resize(mask, (640, 360))
        input = cv2.resize(input, (640, 360))
        mask_name = "Mask"
        input_name = "Input"
        cv2.namedWindow(mask_name)
        cv2.moveWindow(mask_name, 680, 100)
        cv2.imshow(mask_name, mask)
        cv2.imshow(input_name, input)

        # Закрываю окна по нажатию на Escape
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break




if __name__ == '__main__':
    # Видеопоток из файла: python3 video.py -i file -type static -f [имя файла]
    # Видеопоток с вебкамеры: python3 video.py -i webcam
    parser = createParser()
    namespace = parser.parse_args()

    if namespace.input == 'webcam':
        print("input: webcam")
        cap = cv2.VideoCapture(0)
    if namespace.input == 'file':
        print("input: file {}".format(namespace.file))
        cap = cv2.VideoCapture(namespace.file)

    analyzeVideo(namespace.type)

    # Освобождаю программные и аппаратные ресурсы
    cap.release()
    # Закрываю окна
    cv2.destroyAllWindows()
