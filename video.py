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


def analyzeVideo(type, i):
    # Создаю объект для вычитания смеси гауссиан
    # Со статичным фоном лучше работает MOG2, с динамичным — MOG
    #
    # MOG для моделирования каждого фонового пикселя смесью K гауссовых распределений (K = от 3 до 5). 
    # Веса смеси представляют временные пропорции, в которых эти цвета остаются в сцене. 
    # Возможные цвета фона - те, которые остаются дольше и более статичными.
    #
    # MOG2 берет соответствующее количество гауссовых распределений для каждого пикселя.
    # Это обеспечивает лучшую адаптацию к изменениям освещения и т.д.
    if type == 'static':
        fgbg = cv2.createBackgroundSubtractorMOG2()
    else:
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    # Считаю размер кадра
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_area = int(width * height)

    while(cap.isOpened()):
        ret, input = cap.read()
        
        mask = fgbg.apply(input, 0.05)
        # Число белых пикселей в маске
        mask_area = np.sum(mask == 255)

        rate = mask_area / frame_area * 100

        # Отображаю процент белых пикселей
        renderRate(input, '{0:.2f}%'.format(rate))

        # Сравниваю площадь маски с общим числом пикселей. Если их отношение больше определенного значения, вывожу тревогу
        if type == 'static':
            rate = 0.005
        else: 
            rate = 0.015

        if (i == 'webcam'):
            rate = 0.06

        if (mask_area / frame_area > rate):
          renderAlarm(input)
        
        # Ресайзю, двигаю и вывожу видео
        mask, input = cv2.resize(mask, (720, 414)), cv2.resize(input, (720, 414))
        mask_name, input_name = "Mask", "Input"
        cv2.namedWindow(mask_name)
        cv2.moveWindow(mask_name, 680, 100)
        cv2.imshow(mask_name, mask)
        cv2.imshow(input_name, input)

        # Закрываю окна по нажатию на Escape
        k = cv2.waitKey(30) & 0xff
        if k == 13:
            break




if __name__ == '__main__':
    # Видеопоток из файла: python3 video.py -i file -t static -f [имя файла]
    # Видеопоток с вебкамеры: python3 video.py --input webcam --type static
    parser = createParser()
    namespace = parser.parse_args()

    if namespace.input == 'webcam':
        print("input: webcam")
        cap = cv2.VideoCapture(0)
    if namespace.input == 'file':
        print("input: file {}".format(namespace.file))
        cap = cv2.VideoCapture(namespace.file)

    analyzeVideo(namespace.type, namespace.input)

    # Освобождаю программные и аппаратные ресурсы
    cap.release()
    # Закрываю окна
    cv2.destroyAllWindows()
