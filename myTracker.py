import cv2
import numpy as np

import torch
import time

class YoloDetector:

    def __init__(self,model_name):

        self.model = self.load_model(model_name) # Подгрузка модели
        self.classes = self.model.names # Список объектов для трекинга
        #отображение классов, которые имеются в весах
        #print(self.classes)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' #Использование видеопамяти, если представляется возможным
        print("Using Device: ", self.device)

    def load_model(self, model_name):

        if model_name:  #Если вместо None указана какая-то своя модель
            model = torch.hub.load('ultralytics/yolov5', 'custom', path = model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained = True)
        return model

    def score_frame(self, frame):

# Смена размера кадра
        self.model.to(self.device)
        downscale_factor = 2
        width = int(frame.shape[1]/downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame = cv2.resize(frame, (width,height))

        #Применение модели к кадру
        results = self.model(frame)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        return labels,cord

    def class_to_label(self, x):

        return self.classes[int(x)]

    def plot_boxes(self, results, frame, height, width, confidence=0.3):

        labels,cord = results
        detections = []

        n = len(labels)
        x_shape, y_shape = width, height



        for i in range(n):
            row = cord[i]

            #Разметка бокса
            if row[4] >= confidence: #[4]?
                x1,y1,x2,y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)

                #Определение класса объекта

                if self.class_to_label(labels[i]) == 'bicycle':
                    confidence = float(row[4].item())
                    detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], row[4].item(), 'bicycle'))

                if self.class_to_label(labels[i]) == 'car':
                    confidence = float(row[4].item())
                    detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], row[4].item(), 'car'))

                if self.class_to_label(labels[i]) == 'motorcycle':
                    confidence = float(row[4].item())
                    detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], row[4].item(), 'motorcycle'))

                if self.class_to_label(labels[i]) == 'bus':
                    confidence = float(row[4].item())
                    detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], row[4].item(), 'bus'))

                if self.class_to_label(labels[i]) == 'truck':
                    confidence = float(row[4].item())
                    detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], row[4].item(), 'truck'))

                if self.class_to_label(labels[i]) == 'lorry':
                    confidence = float(row[4].item())
                    detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], row[4].item(), 'lorry'))

                if self.class_to_label(labels[i]) == 'trailer':
                    confidence = float(row[4].item())
                    detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], row[4].item(), 'trailer'))

        return frame,detections

#Выбор видео и установка размеров окна просмотра
cap = cv2.VideoCapture("D:\\Users\\Никита\\Desktop\\test4.mp4")


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#Создание объекта yolo детектора и передача модели
detector = YoloDetector(model_name=None)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from deep_sort_realtime.deepsort_tracker import DeepSort

object_tracker = DeepSort(max_age=5,
                          n_init=2,
                          nms_max_overlap=1.0,
                          nn_budget=None,
                          override_track_class=None,
                          embedder="mobilenet",
                          half=True,
                          bgr=True,
                          embedder_gpu=True,
                          embedder_model_name=None,
                          embedder_wts=None,
                          polygon=False,
                          today=None)

while cap.isOpened():
    success, img = cap.read()

    start = time.perf_counter()

    #Отправка кадра на детект
    results = detector.score_frame(img)
    #Отправка кадра на разметку обводки по координатам
    img, detections = detector.plot_boxes(results, img, height=img.shape[0], width=img.shape[1], confidence=0.5)

    #Присвоение id-шника
    tracks = object_tracker.update_tracks(detections, frame=img)

    #Получение id-шника и координат для отрисовки
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()

        bbox = ltrb
        #отрисовка бокса и id-шника
        cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255),2)
        cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

    #Подсчёт FPS
    end = time.perf_counter()
    totalTime = end - start
    fps = 1/ totalTime

    #отрисовка FPS
    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow('MyTracker', img)


    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()


cv2.destroyAllWindows()
