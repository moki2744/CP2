# -*- coding: utf-8 -*-
import cv2
import numpy as np
import glob
import os
from pathlib import Path
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

class yolov4():
    def __init__(self):
        # yolov4 model 생성
        self.net = cv2.dnn.readNet(r"C:\Users\mok\Project2\yolov4_nextlab\yolov4-car_best.weights", 
                                   r"C:\Users\mok\Project2\yolov4_nextlab\yolov4-car.cfg")
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        with open(r"C:\Users\mok\Project2\yolov4_nextlab\car.names", "r") as f: # car.names 파일로 부터 class 확인
            self.classes = [line.strip() for line in f.readlines()]
        
        # EfficientNet model 생성
        self.front_model = load_model(r"C:\Users\mok\Project2\yolov4_nextlab\eff_front_kia+hyundai.hdf5")
        self.rear_model = load_model(r"C:\Users\mok\Project2\yolov4_nextlab\eff_rear.hdf5")

        # 저장 경로 지정 함수
        self.save_dir = self.get_save_dir()

    def detect_car_image(self, directory):
        img_dirs = glob.glob(f'{directory}/**/*.jpg', recursive=True)

        # 이미지 가져오기 (한글 경로가 포함되어 있을 경우, cv2.imread() 함수가 작동하지 않기 때문에, arr 형식으로 불러와야 한다.)
        for img_dir in img_dirs:
            path_arr = np.fromfile(img_dir, np.uint8)
            img = cv2.imdecode(path_arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channels = img.shape
            
            # 한글 라벨링 표시를 위해 PIL을 사용
            img_pil = Image.fromarray(img)
            fontpath = "fonts/gulim.ttc"
            font = ImageFont.truetype(fontpath, 24)
            b,g,r,a = 0,255,0,255
            draw = ImageDraw.Draw(img_pil)

            # Detecting objects
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            # bounding box 좌표 취득
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    # Object detected
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # 좌표
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    if x < 0 or y < 0 or x+w > width or y+h > height:
                        pass
                    else:
                        label = str(self.classes[class_ids[i]])
                        temp_img = img[int(y):int(y+h), int(x):int(x+w)].copy()
                        img_preprocessed = self.preprocess_input(temp_img)

                        if label == 'car_front':
                            front_pred = self.front_model.predict(img_preprocessed)
                            car_front = self.decode_preds(front_pred, front_dict)
                            draw.rectangle((x,y,x+w,y+h), outline=(0,255,0), width=3)
                            draw.text((x, y-30) if y-30 > 0 else (x, y+30), f'{label} {car_front}', 
                                      font=font, fill=(b,g,r,a))

                        elif label == 'car_back':
                            rear_pred = self.rear_model.predict(img_preprocessed)
                            car_rear = self.decode_preds(rear_pred, rear_dict)
                            draw.rectangle((x,y,x+w,y+h), outline=(0,255,0), width=3)
                            draw.text((x, y-30) if y-30 > 0 else (x, y+30), f'{label} {car_rear}', 
                                      font=font, fill=(b,g,r,a))
                        else:
                            pass
            
            self.save_image(self.save_dir, img_dir, img_pil)

    def save_image(self, save_dir, img_dir, img_pil):
        filename = os.path.basename(img_dir)
        img_pil.save(f'{save_dir}/{filename}')

    def preprocess_input(self, img):
        img_resized = cv2.resize(img, (224,224), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        img_tensor = img_to_array(img_resized)
        img_tensor = img_tensor[np.newaxis, ...]
        img_tensor /= 255.
        return img_tensor

    def decode_preds(self, pred, car_dict):
        argmax = np.argmax(pred[0])
        return car_dict[argmax]
    
    def get_save_dir(self):
        save_dir = self.increment_path(Path(r"yolov4_nextlab\result") / 'exp', exist_ok=False)  # 결과를 반환할 exp 폴더를 생성한다 (숫자 증가)
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir
        return save_dir

    def increment_path(self, path, exist_ok=False, sep='', mkdir=False):
        path = Path(path)  # os-agnostic
        if path.exists() and not exist_ok:
            path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

            for n in range(2, 9999):
                p = f'{path}{sep}{n}{suffix}'  # increment path
                if not os.path.exists(p):  #
                    break
            path = Path(p)
        if mkdir:
            path.mkdir(parents=True, exist_ok=True)  # make directory
        return path

  


if __name__ == "__main__":
    front_dict = {0: '기아_K3_17~18', 1: '기아_K3_19~21', 2: '기아_K5_17_mx', 3: '기아_K5_17_sx', 4: '기아_K5_18-19', 5: '기아_K5_20-21', 6: '기아_K7_16-19', 7: '기아_K7_20-21', 8: '기아_K9_14-17', 9: '기아_K9_18-21', 10: '기아_니로_17-18', 11: '기아_니로_19-21', 12: '기아_레이_12-17', 13: '기아_레이_18-21', 14: '기아_모닝_17-18', 15: '기아_모닝_19', 16: '기아_모닝_20-21', 17: '기아_모하비_16-19', 18: '기아_모하비_20-21', 19: '기아_봉고3', 20: '기아_셀토스_18-21', 21: '기아_스토닉_17-21', 22: '기아_스팅어', 23: '기아_스포티지_15', 24: '기아_스포티지_16-18', 25: '기아_스포티지_19-21', 26: '기아_쏘렌토_17-20', 27: '기아_쏘렌토_21', 28: '기아_쏘울_17-18', 29: '기아_쏘울_19', 30: '기아_카니발_17-18', 31: '기아_카니발_19-20', 32: '기아_카니발_21', 33: '현대_i30_17-19', 34: '현대_그랜저_18-19', 35: '현대_그랜저_20-21', 36: '현대_그랜저_~17', 37: '현대_넥쏘_20', 38: '현대_맥스크루즈_17-18', 39: '현대_베뉴_19-21', 40: '현대_벨로스터_17', 41: '현대_벨로스터_18-21', 42: '현대_스타렉스_17', 43: '현대_스타렉스_18-21', 44: '현대_싼타페_17-18', 45: '현대_싼타페_19-20', 46: '현대_싼타페_21', 47: '현대_쏘나타_17', 48: '현대_쏘나타_18-19', 49: '현대_쏘나타_20-21', 50: '현대_아반떼_17-19', 51: '현대_아반떼_19-20', 52: '현대_아반떼_20-21', 53: '현대_아이오닉_17-20', 54: '현대_아이오닉_21', 55: '현대_엑센트_17-19', 56: '현대_코나_17-20', 57: '현대_코나_21', 58: '현대_코나_ev', 59: '현대_투싼_17-19', 60: '현대_투싼_19-20', 61: '현대_투싼_21', 62: '현대_팰리세이드_19-21', 63: '현대_포터2'}

    rear_dict = {0: '현대_i30_16', 1: '현대_i30_17-19', 2: '현대_그랜저_17', 3: '현대_그랜저_18-19', 4: '현대_그랜저_20-21', 5: '현대_넥쏘_20', 6: '현대_맥스크루즈_17-18', 
                 7: '현대_베뉴_19-21', 8: '현대_벨로스터_17', 9: '현대_벨로스터_18-21', 10: '현대_스타렉스_17-21', 11: '현대_싼타페_17-18', 12: '현대_싼타페_19-20', 13: '현대_싼타페_21', 14: '현대_쏘나타_17', 15: '현대_쏘나타_18-19', 16: '현대_쏘나타_20-21', 17: '현대_아반떼_17-19', 18: '현대_아반떼_19-20', 19: '현대_아반떼_20-21', 20: '현대_아이오닉_17-20', 21: '현대_아이오닉_21', 22: '현대_엑센트_17-19', 23: '현대_엑센트_wit', 24: '현대_코나_2017-21', 25: '현대_투싼_17-19', 26: '현대_투싼_19-20', 27: '현대_투싼_21', 28: '현대_팰리세이드_19-21', 29: '현대_포터2_17-21'}
    img_directory = r"C:\Users\mok\Project2\yolov4_nextlab\car_test_sample"
    a = yolov4()
    a.detect_car_image(img_directory)