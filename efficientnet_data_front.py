import os
import shutil
import glob
import shutil
import common_util
import cv2
import numpy as np

def make_folders(labels_path, images_path):
    common_util.check_folder(labels_path) #Labels 폴더가 없다면 생성한다.
    common_util.check_folder(images_path) #images 폴더가 없다면 생성한다.

def make_dir(type):
    if type == 'train':
        listdir = glob.glob(f'{TRAIN_LABEL_FOLDER}/**/*.json', recursive=True) # 라벨 폴더 내 모든 파일들을 가져온다.
        common_util.check_folder(f'{BASE_DIR}/train') #train 폴더가 없다면 생성한다.
        a,b = 'TL','TS'
    elif type == 'validation':
        listdir = glob.glob(f'{VALIDATION_LABEL_FOLDER}/**/*.json', recursive=True) # 라벨 폴더 내 모든 파일들을 가져온다.
        common_util.check_folder(f'{BASE_DIR}/validation') #validation 폴더가 없다면 생성한다.
        a,b = 'VL','VS'

    for json_annotation in listdir:
        img_dir = json_annotation.replace('라벨링데이터', '원천데이터').replace(a,b).replace('.json','.jpg')
        
        annotation_dict = common_util.load_json(json_annotation)            #json 파일을 불러온다
        car_parts = annotation_dict["learningDataInfo"]["objects"]          #json 파일 내 모든 parts들을 불러온다.
        temp_list = [parts['classId'] for parts in car_parts]               #parts들을 temp_list에 담는다.
        if temp_list.count('P10.헤드램프') == 2:    
            if 'P00.차량전체' in temp_list: ### 차량 전체 사진일때는 차량부분만 잘라서 저장
                # 아래에서 opencv로 사진을 자르기 전에 미리 불러놓는 작업
                path_arr = np.fromfile(img_dir, np.uint8)
                img = cv2.imdecode(path_arr, cv2.IMREAD_COLOR)
                for part in car_parts:
                    if part['classId'] == 'P00.차량전체':
                        # 각 parts의 좌표값을 가져온다.
                        xmin = int(part['coords']['tl']['x'])
                        ymin = int(part['coords']['tl']['y'])
                        xmax = int(part['coords']['br']['x'])
                        ymax = int(part['coords']['br']['y'])
                        # 가져온 좌표값대로 사진을 자른다.
                        output = img[ymin:ymax, xmin:xmax]
                        # 사진을 저장한다.
                        extension = os.path.splitext(f'{os.path.basename(img_dir)}')[1] # 이미지 확장자
                        result, encoded_img = cv2.imencode(extension, output)
                        if result:
                            save_dir = get_save_dir(type, img_dir)
                            common_util.check_folder(save_dir)
                            with open(f'{save_dir}/{os.path.basename(img_dir)}', mode='w+b') as f:
                                encoded_img.tofile(f)

            else:   ### 차량 전체 사진이 아닐때는 전체 이미지 파일 그냥 복사
                # 받아온 json_annotation 주소에 해당하는 image 파일을 저장.
                # 사진을 저장할 경로를 지정한다.
                save_dir = get_save_dir(type, img_dir)
                common_util.check_folder(save_dir)
                shutil.copy(img_dir, save_dir)
        else:
            continue

def get_save_dir(type, img_dir):
    brand = img_dir.split('\\')[9].split('_')[1]
    name = img_dir.split('\\')[10].split('_')[1]
    year = img_dir.split('\\')[11].split('_')[0]
    save_dir = f'{BASE_DIR}/{type}/{brand}_{name}_{year}'
    return save_dir

if __name__ == "__main__":
    
    ##### Train 데이터셋에 대해 진행
    TRAIN_LABEL_FOLDER = r"C:\Users\mok\Project2\Car_data\01.Data\1.Training\라벨링데이터\TL3\KI_기아"         #Train Label이 들어있는 folder
    VALIDATION_LABEL_FOLDER = r"C:\Users\mok\Project2\Car_data\01.Data\2.Validation\라벨링데이터\VL3\KI_기아"  #Val Label이 들어있는 folder
    BASE_DIR = r"C:\Users\mok\Project2\eff_front_kia"                                                        # Base directory 지정

    make_dir('train')
    make_dir('validation')