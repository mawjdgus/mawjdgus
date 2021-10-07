REFERENCE : https://herbwood.tistory.com/6 , https://github.com/object-detection-algorithm/R-CNN

# Pytorch로 구현한 R-CNN

전체 데이터셋이 아닌 특정 class만 추출해서 사용한다고 합니다. ( 전체를 하면 시간이 많이 걸리니까 )

R-CNN 모델은 

![image](https://user-images.githubusercontent.com/67318280/136377298-bb33a2e2-2d03-4d9d-b713-a701796c658c.png)

1. Fine tuned AlexNet
2. Linear SVM
3. Bounding box regressor

총 세가지 모델을 이용합니다. 각 모델은 데이터에 대해 positive / negative 정의를 다르게 하고 있기 때문에, 학습에 사용하는 데이터셋이 서로 상이하다고 한다.
그림에서 볼 수 있다시피 서로 다른 모델에 대한 데이터셋을 독립적으로 구축하고, 각각의 방식에 맞는 Custom Dataset을 정의합니다. 사실 모델을 학습하는 부분보다도 서로 다른 데이터셋을 구축하고 load하는 과정이 복잡합니다.

```
|-docs  
|-imgs  
|-py  
  |-data
  |-utils
     |-data
        |-create_bbox_regression_data.py          # Bounding box regressor 학습 데이터 생성
        |-create_classifier_data.py               # linear SVM 학습 데이터 생성
        |-create_finetune_data.py                 # AlexNet fine tune용 데이터 생성
        |-custom_batch_sampler.py                 # mini batch 구성 정의
        |-custom_bbox_regression_dataset.py       # Bounding box regressor custom data loader
        |-custom_classifier_dataset.py            # linear SVM custom data loader
        |-custom_finetune_dataset.py              # AlexNet fine tune custom data loader
        |-custom_hard_negative_mining_dataset.py  # hard negative mining 정의
        |-pascal_voc.py                           # PASCAL VOC 2007 데이터셋 다운로드
        |-pascal_voc_car.py                       # PASCAL VOC 2007 데이터셋에서 car에 해당하는 데이터만 추출
     |-util.py                                    # 기타 메서드 정의
  |-bbox_regression.py                            # Bounding box regressor 학습
  |-car_detector.py                               # 학습시킨 모델을 활용하여 detection
  |-finetune.py                                   # fine tune AlexNet
  |-linear_svm.py                                 # linear SVM 학습
  |-selectivesearch.py                            # Selective search 알고리즘 수행
```

구조가 상당히 복잡하기 때문에 이 글의 저자는 순서를 짜서 차례차례 살펴 보았습니다.

R-CNN 모델 설계 순서
1. PASCAL VOC 데이터셋 다운로드 및 "car" class 해당하는 데이터만 추출
2. 각 모델별 annotation 생성 및 Custom Dataset 정의
3. pre-trained된 AlexNet fine tuning
4. Linear SVM 및 Bounding box regressor 모델 학습
5. 3가지 모델을 모두 활용하여 detection 수행

## 1) PASCAL VOC 데이터셋 다운로드 및 Car class에 해당하는 데이터만 추출

- pascal_voc.py : PASCAL VOC 2007 데이터셋 다운로드
- pascal_voc_car : PASCAL VOC 2007 데이터셋에서 "car"에 해당하는 데이터(이미지, annotation)만 추출

PASCAL VOC 2007 데이터셋은 아래와 같은 구조를 가진다.
```
VOC2007
├── Annotations
├── ImageSets
├── JPEGImages
├── SegmentationClass
└── SegmentationObject
```

여기서 JPEGImages에는 모든 이미지가 jpg 형식으로 저장되어 있으며, Annotations에는 각 이미지 파일에 존재하는 class명, 이미지 크기, boundingbox크기 등이 xml 파일로 저장되어 있습니다. ImageSets 디렉터리에는 각 class이름에 해당하는 텍스트 파일이 존재합니다.
텍스트 파일에는 모든 이미지 파일명과 텍스트 파일명과 동일한 class에 해당하는지 여부(해당할 경우 1, 아닐 경우 -1)가 저장되어 있습니다.(ex: 1111 -1).

pascal_voc_cal.py에서는 ImageSets에 있는 car_trainval.txt 파일을 읽어들여 car에 해당하는 이미지와 xml 파일을 복사하여 별도의 데이터셋을 구축합니다.

## 2) 각 모델별 annotation 생성 및 Custom Dataset 정의
- selectivesearch.py : Selective search 알고리즘 수행
- create_finetune_data.py : AlexNet fine tune을 수행하기 위한 annotation 생성
- create_classifier_data.py : linmear SVM 학습을 위한 annotation 생성
- create_bbox_regression_data.py : Bounding box regressor 학습을 위한 annotation 생성

Selective search 알고리즘은 opencv에서 제공하는 메서드를 통해 수행합니다. AlexNet 모델을 fine tuning하기 위해 데이터셋의 annotation을 생성해주는 코드인 create_finetune_data.py는 아래와 같은 순서로 동작합니다.

```python
def parse_annotation_jpeg(annotation_path, jpeg_path, gs):

    img = cv2.imread(jpeg_path)

    selectivesearch.config(gs, img, strategy='q')
    rects = selectivesearch.get_rects(gs) # region proposals
    bndboxs = parse_xml(annotation_path) # ground truth boxes

    # get size of the biggest bounding box(region proposals)
    maximum_bndbox_size = 0
    for bndbox in bndboxs:
        xmin, ymin, xmax, ymax = bndbox
        bndbox_size = (ymax - ymin) * (xmax - xmin)
        if bndbox_size > maximum_bndbox_size:
            maximum_bndbox_size = bndbox_size

    # Comparing all region proposals and ground truth
    # return a list of iou results for each region proposals
    iou_list = compute_ious(rects, bndboxs)

    positive_list = list()
    negative_list = list()

    for i in range(len(iou_list)):
        xmin, ymin, xmax, ymax = rects[i]
        rect_size = (ymax - ymin) * (xmax - xmin)

        iou_score = iou_list[i]

        # When fine-tuning the pre-trained CNN model
        # positive : iou >= 0.5
        # negative : iou < 0.5
        # Only the bounding box with iou greater than 0.5 is saved
        if iou_score >= 0.5:
            positive_list.append(rects[i])

        # negative : iou < 0.5 And if it is more than 20% of the largest bounding box
        if 0 < iou_list[i] < 0.5 and rect_size > maximum_bndbox_size / 5.0:
            negative_list.append(rects[i])
        else:
            pass

    return positive_list, negative_list
```

1. 이미지에 Selective search 알고리즘을 적용하여 region proposals를 추출합니다. 그리고 해당 이미지에 대한 xml 파일을 읽어들여 ground truth box를 파악합니다.
2. region proposals와 ground truth box를 비교하여 IoU 값을 도출하고 0.5 이상인 sample은 positive_list, 0.5미만인 sample은 negative_list에 저장합니다.
3. 그리고 이미지별 region proposal에 대한 위치를 positive / negative 여부에 따라 서로다른 csv 파일에 저장합니다. 예를 들어 1111.jpg 파일에서 positive sample에 해당하는 bounding box의 좌표는 1111_1.csv 파일에, negative sample에 해당하는 bounding box는 1111_0.csv 파일에 저장합니다.

위의 과정은 **create_classifier_data.py**와 **create_bbox_regressor_data.py**에서도 비슷하게 동작하지만 positive/negative sample에 대한 정의만 다릅니다. 모델에 따른 서로 다른 양성/음성 정의는 다른 글에 써놓으심.

모델별 Custom Dataset정의
- custom_finetune_dataset.py : AlexNet fine tune하기 위한 Custom Dataset 정의
- custom_classifier_dataset.py : Linear SVM 모델을 학습시키기 위한 Custom Dataset 정의
- custom_bbox_regression_dataset.py : Bounding Box regressor 모델을 학습시키기 위한 Custom Dataset 정의
- custom_batch_sampler.py : 양성/음성 sample을 mini batch로 구성
- custom_hard_negative_mining_dataset.py : Hard negative mining을 수행하기 위한 Custom Dataset 정의

