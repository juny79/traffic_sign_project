# 교통 표지판 분류 정확도 상승 전략

초기 기준 제출 점수는 **70.40%**, 전이 학습 후 **80.8947%**, ROI crop 후 **89.1528%**, TTA 적용 후 보고된 최신 점수는 **89.2082%**입니다.

```text
초기 기준선 Accuracy: 0.7039588281868567
전이 학습 Accuracy: 0.8089469517022961
ROI crop Accuracy: 0.8915281076801267
ROI + TTA Accuracy: 0.8920823436262866
```

다음은 새로 학습을 처음부터 시작하는 대신, 현재 최고 모델에서 작은 학습률로 백본을 단계적으로 더 미세 조정합니다. 90% 초과는 목표이지 보장 결과가 아닙니다. 검증 정확도가 떨어지는 단계는 체크포인트에서 자동으로 제외합니다.

## 1. 미세 조정(fine-tuning)

1차 학습(`base_model.trainable = False`) 후, MobileNetV3의 마지막 30~50개 레이어만 학습 가능하게 해 교통 표지판 데이터에 맞춥니다. `BatchNormalization`은 고정하고 낮은 학습률을 사용합니다.

```python code.ipynb
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

new_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.03),
    metrics=["accuracy"],
)

history_stage2 = new_model.fit(
    train_set,
    validation_data=valid_set,
    epochs=25,
    callbacks=callbacks_stage2,
    verbose=1,
)
```

**이유:** 특징 추출기의 후반부가 표지판 테두리, 숫자, 기호 및 색상 패턴을 데이터셋에 맞게 조정합니다. 미세 조정 중 학습률을 크게 두면 사전학습 특징이 망가질 수 있습니다.

## 2. 입력 해상도 비교

Notebook 초기 설정에서 96x96과 128x128을 별도 실험합니다.

```python code.ipynb
img_height, img_width = 128, 128
```

크기를 바꾼 뒤에는 커널을 재시작하고 train/validation/test generator와 모델을 모두 다시 만듭니다. GPU 메모리 부족 시 `batch_size=64` 또는 `32`로 낮춥니다.

**이유:** 고해상도 입력은 작은 숫자와 내부 표식을 더 많이 보존할 수 있습니다. 단, 검증 정확도가 실제로 개선되는지 비교해야 합니다.

## 3. ROI crop으로 배경 줄이기

`Train.csv`에는 `Roi.X1`, `Roi.Y1`, `Roi.X2`, `Roi.Y2`가 있습니다. 전체 이미지 대신 ROI 영역을 잘라 쓰면 도로·하늘 등 배경 의존도를 낮출 수 있습니다.

**스크립트 적용:** `train_improved.py`에서 `--use-roi`를 지정하면 `Train.csv`와 `Test.csv`의 ROI 좌표로 crop합니다. 같은 crop 및 resize를 학습·검증·테스트에 적용하고, 이미지 경계를 벗어나는 좌표는 clip합니다. 두 CSV 모두 `Roi.X1`, `Roi.Y1`, `Roi.X2`, `Roi.Y2` 열을 포함해야 합니다.

80.8947% 점수의 기존 모델과 직접 비교할 때는 같은 해상도, 백본, 에폭, 배치 크기를 유지하고 `--use-roi`만 추가하세요. 결과 파일이 덮어쓰이지 않도록 다른 `--output-dir`을 쓰는 것을 권장합니다. ROI crop이 배경 의존도를 낮출 수 있지만 점수가 반드시 오르는 것은 아닙니다.

## 4. 데이터 증강 강도 조정

현재 `ImageDataGenerator`의 augmentation 설정을 아래와 같이 별도 실험할 수 있습니다.

```python code.ipynb
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input,
    rotation_range=8,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.12,
    brightness_range=(0.75, 1.25),
    fill_mode="reflect",
    validation_split=0.2,
)
```

좌우 반전·상하 반전과 지나친 회전은 쓰지 않는 것을 권장합니다. 표지판의 방향이나 의미가 바뀌거나 현실적이지 않은 예제가 만들어질 수 있습니다.

**이유:** 실제 촬영에서 생기는 작은 위치·크기·밝기 변화에 일반화하도록 돕습니다. 검증 데이터에는 augmentation을 적용하지 않습니다.

## 5. 더 큰 사전학습 모델 비교

MobileNetV3Small보다 큰 `EfficientNetB0` 또는 `MobileNetV3Large`를 128x128에서 비교합니다.

```python code.ipynb
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights="imagenet",
)
```

**이유:** 더 풍부한 특징을 추출해 비슷한 표지판 클래스를 구분할 수 있습니다. GPU 메모리와 학습 시간이 늘 수 있으므로 batch size 32~64부터 시도합니다. 더 큰 모델이 반드시 더 정확한 것은 아닙니다.

## 6. 클래스 불균형

현재 `ImageDataGenerator`에서는 `class_weight` 전달 시 `(image, label, sample_weight)` 구조 오류가 발생했으므로 우선 제외합니다. 적용하려면 `tf.data.Dataset` 또는 사용자 정의 Sequence가 각 샘플의 가중치도 반환하도록 구현해야 합니다.

**이유:** 적은 샘플을 가진 클래스가 학습에서 무시되는 현상을 줄일 수 있습니다. 다만 현재는 미세 조정·ROI·입력 크기 실험을 먼저 권장합니다.

## 7. 제출 매핑과 순서 검증

테스트 generator에서는 `shuffle=False`여야 하며, 예측 전에 reset합니다.

```python code.ipynb
test_set.reset()
predictions = new_model.predict(test_set, verbose=1)
print("예측 수:", len(predictions))
print("테스트 CSV 행 수:", len(test_metadata))
```

두 수가 같아야 합니다. 문자열 클래스 순서를 모델 출력 인덱스에서 원래 ClassId로 변환합니다.

```python code.ipynb
inverse_label_map = {index: name for name, index in test_set.class_indices.items()}
predicted_indices = np.argmax(predictions, axis=1)
predicted_labels = [int(inverse_label_map[int(i)]) for i in predicted_indices]
```

**이유:** 모델 성능과 무관하게 라벨 인덱스나 행 순서가 잘못되면 리더보드 점수가 크게 떨어집니다.

## 8. TTA

`train_improved.py`의 `--tta` 옵션은 원본 이미지와 밝기를 5% 높이거나 낮춘 이미지의 예측 확률을 평균합니다. `--model-path`에 기존 학습 모델을 지정하면 재학습 없이 검증과 추론만 수행합니다. 먼저 검증셋에서 원본 추론과 TTA 정확도를 비교하고, **TTA 정확도가 더 높을 때만** 테스트 제출에 TTA 결과를 사용합니다. 결과는 `inference_strategy.json`에 기록됩니다.

```bash
python train_improved.py --data-dir ./data --model-path ./outputs_roi/best_model.keras --image-size 128 --backbone MobileNetV3Small --use-roi --tta --output-dir ./outputs_roi_tta --batch-size 64
```

`./outputs_roi/best_model.keras`는 실제 89.15% 제출에 사용한 모델 경로로 바꾸세요. ROI crop과 입력 해상도도 해당 모델 학습 때와 동일해야 합니다.

**이유:** 조명 변화에 대한 예측 변동을 줄일 가능성이 있습니다. 좌우 반전과 과도한 기하학적 변형은 표지판 의미를 바꿀 수 있어 적용하지 않습니다. TTA는 추론 시간이 늘어나며 검증에서 개선되지 않으면 자동으로 테스트에 사용하지 않습니다.

## 9. 현재 최고 모델에서 단계적 미세 조정

`--resume-model`은 기존 제출 모델의 가중치부터 이어서 학습합니다. 1단계에서는 마지막 40개 백본 레이어를 `2e-6`, 2단계에서는 BatchNormalization을 고정한 채 전체 백본을 더 낮은 `5e-7` 학습률로 미세 조정합니다. 단계별 Early Stopping과 검증 정확도 기준 체크포인트를 사용하므로 기존 최고 검증 모델보다 나빠진 가중치가 제출에 선택되지 않습니다.

```bash
python train_improved.py --data-dir ./data --resume-model ./outputs_roi/best_model.keras --image-size 128 --backbone MobileNetV3Small --use-roi --fine-tune-epochs 8 --deep-fine-tune-epochs 5 --output-dir ./outputs_ft1 --batch-size 64 --tta
```

`--resume-model` 경로는 실제 89.2082% 제출에 사용한 모델로 바꾸고, 해당 모델과 같은 입력 해상도 및 ROI 전처리를 지정하세요. `--output-dir`은 원본 모델을 보호하도록 반드시 별도 경로를 사용합니다. 우선 8회/5회 정도의 짧은 추가 미세 조정으로 비교하고, GPU 메모리 부족 시 batch size를 32로 낮추세요. 검증 정확도가 향상되지 않으면 새 체크포인트가 선택되지 않으므로 기존 모델 기준 성능을 유지합니다. 다만 리더보드 분포와 검증 분포가 다를 수 있어 점수 상승을 보장하지 않습니다.

## 권장 실험 순서

| 순서 | 변경 | 확인할 점 |
|---|---|---|
| A | 현재 모델 기준선 재실행 | 검증 정확도와 리더보드 점수 |
| B | 후반 30~50개 레이어 미세 조정 | 검증 loss/accuracy 추세 |
| C | 입력 128x128 | 검증 성능 및 GPU 메모리 |
| D | ROI crop (`train_improved.py --use-roi`) | 검증 점수와 리더보드 점수 비교 |
| E | EfficientNetB0/MobileNetV3Large | 성능 대비 학습 비용 |
| F | TTA (`train_improved.py --tta`) | 검증에서 원본보다 개선되는지 |
| G | 기존 모델 단계적 미세 조정 (`--resume-model`) | 각 단계 검증 최고 체크포인트 보존 |

한 실험에서 한 종류의 요소만 변경하고, 입력 크기·모델·증강·학습 횟수·최고 검증 정확도·리더보드 점수를 기록하세요. 검증 정확도와 리더보드 점수가 함께 오르면 개선 가능성이 높습니다. 검증은 올랐지만 리더보드가 떨어지면 데이터 분포, 전처리, 클래스 매핑 및 제출 순서를 점검하세요.
