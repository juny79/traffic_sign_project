# 교통 표지판 분류 정확도 상승 전략

현재 제출 점수는 약 **70.40%**입니다.

```text
Accuracy: 0.7039588281868567
```

아래 실험은 점수 상승을 보장하지 않습니다. 한 번에 하나씩 변경하고 검증 정확도와 리더보드 점수를 비교하세요.

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

**수정 위치:** 현재 `flow_from_directory`를 쓰는 03 데이터셋 구성 셀을 `Train.csv` 기반 `tf.data.Dataset` 또는 사용자 정의 `Sequence`로 교체해야 합니다. 같은 ROI crop, resize, 색상 처리 방식을 학습/검증/테스트 모두에 적용해야 합니다. ROI 좌표를 이미지 범위로 clip하는 것도 필요합니다.

**우선순위:** 코드 변경량이 크므로 미세 조정과 입력 크기 비교를 먼저 수행한 다음 적용합니다.

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

검증 정확도가 안정된 뒤 테스트 이미지의 원본 및 가벼운 변형 이미지의 예측 확률을 평균하는 Test-Time Augmentation을 검토합니다.

**이유:** 위치·밝기 변화에 대한 예측 변동을 줄일 수 있습니다. 단, 검증셋에도 동일한 평가 절차를 적용하여 실제 개선인지 확인해야 합니다.

## 권장 실험 순서

| 순서 | 변경 | 확인할 점 |
|---|---|---|
| A | 현재 모델 기준선 재실행 | 검증 정확도와 리더보드 점수 |
| B | 후반 30~50개 레이어 미세 조정 | 검증 loss/accuracy 추세 |
| C | 입력 128x128 | 검증 성능 및 GPU 메모리 |
| D | ROI crop | 학습·검증·테스트 전처리 일치 |
| E | EfficientNetB0/MobileNetV3Large | 성능 대비 학습 비용 |
| F | TTA | 검증 성능도 함께 상승하는지 |

한 실험에서 한 종류의 요소만 변경하고, 입력 크기·모델·증강·학습 횟수·최고 검증 정확도·리더보드 점수를 기록하세요. 검증 정확도와 리더보드 점수가 함께 오르면 개선 가능성이 높습니다. 검증은 올랐지만 리더보드가 떨어지면 데이터 분포, 전처리, 클래스 매핑 및 제출 순서를 점검하세요.
