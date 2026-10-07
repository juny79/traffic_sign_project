# 교통 표지판 분류 정확도 상승 전략

## 현재 최고 결과

현재 리더보드 최고는 **0.9690419635787807 (96.9042%)**입니다. EfficientNetB2(192×192, ROI, seed 2024)에 `class-weight-power=0.5`를 적용하고, 검증에서 유효성이 선택된 밝기 TTA를 사용한 제출입니다. 전 단계 및 오류 원인과 실험별 비교표는 [leaderboard_accuracy_diagnosis.md](leaderboard_accuracy_diagnosis.md)에 정리했습니다.

역사적 점수 흐름: 초기 기준 0.7039588 → 전이 학습 0.808947 → ROI 0.891528 → 초기 ROI+TTA 0.892082 → 독립 EfficientNetB2 기준 0.966350 → 클래스 가중치 0.968804 → 클래스 가중치+TTA 0.969042. 동등 가중 모델 앙상블은 0.954078로 하락했으므로 재사용하지 않습니다.

아래는 진행 과정에서 검토한 일반 전략입니다. 일부 실험 권고는 초기 저득점 모델 시점의 메모이므로, 현재 최고 모델에 대한 우선순위는 상세 진단 문서의 최종 결론을 따릅니다. 현재까지 추가로 확인된 최고점은 0.9690419635787807입니다.

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

## 6. 클래스 불균형 — 적용 및 결과

`train_improved.py`에 `--class-weight-power`를 추가했습니다. 0은 기존 학습과 같고 0.5는 역빈도 가중치를 제곱근으로 완화해 적용합니다. ROI와도 함께 쓸 수 있으며, validation은 가중하지 않습니다.

seed 2024 EfficientNetB2에서 `class-weight-power=0`은 리더보드 0.9663499604, `0.5`는 0.9688044339를 기록했습니다. 따라서 클래스 가중치는 실제로 개선에 기여했습니다. 그룹 검증 macro F1은 약 0.9711에서 0.9743으로 상승했고 일부 저표본 클래스 recall도 개선됐지만, 소수 클래스 precision trade-off가 있었습니다. 세부 분석과 한계는 leaderboard_accuracy_diagnosis.md를 참조하세요.

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

**실제 최종 적용 결과:** EfficientNetB2 class-weight-power 0.5 모델에서 원본 검증 정확도 0.9796935, TTA 검증 정확도 0.9800766이 나와 `tta_used_for_test=true`로 선택됐습니다. 사용자가 보고한 리더보드 점수는 **0.9690419635787807**, 현재 최고입니다. TTA의 검증 이득은 약 0.0383%p(검증 7,830장 중 3장)이므로 작은 변화이며, 테스트에서의 실제 이득은 리더보드 결과로만 확인됐습니다. 다른 테스트 데이터에서는 점수 상승을 보장하지 않습니다.

## 9. 현재 최고 모델에서 단계적 미세 조정

`--resume-model`은 기존 제출 모델의 가중치부터 이어서 학습합니다. 1단계에서는 마지막 40개 백본 레이어를 `2e-6`, 2단계에서는 BatchNormalization을 고정한 채 전체 백본을 더 낮은 `5e-7` 학습률로 미세 조정합니다. 단계별 Early Stopping과 검증 정확도 기준 체크포인트를 사용하므로 기존 최고 검증 모델보다 나빠진 가중치가 제출에 선택되지 않습니다.

```bash
python train_improved.py --data-dir ./data --resume-model ./outputs_roi/best_model.keras --image-size 128 --backbone MobileNetV3Small --use-roi --fine-tune-epochs 8 --deep-fine-tune-epochs 5 --output-dir ./outputs_ft1 --batch-size 64 --tta
```

`--resume-model` 경로는 실제 89.2082% 제출에 사용한 모델로 바꾸고, 해당 모델과 같은 입력 해상도 및 ROI 전처리를 지정하세요. `--output-dir`은 원본 모델을 보호하도록 반드시 별도 경로를 사용합니다. 우선 8회/5회 정도의 짧은 추가 미세 조정으로 비교하고, GPU 메모리 부족 시 batch size를 32로 낮추세요. 검증 정확도가 향상되지 않으면 새 체크포인트가 선택되지 않으므로 기존 모델 기준 성능을 유지합니다. 다만 리더보드 분포와 검증 분포가 다를 수 있어 점수 상승을 보장하지 않습니다.

## 과거 검토한 실험 방향

| 순서 | 변경 | 확인할 점 |
|---|---|---|
| A | 현재 모델 기준선 재실행 | 검증 정확도와 리더보드 점수 |
| B | 후반 30~50개 레이어 미세 조정 | 검증 loss/accuracy 추세 |
| C | 입력 128x128 | 검증 성능 및 GPU 메모리 |
| D | ROI crop (`train_improved.py --use-roi`) | 검증 점수와 리더보드 점수 비교 |
| E | EfficientNetB0/MobileNetV3Large | 성능 대비 학습 비용 |
| F | TTA (`train_improved.py --tta`) | 검증에서 원본보다 개선되는지 |
| G | 기존 모델 단계적 미세 조정 (`--resume-model`) | 각 단계 검증 최고 체크포인트 보존 |

한 실험에서 한 종류의 요소만 변경하고, 입력 크기·모델·증강·학습 횟수·최고 검증 정확도·리더보드 점수를 기록하세요. 검증 정확도와 리더보드 점수가 함께 오르면 개선 가능성이 높습니다. 검증은 올랐지만 리더보드가 떨어지면 데이터 분포, 전처리, 클래스 매핑 및 제출 순서를 점검하세요. 현재 기준으로는 0.9690419635787807 제출을 최고로 보존하며, 새 실험은 별도 요청이나 새 검증 근거가 있을 때만 진행합니다.
