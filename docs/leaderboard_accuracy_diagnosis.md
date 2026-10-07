# 리더보드 정확도 진단 및 다음 실험 계획

작성일: 2026-10-07

## 요약

현재 최고 리더보드 결과는 **0.9690419636 (96.9042%)**입니다. seed 2024, `class-weight-power=0.5` EfficientNetB2 모델에 검증 선택형 밝기 TTA를 적용한 제출로, TTA 없는 같은 모델의 **0.9688044339**보다 **0.0002375297 (0.0238%p)** 상승했습니다. 기준 가중치 0 모델의 0.9663499604보다 총 **0.2692%p** 높습니다.

TTA는 validation accuracy를 0.9796935에서 0.9800766으로 높여 테스트에도 선택됐습니다 (`outputs_effb2_classweighted_tta/inference_strategy.json`). TTA 증가폭은 작고 한 그룹 검증 fold의 사례에 의존하므로, 이를 최종 결과로 기록하되 추가 점수 향상은 확정적으로 기대하지 않습니다.

동등 가중 앙상블은 기존 제출 점수를 낮췄습니다. 이번 결과에서는 클래스 가중치와 가벼운 밝기 TTA 조합이 실제 리더보드에서 가장 높았으므로, 현 시점에서는 추가 학습·앙상블보다 이 제출을 최고 기준으로 보존합니다.

## 실험 결과

| 실험 | 리더보드 Accuracy | 검증 관련 관찰 | 해석 |
|---|---:|---|---|
| EfficientNetB2 단일 모델 + ROI + TTA | 0.9617577197 | 그룹 검증 원본 0.969084, TTA 0.970865 | 이전 기준 최고 제출. 이후 seed 2024 모델에 추월됨 |
| EfficientNetB2와 기존 ROI 모델 동등 가중 앙상블 + TTA | 0.9540775930 | 원본 앙상블 0.986641, TTA 0.987277 | 리더보드에서 0.7680%p 하락. 검증 선택과 테스트 분포가 일치하지 않음 |
| EfficientNetB2 95% + 보조 모델 5% 후보 | 0.9617577197 | 이후 선택된 검증 분할에서 단일 모델 0.995019, 앙상블 미선택, TTA 미사용 | 기존 예측과 동일. 개선을 입증하지 못함 |
| EfficientNetB2 기준 모델, seed 2024, class-weight-power 0 | 0.9663499604 | 검증 정확도 0.979310 | class-weight paired control |
| EfficientNetB2, seed 2024, class-weight-power 0.5 | 0.9688044339 | 원본 검증 정확도 0.979693 | TTA 없는 기준. power 0보다 0.2454%p 상승 |
| EfficientNetB2 + class-weight-power 0.5 + 밝기 TTA | **0.9690419636** | 검증 원본 0.9796935, TTA 0.9800766, 테스트 TTA 적용 | 현재 최고. 가중 모델보다 0.0238%p 상승 |

※ 0.995019는 이전 fold로 학습한 체크포인트를 다른 fold에서 평가한 값이라 누수 가능성이 있어 독립 검증 점수로 해석하지 않습니다.

## 원인 진단

### 1. 같은 가중치 앙상블은 강한 주 모델의 오답을 보정하지 못함

확률 평균이 좋아지려면 보조 모델이 주 모델과 다른 사례에서 맞혀야 합니다. 이전 ROI 모델의 오류가 독립적이지 않거나 확률 분포가 EfficientNetB2와 다르면, 50%씩 섞는 것은 B2의 올바른 확신을 희석하고 보조 모델의 오답을 끌어들일 수 있습니다. 실제 리더보드 결과가 하락했으므로, 동등 가중 앙상블은 폐기하는 것이 타당합니다.

### 2. 처음 선택된 그룹 검증 분할에 ClassId 24가 없었음

동등 가중 앙상블의 검증 리포트에서 클래스 24의 support가 0이었습니다. 따라서 그 검증 정확도에는 클래스 24를 맞히는 능력이 반영되지 않았습니다. 98%대 검증값은 전체 테스트 클래스 분포를 보증하지 않습니다. 코드에서는 현재 다섯 그룹 분할 중 클래스 누락과 클래스별 분포 편차가 작은 분할을 선택하고, 누락이 남으면 경고하도록 바꿨습니다.

### 3. 검증 분할을 바꾼 뒤 기존 체크포인트를 다시 평가하면 데이터 누수가 생김

기존 EfficientNetB2는 이전 그룹 분할에서 학습된 모델입니다. 검증 분할 선택 로직을 바꾼 뒤 그 체크포인트를 새 검증셋에서 평가하면 새 검증 이미지 일부가 이미 학습에 사용되었을 수 있습니다. 이 경우 0.995 같은 매우 높은 검증값이 나와도 일반화 성능을 뜻하지 않습니다. 새 분할로 모델을 비교하려면 **그 분할에서 처음부터 학습한 모델**을 사용해야 합니다.

### 4. TTA의 검증 이득은 작지만 이번 제출에서는 리더보드 개선 확인

이번 class-weighted 모델의 TTA 향상은 검증에서 약 0.0383%p(7,830장 중 3장)로 작았습니다. 다만 제출 결과는 원본 class-weighted 모델보다 0.0238%p 높은 0.9690419635787807이었습니다. 한 validation fold의 작은 차이로 선택된 TTA이므로 후속 데이터에서 항상 이득을 보장한다고 해석하지 않고, 이번 제출에 대해 확인된 결과로 기록합니다.

## 코드 변경 사항

`train_improved.py`에 다음 진단·실험 기능을 반영했습니다.

- 여러 그룹 검증 후보 중 클래스 누락을 줄이는 분할 선택 및 누락 클래스 경고
- 보조 모델을 주 모델과 다른 해상도로 추론할 수 있도록 입력 크기별 resize
- `--ensemble-primary-weight`로 기준 모델의 가중치를 조절 (기본 0.9)
- 검증에서 단일 모델이 앙상블보다 좋으면 단일 모델을 선택
- `inference_strategy.json`에 앙상블 사용 여부와 가중치 기록
- `--class-weight-power`로 훈련 분할 내 클래스 빈도에 기반한 완만한 표본 가중 가능

이 기능들은 비교 도구이지 리더보드 상승 보장은 아닙니다. 특히 이전 모델과 새 분할을 결합한 검증값은 학습/검증 누수 여부를 먼저 확인해야 합니다.

## 적용한 개선: 완만한 클래스 가중 학습

class-weight-power 0.5 결과에서 검증 정확도는 0.979310에서 0.979693으로 약 0.0383%p(7,830장 중 3장) 올랐고 macro F1은 0.9711에서 0.9743으로 개선됐습니다. class 19 recall은 0.767→0.867, class 29는 0.833→0.867, class 39는 0.883→0.950으로 개선됐습니다. class 37 recall은 1.0을 유지했지만 precision은 0.625→0.566으로 낮아져 trade-off도 있습니다. 리더보드 점수도 0.2454%p 상승했으므로 가중 모델을 현재 기준으로 둡니다.

`train_improved.py`에 `--class-weight-power` 옵션을 추가했습니다. 값 0은 기존 학습과 동일하고, 0.5는 역빈도 가중치를 제곱근으로 완화해 적용합니다. 훈련 분할의 클래스 빈도만으로 가중치를 계산하고 검증 정확도는 가중하지 않습니다.

실험은 같은 seed와 선택된 그룹 fold로 진행했습니다. 재현할 때의 학습 명령은 다음과 같습니다.

```bash train_improved.py
python train_improved.py --data-dir ./data --output-dir ./outputs_effb2_seed2024_control --image-size 192 --backbone EfficientNetB2 --epochs 30 --fine-tune-epochs 25 --deep-fine-tune-epochs 10 --fine-tune-layers 80 --batch-size 16 --group-validation --use-roi --seed 2024 --class-weight-power 0
python train_improved.py --data-dir ./data --output-dir ./outputs_effb2_classweighted --image-size 192 --backbone EfficientNetB2 --epochs 30 --fine-tune-epochs 25 --deep-fine-tune-epochs 10 --fine-tune-layers 80 --batch-size 16 --group-validation --use-roi --seed 2024 --class-weight-power 0.5
```

기준 대비 검증 정확도 차이는 작았지만 macro F1과 일부 저표본 클래스 recall이 개선됐고, 리더보드도 0.2454%p 상승했습니다. 이후 가중 모델에 TTA를 적용했고, 검증에서 TTA가 선택된 뒤 제출 점수가 다시 상승했습니다.

## 최종 적용: 검증 선택형 TTA

가중 단일 모델의 검증 원본 정확도는 0.9796934866이었습니다. 원본과 밝기 ±5% 변형 예측 평균은 0.9800766284를 기록해 약 0.0383%p(검증 7,830장 중 3장) 올랐고, `tta_used_for_test: true`로 테스트에도 적용됐습니다. 별도 결과는 `outputs_effb2_classweighted_tta/`에 저장되어 있습니다.

```bash
python train_improved.py --data-dir ./data --output-dir ./outputs_effb2_classweighted_tta --model-path ./outputs_effb2_classweighted/best_model.keras --image-size 192 --backbone EfficientNetB2 --batch-size 16 --group-validation --use-roi --tta
```

`outputs_effb2_classweighted_tta/inference_strategy.json`은 `tta_used_for_test: true`를 기록합니다. 사용자가 제출한 해당 결과의 리더보드 점수는 **0.9690419635787807**입니다. 검증 증가가 몇 장 수준이라 다른 분할에서도 재현된다고 단정할 수는 없지만, 현재 제출에서 확인된 최고 점수이므로 이를 최고 모델·제출로 보존합니다.

## 결과 관리 및 재현성

1. `outputs_effb2_classweighted_tta/submission.csv`와 함께 리더보드 **0.9690419635787807**을 최고 결과로 기록합니다.
2. 원본 class-weighted 결과 `outputs_effb2_classweighted/submission.csv` (0.9688044339)와 기준 `outputs_effb2_seed2024_control` (0.9663499604)을 비교 기준으로 보존합니다.
3. 이전 동등 가중 앙상블 결과 0.9540775930은 성능 하락 사례로 보존하되, 같은 조합은 재사용하지 않습니다.
4. 최종 제출을 교체하기 전에는 제출 행 수, Test.csv 행 순서, Path 매핑과 ClassId 0~42 인덱스를 확인합니다.
5. 추가 실험은 새 검증 근거나 명확한 가설이 있을 때 별도 output-dir로만 진행합니다. 현재 최고 산출물은 덮어쓰지 않습니다.

## 결론

현재 최고는 seed 2024 EfficientNetB2, `class-weight-power=0.5`, 밝기 TTA 조합의 **0.9690419635787807 (96.9042%)**입니다. 기준 모델부터 클래스 가중치 실험, TTA까지의 변경과 점수를 이 문서에 기록했습니다. 지금은 이 제출을 최종 최고로 보존하고, 추가 실험은 별도 요청이나 새 검증 근거가 있을 때만 진행합니다.
