# 교통 표지판 이미지 분류 성능 개선 프로그램

`docs/accuracy_improvement_strategy.md`의 권장 전략 가운데 미세 조정, 해상도 비교, 데이터 증강, 사전학습 모델 비교, 제출 순서 검증을 적용했습니다.

- MobileNetV3Small ImageNet 가중치를 이용한 전이 학습
- 마지막 40개 레이어 미세 조정 후, 검증 성능이 유지되는 최고 체크포인트에서 전체 백본을 초저학습률로 한 단계 더 미세 조정
- `--resume-model`로 현재 제출 모델에서 이어 학습 가능; 각 단계 최고 검증 정확도 체크포인트 보존
- 96x96 또는 128x128 해상도 선택 가능(기본 128x128)
- MobileNetV3Small, MobileNetV3Large, EfficientNetB0 비교 가능
- 작은 회전·이동·확대/축소·밝기/명암 증강(검증에는 증강 미적용)
- Label Smoothing, 검증 정확도 기준 최적 모델 저장, Early Stopping 및 학습률 자동 감소
- 검증 리포트, 혼동 행렬, 제출 CSV 저장 및 테스트 예측 수 검증
- 학습 중 진행 메시지를 한국어로 출력

정확도 향상은 보장되지 않으므로 한 번에 하나씩 바꾸어 검증 성능과 리더보드 점수를 비교하세요. `--use-roi`로 ROI crop을 적용하고, `--tta`로 검증셋에서 원본·밝기 변형 추론을 비교할 수 있습니다. 검증 정확도가 실제로 더 높을 때에만 테스트 예측에 TTA를 사용합니다.

## 폴더 구조

기존 문서의 데이터 구조를 그대로 사용합니다.

```text
traffic_sign_project/
  train_improved.py
  data/
    Train.csv
    Test.csv
    Train/...
    Test/...
```

`Train.csv`에는 `Path`, `ClassId` 열이 필요하고, `Test.csv`에는 `Path` 열이 필요합니다. `--use-roi`를 쓸 때는 두 CSV 모두 `Roi.X1`, `Roi.Y1`, `Roi.X2`, `Roi.Y2` 열이 있어야 합니다.

## 설치 및 실행

```bash
python -m pip install tensorflow pandas numpy scikit-learn pillow
python train_improved.py --data-dir ./data --image-size 128 --backbone MobileNetV3Small --epochs 30 --fine-tune-epochs 25 --batch-size 64
```

GPU 메모리 부족 시 `--batch-size 32`로 낮추세요. `--model-path`는 기존 모델로 추론만 할 때, `--resume-model`은 기존 모델부터 이어서 미세 조정할 때 사용합니다. 기존 가중치를 덮어쓰지 않도록 새 `--output-dir`을 지정하세요. ROI 설정과 입력 크기는 기존 모델 학습 때와 동일해야 합니다.

```bash
# 기존 89%대 ROI 모델에서 낮은 학습률로 단계적 미세 조정
python train_improved.py --data-dir ./data --resume-model ./outputs_roi/best_model.keras --image-size 128 --backbone MobileNetV3Small --use-roi --fine-tune-epochs 8 --deep-fine-tune-epochs 5 --output-dir ./outputs_ft1 --batch-size 64 --tta

# 기존 모델의 추론 TTA 비교만 실행
python train_improved.py --data-dir ./data --model-path ./outputs_roi/best_model.keras --image-size 128 --backbone MobileNetV3Small --use-roi --tta --output-dir ./outputs_roi_tta --batch-size 64
```

`--deep-fine-tune-epochs 0`이면 전체 백본 추가 미세 조정을 끌 수 있습니다. 검증 성능이 좋아지지 않으면 체크포인트 저장 로직이 기존 최고 검증 모델을 유지합니다.

## 결과 파일

`outputs/`에 다음 파일이 생성됩니다.

- `best_model.keras`: 검증 정확도가 가장 높은 모델
- `submission.csv`: `Path,label` 형식의 제출 파일
- `training_log.csv`: 회차별 학습 기록
- `classification_report.json`: 클래스별 precision, recall, F1 점수
- `confusion_matrix.csv`: 43개 클래스 혼동 행렬
- `inference_strategy.json`: 원본/TTA 검증 정확도와 테스트에 TTA를 적용했는지 여부

## 기존 프로그램보다 개선된 이유

문서의 기존 모델은 `Conv2D(filters=5) -> MaxPooling -> Flatten -> Dense` 구조로 매우 작아 표지판의 색상·모양·세부 경계를 충분히 표현하기 어렵습니다. 개선 버전은 ImageNet으로 사전학습된 MobileNetV3 계열 또는 EfficientNetB0에서 특징을 추출한 뒤 표지판 분류기를 학습하고, 후반 레이어를 낮은 학습률로 미세 조정합니다. 클래스 불균형 가중치는 정확도 전략 문서의 권장 우선순위에 따라 이번 기본 실험에서는 적용하지 않았으며, 별도 실험 요소로 남겨 두었습니다. 성능은 검증 정확도로 추적하며, 검증 결과가 실제 리더보드 개선을 보장하지는 않습니다.
