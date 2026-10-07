# 교통 표지판 이미지 분류 성능 개선 프로그램

`docs`의 기존 코드(32x32 단일 합성곱층, 5회 학습)를 기반으로 다음을 적용했습니다.

- 입력 크기를 48x48로 확대해 작은 표지판의 세부 특징 보존
- 회전, 이동, 확대·축소, 명암 데이터 증강
- 4단계 잔차 CNN, Batch Normalization, Dropout, Global Average Pooling
- 클래스별 이미지 수를 반영한 `class_weight` 적용
- Label Smoothing으로 과적합과 과도한 확신 완화
- 검증 정확도 기준 최적 모델 저장
- Early Stopping과 학습률 자동 감소
- 검증 리포트, 혼동 행렬, 제출 CSV 저장
- 학습 중 표시되는 진행 메시지를 한국어로 통일

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

`Train.csv`에는 `Path`, `ClassId` 열이 필요하고, `Test.csv`에는 `Path` 열이 필요합니다.

## 설치 및 실행

```bash
python -m pip install tensorflow pandas numpy scikit-learn pillow
python train_improved.py --data-dir ./data --epochs 50
```

처음에는 GPU 메모리와 시간을 확인하기 위해 `--epochs 2`로 실행해 볼 수 있습니다.

## 결과 파일

`outputs/`에 다음 파일이 생성됩니다.

- `best_model.keras`: 검증 정확도가 가장 높은 모델
- `submission.csv`: `Path,label` 형식의 제출 파일
- `training_log.csv`: 회차별 학습 기록
- `classification_report.json`: 클래스별 precision, recall, F1 점수
- `confusion_matrix.csv`: 43개 클래스 혼동 행렬

## 기존 프로그램보다 개선된 이유

문서의 기존 모델은 `Conv2D(filters=5) -> MaxPooling -> Flatten -> Dense` 구조로 매우 작아 표지판의 색상·모양·세부 경계를 충분히 표현하기 어렵습니다. 개선 버전은 더 깊은 특징 추출과 잔차 연결을 사용하고, 학습 데이터의 클래스 불균형을 보정합니다. 또한 검증 데이터로 최적 시점을 선택하므로 단순히 5회 학습하는 것보다 일반화 성능을 안정적으로 높일 수 있습니다.
