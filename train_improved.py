"""GTSRB 교통 표지판 분류 성능 개선 학습 프로그램.

실행 예시:
    python train_improved.py --data-dir ./data --image-size 128 --epochs 30 --fine-tune-epochs 25

Train.csv, Test.csv와 이미지 폴더가 data-dir 아래에 있어야 합니다.
모든 로그와 오류 메시지는 한국어로 출력합니다.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

# TensorFlow의 불필요한 C++ 로그를 줄입니다.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
SEED = 2022
NUM_CLASSES = 43
IMAGE_SIZE = 128
LABEL_SMOOTHING = 0.03
ROI_COLUMNS = ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2")


def image_size() -> tuple[int, int]:
    """현재 실행에서 설정된 입력 해상도를 반환합니다."""
    return (IMAGE_SIZE, IMAGE_SIZE)


IMG_SIZE = image_size()


def seed_everything() -> None:
    """재현 가능한 학습을 위한 시드를 설정합니다."""
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def read_metadata(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "Train.csv")
    test = pd.read_csv(data_dir / "Test.csv")
    required = {"Path", "ClassId"}
    if not required.issubset(train.columns):
        raise ValueError("Train.csv에 Path와 ClassId 열이 필요합니다.")
    if "Path" not in test.columns:
        raise ValueError("Test.csv에 Path 열이 필요합니다.")
    train["ClassId"] = pd.to_numeric(train["ClassId"], errors="raise").astype("int32")
    if not train["ClassId"].between(0, NUM_CLASSES - 1).all():
        raise ValueError("ClassId는 0부터 42 사이여야 합니다.")
    return train, test


def validate_roi_metadata(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """ROI crop을 요청했을 때 양쪽 CSV의 ROI 좌표를 검증합니다."""
    for name, frame in (("Train.csv", train), ("Test.csv", test)):
        missing = set(ROI_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{name}에 ROI 좌표 열이 없습니다: {', '.join(sorted(missing))}")
        columns = list(ROI_COLUMNS)
        frame.loc[:, columns] = frame.loc[:, columns].apply(
            pd.to_numeric, errors="raise"
        )
        roi_values = frame.loc[:, columns].to_numpy(dtype=np.float64)
        if not np.isfinite(roi_values).all():
            raise ValueError(f"{name}의 ROI 좌표에 비어 있거나 유한하지 않은 값이 있습니다.")
        if (frame["Roi.X2"] <= frame["Roi.X1"]).any() or (
            frame["Roi.Y2"] <= frame["Roi.Y1"]
        ).any():
            raise ValueError(f"{name}에 유효하지 않은 ROI 좌표가 있습니다.")


def image_path(data_dir: Path, relative_path: str) -> str:
    return str((data_dir / relative_path).resolve())


def load_image(path: tf.Tensor, roi: tf.Tensor | None = None) -> tf.Tensor:
    data = tf.io.read_file(path)
    image = tf.io.decode_image(data, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    if roi is not None:
        roi = tf.cast(tf.round(roi), tf.int32)
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        x1 = tf.clip_by_value(roi[0], 0, width - 1)
        y1 = tf.clip_by_value(roi[1], 0, height - 1)
        x2 = tf.clip_by_value(roi[2], x1 + 1, width)
        y2 = tf.clip_by_value(roi[3], y1 + 1, height)
        image = tf.image.crop_to_bounding_box(image, y1, x1, y2 - y1, x2 - x1)
    image = tf.image.resize(image, IMG_SIZE, method="bilinear")
    # MobileNetV3/EfficientNet의 기본 전처리는 0~255 픽셀 입력을 받습니다.
    return tf.cast(image, tf.float32)


def make_dataset(
    frame: pd.DataFrame,
    data_dir: Path,
    training: bool,
    batch_size: int,
    use_roi: bool = False,
    mixup_alpha: float = 0.0,
) -> tf.data.Dataset:
    paths = [image_path(data_dir, p) for p in frame["Path"].astype(str)]
    labels = frame["ClassId"].to_numpy(dtype=np.int32)
    if use_roi:
        roi_values = frame.loc[:, list(ROI_COLUMNS)].to_numpy(dtype=np.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels, roi_values))
        ds = ds.map(
            lambda p, y, r: (load_image(p, r), tf.one_hot(y, NUM_CLASSES)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(
            lambda p, y: (load_image(p), tf.one_hot(y, NUM_CLASSES)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    if training:
        ds = ds.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if training and mixup_alpha > 0:
        alpha = tf.constant(mixup_alpha, dtype=tf.float32)

        def mixup_batch(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            batch_count = tf.shape(images)[0]
            concentration = tf.fill([batch_count], alpha)
            gamma_a = tf.random.gamma([], concentration)
            gamma_b = tf.random.gamma([], concentration)
            weights = gamma_a / (gamma_a + gamma_b + 1e-7)
            permutation = tf.random.shuffle(tf.range(batch_count))
            image_weights = weights[:, None, None, None]
            label_weights = weights[:, None]
            mixed_images = images * image_weights + tf.gather(images, permutation) * (1.0 - image_weights)
            mixed_labels = labels * label_weights + tf.gather(labels, permutation) * (1.0 - label_weights)
            return mixed_images, mixed_labels

        ds = ds.map(mixup_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def make_test_dataset(
    frame: pd.DataFrame, data_dir: Path, batch_size: int, use_roi: bool = False
) -> tf.data.Dataset:
    paths = [image_path(data_dir, p) for p in frame["Path"].astype(str)]
    if use_roi:
        roi_values = frame.loc[:, list(ROI_COLUMNS)].to_numpy(dtype=np.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, roi_values))
        ds = ds.map(
            lambda p, r: load_image(p, r), num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def predict_with_tta(model: tf.keras.Model, image_ds: tf.data.Dataset) -> np.ndarray:
    """원본과 약한 밝기 변화 두 가지의 예측 확률을 평균합니다."""
    original_probabilities = model.predict(image_ds, verbose=0)
    brighter_ds = image_ds.map(
        lambda images: tf.clip_by_value(images * 1.05, 0.0, 255.0),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    darker_ds = image_ds.map(
        lambda images: tf.clip_by_value(images * 0.95, 0.0, 255.0),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    brighter_probabilities = model.predict(brighter_ds, verbose=0)
    darker_probabilities = model.predict(darker_ds, verbose=0)
    return np.mean(
        [original_probabilities, brighter_probabilities, darker_probabilities], axis=0
    )


def make_callbacks(
    output_dir: Path, stage: str, patience: int, initial_best: float | None = None
) -> list[tf.keras.callbacks.Callback]:
    """단계별 체크포인트·조기 종료·학습률 조정 콜백을 구성합니다."""
    return [
        KoreanProgress(),
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / "best_model.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            initial_value_threshold=initial_best,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=0
        ),
        tf.keras.callbacks.CSVLogger(
            output_dir / "training_log.csv", append=(stage != "특징 추출")
        ),
    ]


class KoreanProgress(tf.keras.callbacks.Callback):
    """영문 기본 학습 출력 대신 한국어 진행 상황을 표시합니다."""

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        print(
            f"[진행] {epoch + 1}회차 | "
            f"학습 정확도 {logs.get('accuracy', 0):.2%} | "
            f"검증 정확도 {logs.get('val_accuracy', 0):.2%} | "
            f"검증 손실 {logs.get('val_loss', 0):.4f}"
        )


def build_model(backbone_name: str) -> tuple[tf.keras.Model, tf.keras.Model]:
    """ImageNet 사전학습 특징 추출기와 표지판 분류 헤드를 구성합니다."""
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = tf.keras.layers.RandomRotation(6 / 360, fill_mode="reflect", seed=SEED)(inputs)
    x = tf.keras.layers.RandomTranslation(0.06, 0.06, fill_mode="reflect", seed=SEED + 1)(x)
    x = tf.keras.layers.RandomZoom((-0.10, 0.10), fill_mode="reflect", seed=SEED + 2)(x)
    x = tf.keras.layers.RandomContrast(0.15, seed=SEED + 3)(x)
    x = tf.keras.layers.RandomBrightness(
        0.18, value_range=(0, 255), seed=SEED + 4
    )(x)
    # Keras 3 GaussianNoise expects stddev in [0, 1]; normalize temporarily so
    # 3/255 noise corresponds to roughly three intensity levels on 8-bit pixels.
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(x)
    x = tf.keras.layers.GaussianNoise(3.0 / 255.0, seed=SEED + 5)(x)
    x = tf.keras.layers.Rescaling(255.0)(x)

    # MobileNetV3의 기본 전처리 계층이 입력 픽셀값(0~255)을 처리합니다.
    backbones = {
        "MobileNetV3Small": tf.keras.applications.MobileNetV3Small,
        "MobileNetV3Large": tf.keras.applications.MobileNetV3Large,
        "EfficientNetB0": tf.keras.applications.EfficientNetB0,
        "EfficientNetB2": tf.keras.applications.EfficientNetB2,
    }
    base_model = backbones[backbone_name](
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    return model, base_model


def main() -> None:
    parser = argparse.ArgumentParser(description="교통 표지판 분류 모델 성능 개선 학습")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="data 폴더 경로")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="결과 저장 폴더")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="기존 best_model.keras 경로(지정하면 재학습을 생략하고 추론만 수행)",
    )
    parser.add_argument(
        "--resume-model",
        type=Path,
        default=None,
        help="기존 학습 모델에서 이어서 단계별 미세 조정",
    )
    parser.add_argument("--epochs", type=int, default=30, help="특징 추출기 학습 최대 회차")
    parser.add_argument("--fine-tune-epochs", type=int, default=25, help="마지막 레이어 미세 조정 최대 회차")
    parser.add_argument(
        "--deep-fine-tune-epochs",
        type=int,
        default=10,
        help="전체 백본을 초저학습률로 추가 미세 조정할 최대 회차(0이면 생략)",
    )
    parser.add_argument("--image-size", type=int, choices=(96, 128, 160, 192, 224), default=128, help="정사각형 입력 해상도")
    parser.add_argument(
        "--backbone",
        choices=("MobileNetV3Small", "MobileNetV3Large", "EfficientNetB0", "EfficientNetB2"),
        default="MobileNetV3Small",
        help="ImageNet 사전학습 특징 추출기",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="배치 크기")
    parser.add_argument(
        "--fine-tune-layers", type=int, default=80,
        help="백본 미세 조정 때 마지막으로 학습할 레이어 수",
    )
    parser.add_argument(
        "--mixup-alpha", type=float, default=0.0,
        help="학습 배치 MixUp 강도(0이면 비활성화; 보통 0.1~0.2)",
    )
    parser.add_argument(
        "--group-validation", action="store_true",
        help="파일명 촬영 시퀀스 단위로 학습/검증을 분리해 유사 프레임 누수를 줄임",
    )
    parser.add_argument(
        "--use-roi",
        action="store_true",
        help="Train.csv/Test.csv의 Roi.X1~Roi.Y2 좌표로 표지판 영역을 crop",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="검증에서 원본·밝기 변형 예측을 비교하고, 개선될 때 테스트 TTA를 사용",
    )
    args = parser.parse_args()
    if args.model_path is not None and args.resume_model is not None:
        parser.error("--model-path와 --resume-model은 동시에 사용할 수 없습니다.")
    if (
        args.epochs < 1
        or args.fine_tune_epochs < 0
        or args.deep_fine_tune_epochs < 0
        or args.batch_size < 1
        or args.fine_tune_layers < 1
        or args.mixup_alpha < 0
    ):
        parser.error(
            "epochs와 batch-size는 1 이상, 두 fine-tune-epochs 옵션은 0 이상이어야 합니다."
        )
    global IMAGE_SIZE, IMG_SIZE
    IMAGE_SIZE = args.image_size
    IMG_SIZE = (IMAGE_SIZE, IMAGE_SIZE)
    seed_everything()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[시작] 교통 표지판 분류 성능 개선 학습을 시작합니다.")
    print(f"[정보] 데이터 경로: {args.data_dir.resolve()}")
    print(f"[정보] TensorFlow: {tf.__version__}, GPU 수: {len(tf.config.list_physical_devices('GPU'))}")
    print(
        f"[정보] 입력 해상도: {IMAGE_SIZE}x{IMAGE_SIZE}, "
        f"모델: {args.backbone}, 배치 크기: {args.batch_size}"
    )
    train, test = read_metadata(args.data_dir)
    if args.use_roi:
        validate_roi_metadata(train, test)
        print("[정보] 학습·검증·테스트에 CSV ROI crop을 동일하게 적용합니다.")
    if args.group_validation:
        def recording_group(relative_path: str) -> str:
            normalized = str(relative_path).replace("\\", "/")
            path = Path(normalized)
            tokens = path.stem.split("_")
            if len(tokens) >= 3:
                return f"{path.parent.as_posix()}_{tokens[1]}"
            return str(relative_path)

        groups = train["Path"].astype(str).map(recording_group)
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        train_indices, valid_indices = next(
            splitter.split(train, train["ClassId"], groups=groups)
        )
        train_df, valid_df = train.iloc[train_indices], train.iloc[valid_indices]
        print("[검증] 촬영 시퀀스 그룹 분할을 적용했습니다.")
    else:
        train_df, valid_df = train_test_split(
            train, test_size=0.2, random_state=SEED, stratify=train["ClassId"]
        )
    print(f"[정보] 학습 이미지: {len(train_df):,}장, 검증 이미지: {len(valid_df):,}장, 테스트 이미지: {len(test):,}장")
    if args.mixup_alpha > 0:
        print(f"[증강] MixUp 사용(alpha={args.mixup_alpha:g})")

    train_ds = make_dataset(
        train_df, args.data_dir, True, args.batch_size,
        use_roi=args.use_roi, mixup_alpha=args.mixup_alpha,
    )
    valid_ds = make_dataset(
        valid_df, args.data_dir, False, args.batch_size, use_roi=args.use_roi
    )
    test_ds = make_test_dataset(
        test, args.data_dir, args.batch_size, use_roi=args.use_roi
    )

    if args.model_path is not None:
        if not args.model_path.is_file():
            parser.error(f"모델 파일을 찾을 수 없습니다: {args.model_path}")
        model = tf.keras.models.load_model(args.model_path)
        if tuple(model.input_shape[1:3]) != IMG_SIZE:
            parser.error(
                f"모델 입력 크기 {model.input_shape[1:3]}와 --image-size 설정 {IMG_SIZE}가 다릅니다."
            )
        print(f"[추론] 기존 모델을 불러와 재학습을 생략합니다: {args.model_path}")
    elif args.resume_model is not None:
        if not args.resume_model.is_file():
            parser.error(f"이어 학습할 모델 파일을 찾을 수 없습니다: {args.resume_model}")
        model = tf.keras.models.load_model(args.resume_model)
        if tuple(model.input_shape[1:3]) != IMG_SIZE:
            parser.error(
                f"모델 입력 크기 {model.input_shape[1:3]}와 --image-size 설정 {IMG_SIZE}가 다릅니다."
            )
        base_model = next(
            (layer for layer in model.layers if isinstance(layer, tf.keras.Model)), None
        )
        if base_model is None:
            parser.error("기존 모델에서 사전학습 백본을 찾지 못해 미세 조정을 진행할 수 없습니다.")
        print(f"[이어 학습] 기존 가중치에서 미세 조정을 시작합니다: {args.resume_model}")
        base_model.trainable = True
        for layer in base_model.layers[:-args.fine_tune_layers]:
            layer.trainable = False
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-6),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
            metrics=["accuracy"],
        )
        best_checkpoint = args.output_dir / "best_model.keras"
        best_validation_accuracy = model.evaluate(
            valid_ds, return_dict=True, verbose=0
        )["accuracy"]
        model.save(best_checkpoint)
        if args.fine_tune_epochs > 0:
            print("[미세 조정 1단계] 백본 후반부만 2e-6 학습률로 조정합니다.")
            model.fit(
                train_ds,
                validation_data=valid_ds,
                epochs=args.fine_tune_epochs,
                callbacks=make_callbacks(
                    args.output_dir,
                    "이어 학습 후반부",
                    patience=5,
                    initial_best=best_validation_accuracy,
                ),
                verbose=0,
            )
        if args.deep_fine_tune_epochs > 0:
            model = tf.keras.models.load_model(best_checkpoint)
            base_model = next(
                layer for layer in model.layers if isinstance(layer, tf.keras.Model)
            )
            base_model.trainable = True
            for layer in base_model.layers:
                layer.trainable = not isinstance(
                    layer, tf.keras.layers.BatchNormalization
                )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=5e-7),
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
                metrics=["accuracy"],
            )
            best_validation_accuracy = model.evaluate(
                valid_ds, return_dict=True, verbose=0
            )["accuracy"]
            print("[미세 조정 2단계] BN을 고정하고 전체 백본을 5e-7 학습률로 미세 조정합니다.")
            model.fit(
                train_ds,
                validation_data=valid_ds,
                epochs=args.deep_fine_tune_epochs,
                callbacks=make_callbacks(
                    args.output_dir,
                    "이어 학습 전체 백본",
                    patience=4,
                    initial_best=best_validation_accuracy,
                ),
                verbose=0,
            )
        model = tf.keras.models.load_model(best_checkpoint)
    else:
        model, base_model = build_model(args.backbone)
        print(f"[학습 1단계] {args.backbone} 특징 추출기를 고정하고 분류 헤드를 학습합니다.")
        model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=args.epochs,
            callbacks=make_callbacks(args.output_dir, "특징 추출", patience=8),
            verbose=0,
        )

        if args.fine_tune_epochs > 0:
            best_stage1 = model.evaluate(valid_ds, return_dict=True, verbose=0)["accuracy"]
            print(f"[학습 2단계] 마지막 {args.fine_tune_layers}개 레이어를 낮은 학습률로 미세 조정합니다.")
            base_model.trainable = True
            for layer in base_model.layers[:-args.fine_tune_layers]:
                layer.trainable = False
            # 작은 배치에서 BN 통계를 흔들지 않도록 모든 BatchNormalization을 고정합니다.
            for layer in base_model.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
                metrics=["accuracy"],
            )
            model.fit(
                train_ds,
                validation_data=valid_ds,
                epochs=args.fine_tune_epochs,
                callbacks=make_callbacks(
                    args.output_dir, "미세 조정", patience=8, initial_best=best_stage1
                ),
                verbose=0,
            )

        # 기존 단계 중 검증 정확도가 가장 높았던 체크포인트를 다음 단계의 출발점으로 삼습니다.
        best_checkpoint = args.output_dir / "best_model.keras"
        best_model = tf.keras.models.load_model(best_checkpoint)
        model.set_weights(best_model.get_weights())
        del best_model
        best_validation_accuracy = model.evaluate(
            valid_ds, return_dict=True, verbose=0
        )["accuracy"]

        if args.deep_fine_tune_epochs > 0:
            print(
                "[학습 3단계] BatchNormalization을 고정하고 전체 백본을 "
                "초저학습률(2e-6)로 추가 미세 조정합니다."
            )
            base_model.trainable = True
            for layer in base_model.layers:
                layer.trainable = not isinstance(
                    layer, tf.keras.layers.BatchNormalization
                )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=2e-6),
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
                metrics=["accuracy"],
            )
            model.fit(
                train_ds,
                validation_data=valid_ds,
                epochs=args.deep_fine_tune_epochs,
                callbacks=make_callbacks(
                    args.output_dir,
                    "전체 백본 미세 조정",
                    patience=5,
                    initial_best=best_validation_accuracy,
                ),
                verbose=0,
            )

        # 모든 단계 중 검증 정확도가 가장 높았던 모델을 평가와 제출에 사용합니다.
        model = tf.keras.models.load_model(best_checkpoint)
    result = model.evaluate(valid_ds, return_dict=True, verbose=0)
    valid_image_ds = valid_ds.map(
        lambda images, labels: images, num_parallel_calls=tf.data.AUTOTUNE
    )
    valid_probabilities = model.predict(valid_image_ds, verbose=0)
    valid_labels = valid_df["ClassId"].to_numpy(dtype=np.int32)
    baseline_valid_accuracy = float(
        np.mean(np.argmax(valid_probabilities, axis=1) == valid_labels)
    )
    use_tta = False
    selected_valid_probabilities = valid_probabilities
    tta_valid_accuracy = None
    if args.tta:
        tta_valid_probabilities = predict_with_tta(model, valid_image_ds)
        tta_valid_accuracy = float(
            np.mean(np.argmax(tta_valid_probabilities, axis=1) == valid_labels)
        )
        use_tta = tta_valid_accuracy > baseline_valid_accuracy
        if use_tta:
            selected_valid_probabilities = tta_valid_probabilities
        print(
            f"[검증 TTA] 원본 {baseline_valid_accuracy:.4%} | "
            f"TTA {tta_valid_accuracy:.4%} | "
            f"테스트 적용: {'예' if use_tta else '아니오'}"
        )
    selected_valid_accuracy = float(
        np.mean(np.argmax(selected_valid_probabilities, axis=1) == valid_labels)
    )
    print(
        f"[완료] 검증 정확도: {selected_valid_accuracy:.4%} "
        f"(원본 모델 {baseline_valid_accuracy:.4%}), "
        f"검증 손실: {result['loss']:.4f}"
    )
    valid_pred = np.argmax(selected_valid_probabilities, axis=1)
    with open(args.output_dir / "inference_strategy.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline_validation_accuracy": baseline_valid_accuracy,
                "tta_validation_accuracy": tta_valid_accuracy,
                "tta_used_for_test": use_tta,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    report = classification_report(valid_df["ClassId"], valid_pred, labels=list(range(NUM_CLASSES)), output_dict=True, zero_division=0)
    with open(args.output_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    np.savetxt(args.output_dir / "confusion_matrix.csv", confusion_matrix(valid_df["ClassId"], valid_pred, labels=list(range(NUM_CLASSES))), fmt="%d", delimiter=",")

    test_probabilities = (
        predict_with_tta(model, test_ds) if use_tta else model.predict(test_ds, verbose=0)
    )
    if len(test_probabilities) != len(test):
        raise RuntimeError(
            f"테스트 예측 수({len(test_probabilities)})와 CSV 행 수({len(test)})가 다릅니다."
        )
    test_pred = np.argmax(test_probabilities, axis=1)
    pd.DataFrame({"Path": test["Path"], "label": test_pred}).to_csv(args.output_dir / "submission.csv", index=False, encoding="utf-8-sig")
    if args.model_path is None:
        print(f"[저장] 최적 모델: {args.output_dir / 'best_model.keras'}")
    else:
        print(f"[정보] 사용한 모델: {args.model_path}")
    print(f"[저장] 제출 파일: {args.output_dir / 'submission.csv'}")
    print("[완료] 모든 학습 과정이 끝났습니다.")


if __name__ == "__main__":
    main()
