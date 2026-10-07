"""GTSRB 교통 표지판 분류 성능 개선 학습 프로그램.

실행 예시:
    python train_improved.py --data-dir ./data --epochs 50

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
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

SEED = 2022
IMG_SIZE = (48, 48)
NUM_CLASSES = 43


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


def image_path(data_dir: Path, relative_path: str) -> str:
    return str((data_dir / relative_path).resolve())


def load_image(path: tf.Tensor) -> tf.Tensor:
    data = tf.io.read_file(path)
    image = tf.io.decode_image(data, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE, method="bilinear")
    return tf.cast(image, tf.float32) / 255.0


def make_dataset(frame: pd.DataFrame, data_dir: Path, training: bool, batch_size: int) -> tf.data.Dataset:
    paths = [image_path(data_dir, p) for p in frame["Path"].astype(str)]
    labels = frame["ClassId"].to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (load_image(p), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


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


def build_model() -> tf.keras.Model:
    """작은 CNN 대신 BatchNorm, 잔차 연결, 전역 평균 풀링을 사용합니다."""
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = tf.keras.layers.RandomRotation(0.04, fill_mode="reflect", seed=SEED)(inputs)
    x = tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="reflect", seed=SEED + 1)(x)
    x = tf.keras.layers.RandomZoom((-0.12, 0.12), fill_mode="reflect", seed=SEED + 2)(x)
    x = tf.keras.layers.RandomContrast(0.15, seed=SEED + 3)(x)

    for filters in (32, 64, 128, 256):
        shortcut = x
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if shortcut.shape[-1] != filters:
            shortcut = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Dropout(0.15)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="교통 표지판 분류 모델 성능 개선 학습")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="data 폴더 경로")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="결과 저장 폴더")
    parser.add_argument("--epochs", type=int, default=50, help="최대 학습 횟수")
    parser.add_argument("--batch-size", type=int, default=128, help="배치 크기")
    args = parser.parse_args()
    seed_everything()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[시작] 교통 표지판 분류 성능 개선 학습을 시작합니다.")
    print(f"[정보] 데이터 경로: {args.data_dir.resolve()}")
    print(f"[정보] TensorFlow: {tf.__version__}, GPU 수: {len(tf.config.list_physical_devices('GPU'))}")
    train, test = read_metadata(args.data_dir)
    train_df, valid_df = train_test_split(train, test_size=0.2, random_state=SEED, stratify=train["ClassId"])
    print(f"[정보] 학습 이미지: {len(train_df):,}장, 검증 이미지: {len(valid_df):,}장, 테스트 이미지: {len(test):,}장")

    train_ds = make_dataset(train_df, args.data_dir, True, args.batch_size)
    valid_ds = make_dataset(valid_df, args.data_dir, False, args.batch_size)
    test_paths = tf.data.Dataset.from_tensor_slices([image_path(args.data_dir, p) for p in test["Path"]])
    test_ds = test_paths.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight("balanced", classes=classes, y=train_df["ClassId"])
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    model = build_model()
    callbacks = [
        KoreanProgress(),
        tf.keras.callbacks.ModelCheckpoint(args.output_dir / "best_model.keras", monitor="val_accuracy", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=0),
        tf.keras.callbacks.CSVLogger(args.output_dir / "training_log.csv"),
    ]
    print("[학습] 데이터 불균형 보정과 증강을 적용합니다.")
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0,
    )

    result = model.evaluate(valid_ds, return_dict=True, verbose=0)
    print(f"[완료] 검증 정확도: {result['accuracy']:.4%}, 검증 손실: {result['loss']:.4f}")
    valid_pred = np.argmax(model.predict(valid_ds, verbose=0), axis=1)
    report = classification_report(valid_df["ClassId"], valid_pred, labels=list(range(NUM_CLASSES)), output_dict=True, zero_division=0)
    with open(args.output_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    np.savetxt(args.output_dir / "confusion_matrix.csv", confusion_matrix(valid_df["ClassId"], valid_pred, labels=list(range(NUM_CLASSES))), fmt="%d", delimiter=",")

    test_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    pd.DataFrame({"Path": test["Path"], "label": test_pred}).to_csv(args.output_dir / "submission.csv", index=False, encoding="utf-8-sig")
    print(f"[저장] 최적 모델: {args.output_dir / 'best_model.keras'}")
    print(f"[저장] 제출 파일: {args.output_dir / 'submission.csv'}")
    print("[완료] 모든 학습 과정이 끝났습니다.")


if __name__ == "__main__":
    main()
