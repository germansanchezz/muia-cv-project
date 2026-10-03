from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import rasterio
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "xview_recognition"
TRAIN_ANNOTATIONS_PATH = DATASET_DIR / "xview_ann_train.json"
TRAIN_IMAGES_DIR = DATASET_DIR / "xview_train"
TEST_IMAGES_DIR = DATASET_DIR / "xview_test"

CATEGORIES = {
    0: "Cargo plane",
    1: "Small car",
    2: "Bus",
    3: "Truck",
    4: "Motorboat",
    5: "Fishing vessel",
    6: "Dump truck",
    7: "Excavator",
    8: "Building",
    9: "Helipad",
    10: "Storage tank",
    11: "Shipping container",
    12: "Pylon",
}
CATEGORY_TO_INDEX = {name: index for index, name in CATEGORIES.items()}


@dataclass
class GenericObject:
    bb: tuple[int, int, int, int] = (-1, -1, -1, -1)
    category: str = ""
    score: float = -1.0


@dataclass
class GenericImage:
    filename: str
    width: int = 224
    height: int = 224
    tile: np.ndarray = field(default_factory=lambda: np.array([-1, -1, -1, -1]))
    objects: list[GenericObject] = field(default_factory=list)

    def add_object(self, obj: GenericObject) -> None:
        self.objects.append(obj)


def load_geoimage(
    filename: str | Path,
    dataset_dir: Path = DATASET_DIR,
    normalize: bool = False,
) -> np.ndarray:
    image_path = Path(filename)
    if not image_path.is_absolute():
        image_path = dataset_dir / image_path

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(image_path, "r") as src_raster:
            image = np.moveaxis(src_raster.read(), 0, -1)

    image = image.astype(np.float32)
    if normalize:
        image /= 255.0
    return image


def load_annotations(annotation_path: Path = TRAIN_ANNOTATIONS_PATH) -> dict:
    with annotation_path.open(encoding="utf-8") as annotation_file:
        return json.load(annotation_file)


def build_training_annotations(json_data: dict) -> list[GenericImage]:
    images_by_id = {
        image_data["image_id"]: image_data
        for image_data in json_data["images"].values()
    }
    annotations = []
    for annotation_data in json_data["annotations"].values():
        image_data = images_by_id[annotation_data["image_id"]]
        image = GenericImage(
            filename=image_data["filename"],
            width=int(image_data["width"]),
            height=int(image_data["height"]),
            tile=np.array([0, 0, image_data["width"], image_data["height"]]),
        )
        bbox = tuple(int(value) for value in annotation_data["bbox"])
        image.add_object(
            GenericObject(
                bb=bbox,
                category=annotation_data["category_id"],
            )
        )
        annotations.append(image)
    return annotations


def build_test_annotations(test_dir: Path = TEST_IMAGES_DIR) -> list[GenericImage]:
    annotations = []
    for image_path in sorted(test_dir.glob("*.tif")):
        relative_filename = image_path.relative_to(DATASET_DIR).as_posix()
        image = GenericImage(filename=relative_filename)
        image.add_object(GenericObject(bb=(0, 0, 224, 224)))
        annotations.append(image)
    return annotations


def split_annotations(
    annotations: Sequence[GenericImage],
    validation_fraction: float = 0.1,
    random_state: int = 1,
    stratify: bool = False,
) -> tuple[list[GenericImage], list[GenericImage]]:
    from sklearn.model_selection import train_test_split

    labels = [image.objects[0].category for image in annotations] if stratify else None
    train, validation = train_test_split(
        list(annotations),
        test_size=validation_fraction,
        random_state=random_state,
        shuffle=True,
        stratify=labels,
    )
    return train, validation


def object_records(
    annotations: Iterable[GenericImage],
) -> list[tuple[str, GenericObject]]:
    return [
        (image.filename, obj)
        for image in annotations
        for obj in image.objects
    ]


def image_generator(
    records: list[tuple[str, GenericObject]],
    batch_size: int,
    categories: dict[int, str] = CATEGORIES,
    dataset_dir: Path = DATASET_DIR,
    do_shuffle: bool = False,
    shuffle: bool | None = None,
    normalize: bool = False,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if shuffle is not None:
        do_shuffle = shuffle
    category_to_index = {name: index for index, name in categories.items()}
    while True:
        current_records = list(records)
        if do_shuffle:
            np.random.shuffle(current_records)
        for start in range(0, len(current_records), batch_size):
            group = current_records[start:start + batch_size]
            images = np.asarray([
                load_geoimage(filename, dataset_dir=dataset_dir, normalize=normalize)
                for filename, _ in group
            ], dtype=np.float32)
            labels = np.zeros((len(group), len(categories)), dtype=np.float32)
            for index, (_, obj) in enumerate(group):
                labels[index, category_to_index[obj.category]] = 1.0
            yield images, labels


def steps_for_records(records: Sequence[tuple[str, GenericObject]], batch_size: int) -> int:
    return math.ceil(len(records) / batch_size)


def draw_confusion_matrix(cm: np.ndarray, categories: dict[int, str]) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=[6.4 * pow(len(categories), 0.5), 4.8 * pow(len(categories), 0.5)])
    ax = fig.add_subplot(111)
    cm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], np.finfo(np.float64).eps)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.colormaps["Blues"])
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=list(categories.values()),
        yticklabels=list(categories.values()),
        ylabel="Annotation",
        xlabel="Prediction",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], ".2f"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=int(20 - pow(len(categories), 0.5)),
            )
    fig.tight_layout()
    plt.show()


def predictions_for_annotations(
    model,
    annotations: Iterable[GenericImage],
    dataset_dir: Path = DATASET_DIR,
) -> tuple[list[str], list[str]]:
    y_true, y_pred = [], []
    category_names = list(CATEGORIES.values())
    for image in annotations:
        input_image = np.expand_dims(
            load_geoimage(image.filename, dataset_dir=dataset_dir),
            axis=0,
        )
        prediction = model.predict(input_image, verbose=0)[0]
        predicted_category = category_names[int(np.argmax(prediction))]
        for obj in image.objects:
            y_true.append(obj.category)
            y_pred.append(predicted_category)
    return y_true, y_pred


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    categories: dict[int, str] = CATEGORIES,
) -> dict:
    labels = list(categories.values())
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix": cm,
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }


def save_prediction_json(
    model,
    annotations: Iterable[GenericImage],
    output_path: Path,
    dataset_dir: Path = DATASET_DIR,
) -> None:
    category_names = list(CATEGORIES.values())
    predictions_data = {"images": {}, "annotations": {}}
    for index, image in enumerate(annotations):
        image_id = Path(image.filename).name
        predictions_data["images"][str(index)] = {
            "image_id": image_id,
            "filename": image.filename,
            "width": image.width,
            "height": image.height,
        }
        input_image = np.expand_dims(
            load_geoimage(image.filename, dataset_dir=dataset_dir),
            axis=0,
        )
        prediction = model.predict(input_image, verbose=0)[0]
        predicted_category = category_names[int(np.argmax(prediction))]
        obj = image.objects[0]
        predictions_data["annotations"][str(index)] = {
            "image_id": image_id,
            "category_id": predicted_category,
            "bbox": list(obj.bb),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(predictions_data, output_file, indent=2)


def build_dataset_in_memory(
    records: list[tuple[str, GenericObject]],
    categories: dict[int, str] = CATEGORIES,
    dataset_dir: Path = DATASET_DIR,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    category_to_index = {name: index for index, name in categories.items()}
    
    # Preasignar arrays de NumPy para evitar fragmentación de memoria
    X = np.empty((len(records), 224, 224, 3), dtype=np.float32)
    y = np.zeros((len(records), len(categories)), dtype=np.float32)
    
    for i, (filename, obj) in enumerate(records):
        X[i] = load_geoimage(filename, dataset_dir=dataset_dir, normalize=normalize)
        y[i, category_to_index[obj.category]] = 1.0
        
    return X, y
