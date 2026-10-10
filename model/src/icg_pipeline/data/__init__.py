"""Image and mask loading, preprocessing, and segmentation postprocessing."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _read_image(path: Path, channels: int) -> np.ndarray:
    mode = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    if channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def preprocess_image(
    image: np.ndarray,
    image_size: Tuple[int, int] = (256, 256),
    channels: int = 1,
) -> torch.Tensor:
    """Convert an OpenCV image into normalized CHW model input (size is H, W)."""
    if channels not in (1, 3):
        raise ValueError(f"channels must be 1 or 3, got {channels}")
    if image.ndim == 3 and channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2 and channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    interpolation = cv2.INTER_AREA if image.shape[0] > image_size[0] else cv2.INTER_LINEAR
    image = cv2.resize(image, (image_size[1], image_size[0]), interpolation=interpolation)
    if channels == 1:
        image = image[..., np.newaxis]
    image = image.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)


class ICGDataset(Dataset):
    """Paired images and binary masks; the foreground meaning is set by annotations."""

    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        augmentations: Any = None,
        is_train: bool = True,
        channels: int = 1,
        require_masks: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = tuple(image_size)
        self.is_train = is_train
        self.channels = channels
        image_dir, mask_dir = self.data_dir / "images", self.data_dir / "masks"
        if image_dir.is_dir() and mask_dir.is_dir():
            pairs = [
                (path, self._mask_for(mask_dir, path.stem))
                for path in sorted(image_dir.iterdir())
                if path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        else:
            pairs = [
                (path, self._mask_for(self.data_dir, path.stem, suffix="_mask"))
                for path in sorted(self.data_dir.iterdir())
                if path.suffix.lower() in IMAGE_EXTENSIONS
                and not path.stem.endswith("_mask")
            ]
        if require_masks:
            missing = [str(image) for image, mask in pairs if mask is None]
            if missing:
                raise ValueError(f"Missing masks for {len(missing)} image(s), e.g. {missing[0]}")
            pairs = [(image, mask) for image, mask in pairs if mask is not None]
        if not pairs:
            raise ValueError(f"No {'image/mask pairs' if require_masks else 'images'} found in {self.data_dir}")
        self.pairs = pairs
        self.augmentations = augmentations
        logger.info("Loaded %d image/mask pairs from %s", len(pairs), self.data_dir)

    @staticmethod
    def _mask_for(directory: Path, stem: str, suffix: str = "") -> Optional[Path]:
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            candidate = directory / f"{stem}{suffix}{ext}"
            if candidate.is_file():
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path, mask_path = self.pairs[idx]
        image = _read_image(image_path, self.channels)
        if mask_path is None:
            raise ValueError(f"Missing annotation for training sample: {image_path}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        if mask.shape != image.shape[:2]:
            raise ValueError(f"Image/mask size mismatch: {image_path} and {mask_path}")
        mask = (mask > 0).astype(np.uint8)
        height, width = self.image_size
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA if image.shape[0] > height else cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        if self.is_train:
            if np.random.random() < 0.5:
                image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
            if np.random.random() < 0.5:
                image, mask = cv2.flip(image, 0), cv2.flip(mask, 0)
            if self.image_size[0] == self.image_size[1]:
                rotations = np.random.randint(4)
                image, mask = np.rot90(image, rotations), np.rot90(mask, rotations)
        if self.augmentations is not None:
            transformed = self.augmentations(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
            if isinstance(image, torch.Tensor):
                image_tensor = image.float()
                if image_tensor.max() > 1:
                    image_tensor = (image_tensor / 255.0 - 0.5) / 0.5
                if image_tensor.ndim == 3 and image_tensor.shape[-1] in (1, 3):
                    image_tensor = image_tensor.permute(2, 0, 1)
                mask_tensor = torch.as_tensor(mask).long()
                return {"image": image_tensor, "mask": (mask_tensor > 0).long(), "image_path": str(image_path)}
        image = np.ascontiguousarray(image)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1) if image.ndim == 3 else image[np.newaxis]).float()
        image_tensor = (image_tensor / 255.0 - 0.5) / 0.5
        return {
            "image": image_tensor,
            "mask": torch.from_numpy((np.asarray(mask) > 0).astype(np.int64)),
            "image_path": str(image_path),
        }


def get_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data = config["data"]
    channels = int(data.get("in_channels", config.get("model", {}).get("in_channels", 1)))
    options = {
        "image_size": tuple(data["image_size"]),
        "channels": channels,
    }
    datasets = [
        ICGDataset(data[key], is_train=(key == "train_data_dir"), **options)
        for key in ("train_data_dir", "val_data_dir", "test_data_dir")
    ]
    loaders = [
        DataLoader(
            dataset,
            batch_size=data["batch_size"],
            shuffle=index == 0,
            num_workers=data.get("num_workers", 0),
            pin_memory=data.get("pin_memory", False),
            drop_last=index == 0 and len(dataset) >= data["batch_size"],
        )
        for index, dataset in enumerate(datasets)
    ]
    return tuple(loaders)


def postprocess_mask(
    logits: torch.Tensor,
    original_size: Tuple[int, int],
) -> np.ndarray:
    """Decode two-class logits with softmax/argmax; return a 0/255 mask at H,W."""
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 2:
        raise ValueError(f"Expected logits shaped (1, 2, H, W), got {tuple(logits.shape)}")
    mask = logits.softmax(dim=1).argmax(dim=1)[0].byte().cpu().numpy()
    mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8) * 255


def extract_contours(mask: np.ndarray, min_area: float = 100.0) -> List[np.ndarray]:
    """Return area-filtered external components, largest first; caller selects policy."""
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return sorted(
        (contour for contour in contours if cv2.contourArea(contour) >= min_area),
        key=cv2.contourArea,
        reverse=True,
    )


def contour_to_points(contour: np.ndarray, num_points: int = 100) -> np.ndarray:
    """Uniformly resample a closed contour by arc length into float32 (x,y) points."""
    points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3 or num_points < 3:
        raise ValueError("A closed contour and at least three output points are required")
    closed = np.vstack((points, points[0]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= 0:
        raise ValueError("Contour has zero perimeter")
    samples = np.linspace(0, cumulative[-1], num_points, endpoint=False)
    return np.column_stack(
        [np.interp(samples, cumulative, closed[:, axis]) for axis in range(2)]
    ).astype(np.float32)