import os
from collections import defaultdict

import torch

from PIL import Image
from typing import Tuple, Dict, Union

from torch.utils.data.dataset import Dataset
from torchvision import transforms

from src.utils import write_log_message


class BaseDataset(Dataset):
    def __init__(self, path: str, size: int, max_detections: int, augmentation: bool) -> None:
        """
        Initializes the base dataset.

        Args:
            path (str): The directory containing images.
            size (int): The image size.
            max_detections (int): The maximum number of detections.
            augmentation (bool): Whether to apply data augmentation or not.
        """
        super().__init__()

        self.size = size
        self.max_detections = max_detections

        # TODO Add logic for augmentation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory '{path}' does not exist.")

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        """
        Loads an image from a file path.

        Args:
            path (str): The path to the file.

        Returns:
            Image.Image: A PIL Image.
        """
        # Load image file formats using PIL
        _, ext = os.path.splitext(path)
        if ext.lower() in ['.jpg', '.jpeg', '.png']:
            image = Image.open(path).convert('RGB')
        else:
            raise ValueError(f"Extension '{ext}' in '{path}' is not a supported file format.")

        return image

    @staticmethod
    def _load_gt(path: str, data_format: str = "txt") -> torch.Tensor:
        """
        Loads class labels and bounding boxes from a TXT file and converts them to a torch.Tensor.

        Args:
            path (str): The TXT file containing the ground truth.
            data_format (str): The data format.

        Returns:
            A tensor of shape (N, 5) where N is the number of elements in the ground truth. Each row contains a
            Tensor [cls, x1, y1, x2, y2].
        """
        if data_format.lower() == "txt":
            with open(path, 'r') as file:
                lines = file.readlines()

            # Convert each line to a list of bboxes with the corresponding class label
            labels = []
            for line in lines:
                parts = line.strip().split()
                cls = int(parts[0])
                bbox = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                labels.append([cls, bbox[0], bbox[1], bbox[2], bbox[3]])

            # Convert the list of lists to a torch.Tensor
            gt = torch.tensor(labels, dtype=torch.float)
        else:
            raise ValueError(f"'{data_format}' is not a supported data format.")

        return gt

    def adjust_gt_size(self, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adjust the size of the ground truth tensor along the first dimension and create a padding mask.

        Args:
            gt (torch.Tensor): The input tensor to pad or truncate.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The padded or truncated tensor and the padding mask.
        """
        # Get the current shape of the tensor
        current_size = gt.shape[0]

        # Create a mask to save which entries are part of the gt
        mask = torch.ones(gt.shape, dtype=torch.bool)

        # Padding is needed
        if current_size < self.max_detections:
            padding_size = self.max_detections - current_size

            # Create a padding tensor
            padding_tensor = torch.zeros(
                (padding_size, 5),
                dtype=gt.dtype,
                device=gt.device
            )

            # Concatenate the original tensor and the padding tensor
            result = torch.cat([gt, padding_tensor], dim=0)

            # Extend the mask with zeros for the padding
            padding_mask = torch.zeros(padding_tensor.shape, dtype=torch.bool, device=mask.device)
            mask = torch.cat([mask, padding_mask], dim=0)
        else:
            # Reduce the tensor to the maximum number of detections
            result = gt[:self.max_detections]
            mask = mask[:self.max_detections]

        return result, mask

    def _get_files_and_gt(self, path: str, data_format: str = "txt") -> dict:
        """
        Gets a dictionary of image paths and ground truth.

        Args:
            path (str): The root directory containing images.
            data_format (str): The data format.

        Returns:
            dict: A dictionary where the keys are image paths and the values contains the ground truth.
        """
        # Allowed file extensions
        allowed_file_ext = ['.jpg', '.jpeg', '.png']

        paths = defaultdict()
        for root, dirs, files in os.walk(path):
            if not dirs:
                for file in files:
                    base_name, ext = os.path.splitext(file)

                    # Only add files with the allowed file extension
                    if ext not in allowed_file_ext:
                        continue

                    # Ignore files without a ground truth annotation
                    gt_path = os.path.join(root, f"{base_name}.{data_format}")
                    if not os.path.exists(gt_path):
                        continue

                    # Load the gt and ignore documents with more than the maximum number of detections
                    gt_data = self._load_gt(path=gt_path, data_format=data_format)
                    if not gt_data.shape[0] or gt_data.shape[0] > self.max_detections:
                        continue

                    # Add the path of the page image and the ground truth
                    paths[os.path.join(root, file)] = self._load_gt(gt_path)

        return paths

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resizes the image to the specified size.

        Args:
            image (Image.Image): The image that will be resized.

        Returns:
            Image.Image: A PIL image augmented and in the appropriate size.
        """
        return image.resize((self.size, self.size))


class DetectionDataset(BaseDataset):
    def __init__(self, path: str, size: int, max_detections: int = 100, augmentation: bool = False) -> None:
        """
        Initializes the detection dataset.

        Args:
            path (str): The directory containing images.
            size (int): The image size.
            max_detections (int): The maximum number of detections.
            augmentation (bool): Whether to apply data augmentation or not.
        """
        super().__init__(path=path, size=size, max_detections=max_detections, augmentation=augmentation)

        self.paths_and_gt = self._get_files_and_gt(path)
        self.page_images = list(self.paths_and_gt.keys())
        write_log_message(f"{len(self.page_images)} pages have been found.")

    def __len__(self) -> int:
        return len(self.page_images)

    def __getitem__(self, index: int) -> Dict[str, Union[str, torch.Tensor]]:
        """
        Gets an image and the corresponding ground truth label.

        Args:
            index (int): The index of the image.

        Returns:
            Tuple: A tuple containing the image tensor and the ground truth label.
        """
        # Load the image
        image = self._load_image(self.page_images[index])

        # Resize the image
        image = self._resize_image(image)

        # Normalize the image and transform it to a tensor.
        image = self.transform(image)

        # Get the ground truth
        gt = self.paths_and_gt[self.page_images[index]]

        # Adjust the size of the ground truth tensor to match the maximum number of detections
        gt, mask = self.adjust_gt_size(gt)

        return dict(path=self.page_images[index], image_tensor=image, gt_tensor=gt, mask=mask)
