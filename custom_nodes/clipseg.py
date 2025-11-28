from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from PIL import Image
import torch
import numpy as np

import matplotlib.cm as cm


import cv2

from scipy.ndimage import gaussian_filter

from typing import Optional, Tuple
import threading

import comfy.model_management as mm

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="safetensors")



"""Helper methods for CLIPSeg nodes"""

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a numpy array and scale its values to 0-255."""
    tensor_cpu = tensor.detach().to("cpu")

    if tensor_cpu.dtype != torch.float32:
        tensor_cpu = tensor_cpu.float()

    tensor_cpu = tensor_cpu.clamp(0, 1).contiguous()
    array = tensor_cpu.numpy()

    if array.ndim == 4 and array.shape[0] == 1:
        array = array.squeeze(0)

    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (3, 4):
        array = np.transpose(array, (1, 2, 0))

    array = np.clip(array, 0, 1)
    return (array * 255).astype(np.uint8)

def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a tensor and scale its values from 0-255 to 0-1."""
    array = array.astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]

def apply_colormap(mask: torch.Tensor, colormap) -> np.ndarray:
    """Apply a colormap to a tensor and convert it to a numpy array."""
    mask_cpu = mask.detach().to("cpu")

    if mask_cpu.dtype != torch.float32:
        mask_cpu = mask_cpu.float()

    mask_cpu = mask_cpu.clamp(0, 1).contiguous()
    mask_np = mask_cpu.numpy().squeeze()
    colored_mask = colormap(mask_np)[:, :, :3]
    colored_mask = np.clip(colored_mask, 0, 1)
    return (colored_mask * 255).astype(np.uint8)

def resize_image(image: np.ndarray, dimensions: Tuple[int, int]) -> np.ndarray:
    """Resize an image to the given dimensions using linear interpolation."""
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_LINEAR)

def overlay_image(background: np.ndarray, foreground: np.ndarray, alpha: float) -> np.ndarray:
    """Overlay the foreground image onto the background with a given opacity (alpha)."""
    return cv2.addWeighted(background, 1 - alpha, foreground, alpha, 0)

def dilate_mask(mask: torch.Tensor, dilation_factor: float) -> torch.Tensor:
    """Dilate a mask using a square kernel with a given dilation factor."""
    kernel_size = int(dilation_factor * 2) + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_cpu = mask.detach().to("cpu")

    if mask_cpu.dtype != torch.float32:
        mask_cpu = mask_cpu.float()

    mask_np = mask_cpu.clamp(0, 1).contiguous().numpy()
    mask_np = np.clip(mask_np, 0, 1).astype(np.float32, copy=False)
    mask_dilated = cv2.dilate(mask_np, kernel, iterations=1)
    mask_dilated = np.clip(mask_dilated, 0, 1).astype(np.float32, copy=False)
    return torch.from_numpy(mask_dilated)



class CLIPSeg:
    _MODEL_ID = "CIDAS/clipseg-rd64-refined"
    _PROCESSOR: Optional[CLIPSegProcessor] = None
    _MODEL_CACHE: dict[str, CLIPSegForImageSegmentation] = {}
    _CACHE_LOCK = threading.Lock()

    def __init__(self):
        pass

    @classmethod
    def _get_device(cls, device_selection: Optional[str]) -> torch.device:
        if device_selection and device_selection != "default":
            return torch.device(device_selection)
        return mm.get_torch_device()

    @classmethod
    def _get_processor_and_model(cls, device: torch.device) -> Tuple[CLIPSegProcessor, CLIPSegForImageSegmentation]:
        processor = cls._PROCESSOR
        model = cls._MODEL_CACHE.get(str(device))

        if processor is None or model is None:
            with cls._CACHE_LOCK:
                if cls._PROCESSOR is None:
                    cls._PROCESSOR = CLIPSegProcessor.from_pretrained(cls._MODEL_ID)
                if str(device) not in cls._MODEL_CACHE:
                    loaded_model = CLIPSegForImageSegmentation.from_pretrained(cls._MODEL_ID)
                    loaded_model.to(device)
                    loaded_model.eval()
                    cls._MODEL_CACHE[str(device)] = loaded_model
                processor = cls._PROCESSOR
                model = cls._MODEL_CACHE[str(device)]

        return processor, model
    
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {"required":
                    {
                        "image": ("IMAGE",),
                        "text": ("STRING", {"multiline": False}),
                        
                     },
                "optional":
                    {
                        "blur": ("FLOAT", {"min": 0, "max": 15, "step": 0.1, "default": 7}),
                        "threshold": ("FLOAT", {"min": 0, "max": 1, "step": 0.05, "default": 0.4}),
                        "dilation_factor": ("INT", {"min": 0, "max": 10, "step": 1, "default": 4}),
                        "device": (["default", "cpu", "cuda"], {"default": "default"}),
                    }
                }

    CATEGORY = "image"
    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("Mask","Heatmap Mask", "BW Mask")

    FUNCTION = "segment_image"
    def segment_image(self, image: torch.Tensor, text: str, blur: float, threshold: float, dilation_factor: int, device: str = "default") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a segmentation mask from an image and a text prompt using CLIPSeg.

        Args:
            image (torch.Tensor): The image to segment.
            text (str): The text prompt to use for segmentation.
            blur (float): How much to blur the segmentation mask.
            threshold (float): The threshold to use for binarizing the segmentation mask.
            dilation_factor (int): How much to dilate the segmentation mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The segmentation mask, the heatmap mask, and the binarized mask.
        """
            
        # Convert the Tensor to a PIL image
        image_np = tensor_to_numpy(image)
        # Create a PIL image from the numpy array
        i = Image.fromarray(image_np, mode="RGB")

        execution_device = self._get_device(device)
        processor, model = self._get_processor_and_model(execution_device)

        prompt = text

        input_prc = processor(text=prompt, images=[i], return_tensors="pt")
        input_prc = {k: v.to(execution_device) if hasattr(v, "to") else v for k, v in input_prc.items()}

        # Predict the segmentation mask
        with torch.no_grad():
            outputs = model(**input_prc)

        # see https://huggingface.co/blog/clipseg-zero-shot
        preds = outputs.logits.unsqueeze(1)
        tensor = torch.sigmoid(preds[0][0]).cpu() # get the mask
                
        # Apply a threshold to the original tensor to cut off low values
        thresh = threshold
        tensor_thresholded = torch.where(tensor > thresh, tensor, torch.tensor(0, dtype=torch.float))

        # Apply Gaussian blur to the thresholded tensor
        sigma = blur
        tensor_smoothed = gaussian_filter(tensor_thresholded.numpy(), sigma=sigma)
        tensor_smoothed = torch.from_numpy(tensor_smoothed).float()

        # Normalize the smoothed tensor to [0, 1]
        mask_min = tensor_smoothed.min()
        mask_max = tensor_smoothed.max()
        range_val = mask_max - mask_min
        if range_val > 0:
            mask_normalized = (tensor_smoothed - mask_min) / range_val
        else:
            mask_normalized = torch.zeros_like(tensor_smoothed)

        # Dilate the normalized mask
        mask_dilated = dilate_mask(mask_normalized, dilation_factor)

        # Convert the mask to a heatmap and a binary mask
        heatmap = apply_colormap(mask_dilated, cm.viridis)
        binary_mask = apply_colormap(mask_dilated, cm.Greys_r)

        # Overlay the heatmap and binary mask on the original image
        dimensions = (image_np.shape[1], image_np.shape[0])
        heatmap_resized = resize_image(heatmap, dimensions)
        binary_mask_resized = resize_image(binary_mask, dimensions)

        alpha_heatmap, alpha_binary = 0.5, 1
        overlay_heatmap = overlay_image(image_np, heatmap_resized, alpha_heatmap)
        overlay_binary = overlay_image(image_np, binary_mask_resized, alpha_binary)

        # Convert the numpy arrays to tensors
        image_out_heatmap = numpy_to_tensor(overlay_heatmap)
        image_out_binary = numpy_to_tensor(overlay_binary)

        # Save or display the resulting binary mask
        binary_mask_image = Image.fromarray(binary_mask_resized[..., 0])

        # convert PIL image to numpy array
        tensor_bw = binary_mask_image.convert("RGB")
        tensor_bw = np.array(tensor_bw).astype(np.float32) / 255.0
        tensor_bw = torch.from_numpy(tensor_bw)[None,]
        tensor_bw = tensor_bw.squeeze(0)[..., 0]

        return tensor_bw, image_out_heatmap, image_out_binary

    #OUTPUT_NODE = False

class CombineMasks:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "input_image": ("IMAGE", ),
                        "mask_1": ("MASK", ), 
                        "mask_2": ("MASK", ),
                    },
                "optional": 
                    {
                        "mask_3": ("MASK",), 
                    },
                }
        
    CATEGORY = "image"
    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("Combined Mask","Heatmap Mask", "BW Mask")

    FUNCTION = "combine_masks"
            
    def combine_masks(self, input_image: torch.Tensor, mask_1: torch.Tensor, mask_2: torch.Tensor, mask_3: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A method that combines two or three masks into one mask. Takes in tensors and returns the mask as a tensor, as well as the heatmap and binary mask as tensors."""

        # Combine masks
        combined_mask = mask_1 + mask_2 + mask_3 if mask_3 is not None else mask_1 + mask_2


        # Convert image and masks to numpy arrays
        image_np = tensor_to_numpy(input_image)
        heatmap = apply_colormap(combined_mask, cm.viridis)
        binary_mask = apply_colormap(combined_mask, cm.Greys_r)

        # Resize heatmap and binary mask to match the original image dimensions
        dimensions = (image_np.shape[1], image_np.shape[0])
        heatmap_resized = resize_image(heatmap, dimensions)
        binary_mask_resized = resize_image(binary_mask, dimensions)

        # Overlay the heatmap and binary mask onto the original image
        alpha_heatmap, alpha_binary = 0.5, 1
        overlay_heatmap = overlay_image(image_np, heatmap_resized, alpha_heatmap)
        overlay_binary = overlay_image(image_np, binary_mask_resized, alpha_binary)

        # Convert overlays to tensors
        image_out_heatmap = numpy_to_tensor(overlay_heatmap)
        image_out_binary = numpy_to_tensor(overlay_binary)

        return combined_mask, image_out_heatmap, image_out_binary

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "CLIPSeg": CLIPSeg,
    "CombineSegMasks": CombineMasks,
}
