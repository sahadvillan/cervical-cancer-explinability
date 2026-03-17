import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import base64
from io import BytesIO

def generate_grad_cam_base64(model, image_tensor, original_image_np):
    """
    Generates a Grad-CAM heatmap overlaid on the original image and returns it as a Base64 string.
    
    Args:
        model (nn.Module): The PyTorch model (CervicalCancerModel).
        image_tensor (torch.Tensor): The preprocessed 4D tensor (1, C, H, W).
        original_image_np (np.ndarray): The original image as a float32 RGB numpy array (scaled 0-1).
        
    Returns:
        str: Base64 encoded JPEG string.
    """
    # Assuming model has a get_target_layer() method
    target_layers = [model.get_target_layer()]

    # Construct the CAM object once, and then re-use it on many images
    with GradCAM(model=model, target_layers=target_layers) as cam:
        # You can set target_category to None for the highest scoring category
        targets = None
        
        # Generate the grayscale CAM heatmap
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
        
        # Take the first channel since we passed a batch of 1
        grayscale_cam = grayscale_cam[0, :]
        
        # The original image needs to be float32, RGB and scaled [0, 1]
        # Overlay the heatmap
        visualization = show_cam_on_image(original_image_np, grayscale_cam, use_rgb=True)
        
        # Convert numpy array (RGB) to PIL Image
        pil_img = Image.fromarray(visualization)
        
        # Convert to Base64
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
    return img_str
