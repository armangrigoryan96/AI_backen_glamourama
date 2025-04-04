import os
import sys
import torch
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter
sys.path.append('./')

# my utils
def crop(human_img: Image, coords: tuple):
    top, left, bottom, right = coords
    cropped_img = human_img.crop((left, top, right, bottom))
    crop_size = cropped_img.size
    human_img = cropped_img.resize((768,1024))
    return human_img, crop_size
    
def get_category_coords(mask_image: Image):
    # Ensure the image is grayscale (convert if necessary)
    mask_image = mask_image.convert("L")
    
    # Threshold to create a binary mask (0 and 1)
    binary_mask = mask_image.point(lambda p: 1 if p > 0 else 0)
    
    # Convert to a NumPy array
    mask_array = np.array(binary_mask)
    
    # Find indices where the value is 1
    indices = np.argwhere(mask_array == 1)
    
    if indices.size == 0:
        raise ValueError("No value '1' found in the matrix.")
    
    # Compute the minimum and maximum coordinates
    top, left = indices.min(axis=0)
    bottom, right = indices.max(axis=0)
    
    return top, left, bottom, right

def image_to_base64(img):
    """Converts a PIL Image to a base64 string."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def resize_with_padding_and_blur(image, target_size=(768, 1024)):
    # Calculate the aspect ratio of the original image
    original_width, original_height = image.size
    target_width, target_height = target_size
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    # Resize the image while maintaining its aspect ratio
    if original_aspect_ratio > target_aspect_ratio:
        # Wider than target aspect ratio
        new_width = target_width
        new_height = int(target_width / original_aspect_ratio)
    else:
        # Taller than target aspect ratio
        new_height = target_height
        new_width = int(target_height * original_aspect_ratio)

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Create a blurred version of the original image for the background
    blurred_background = image.filter(ImageFilter.GaussianBlur(20))
    blurred_background = blurred_background.resize(target_size, Image.LANCZOS)

    # Calculate position to paste the resized image
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    # Paste the resized image onto the blurred background
    blurred_background.paste(resized_image, (paste_x, paste_y))

    return blurred_background

# schem utils
def memory_check(id=None):
    for id in [0]:
        gpu_id  = id
        allocated_memory = torch.cuda.memory_allocated(gpu_id)  # Memory allocated by tensors

        print(f"Allocated memory on GPU {gpu_id}: {allocated_memory / (1024 ** 2)} MB")

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask

def add_logo(img_ndarray):
    if not isinstance(img_ndarray, np.ndarray):
        img_ndarray = np.array(img_ndarray)
    background = Image.fromarray(img_ndarray)#cv2.cvtColor(img_ndarray, cv2.COLOR_BGR2RGB))


    logo_path = f'./data/logo/logo.png'
    logo = Image.open(logo_path)


    coef = 0.08
    logo_size = (int(logo.width * coef), int(logo.height * coef))  # Resize to 20% of the original size
    logo = logo.resize(logo_size)

    # Calculate the position for the logo (bottom-right corner)
    background_width, background_height = background.size
    logo_width, logo_height = logo.size
    position = (background_width - logo_width,
                background_height - logo_height)

    # Paste the logo onto the background image at the calculated position
    background.paste(logo, position, logo)

    return background

def get_base_path():
    config_path = os.path.expanduser('~/.config/myapp/config')
    with open(config_path, 'r') as file:
        config = file.read()
    base_path = config.split('=')[1].strip().strip("'\"")
    return base_path
