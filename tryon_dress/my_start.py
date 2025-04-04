import os
import sys
sys.path.append('./')

from flask import Flask, request, jsonify, flash
from PIL import Image
from typing import List
import torch
from torchvision import transforms
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import DDPMScheduler, AutoencoderKL
from dress_mask import get_mask_location
from pose_parsing.humanparsing.run_parsing import Parsing
from pose_parsing.openpose.run_openpose import OpenPose
from size_calculator_utils import calculate, file_storage_to_cv2
from tryon_dress.utils import (
    memory_check, pil_to_binary_mask, add_logo, resize_with_padding_and_blur, get_category_coords, image_to_base64
)
from base.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from base.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from base.unet_hacked_tryon import UNet2DConditionModel
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

device0 = 'cuda:0'
device1 = device0

def initialize_models(base_path):
    """Initialize all required models and components."""
    memory_check()

    unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet", torch_dtype=torch.float16)
    tokenizer_one = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer_2", use_fast=False)
    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
    
    text_encoder_one = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device0)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(base_path, subfolder="text_encoder_2", torch_dtype=torch.float16).to(device1)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(base_path, subfolder="image_encoder", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae", torch_dtype=torch.float16)
    
    unet.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    unet_encoder = UNet2DConditionModel_ref.from_pretrained(base_path, subfolder="unet_encoder", torch_dtype=torch.float16).to(device0)
    unet_encoder.requires_grad_(False)

    parsing_model = Parsing(gpu_id=1)
    openpose_model = OpenPose(gpu_id=1)

    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16
    )
    pipe.unet_encoder = unet_encoder

    return pipe, parsing_model, openpose_model, tensor_transform

def process_image(image, size=(768, 1024), apply_padding=False):
    """Resize and optionally apply padding to the image."""
    if apply_padding:
        return resize_with_padding_and_blur(image)
    return image.resize(size)

def start_tryon(person_img, garm_img, garment_des, dress, category, parsing_model, openpose_model, pipe, tensor_transform, denoise_steps=32, seed=0):
    """Run the try-on pipeline."""
    garm_img = process_image(garm_img.convert("RGB"))
    human_img_orig = process_image(person_img.convert("RGB"), apply_padding=True)

    keypoints = openpose_model(human_img_orig.resize((384, 512)))
    model_parse, _ = parsing_model(human_img_orig.resize((384, 512)))
    mask, _ = get_mask_location('dc', category, model_parse, keypoints)
    mask = mask.resize((768, 1024))

    human_img_arg = convert_PIL_to_numpy(_apply_exif_orientation(human_img_orig.resize((384, 512))), format="BGR")

    prompt = f"model is wearing {garment_des}"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
    
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            prompt_embeds, negative_prompt_embeds, *_ = pipe.encode_prompt(
                prompt, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=negative_prompt, device=torch.device(device0)
            )

            pose_img_tensor = tensor_transform(human_img_orig).unsqueeze(0).to(device0, torch.float16)
            garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device0, torch.float16)
            generator = torch.Generator(device0).manual_seed(seed) if seed is not None else None

            images = pipe(
                prompt_embeds=prompt_embeds.to(device0, torch.float16),
                negative_prompt_embeds=negative_prompt_embeds.to(device0, torch.float16),
                num_inference_steps=denoise_steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_img_tensor,
                cloth=garm_tensor,
                mask_image=mask,
                height=1024,
                width=768,
                guidance_scale=2.0,
                device=device0
            )[0]

    output_img = images[0]
    keypoints = openpose_model(output_img.resize((384, 512)))
    model_parse, _ = parsing_model(output_img.resize((384, 512)))
    mask, _ = get_mask_location('dc', category, model_parse, keypoints)
    top, left, bottom, right = get_category_coords(mask)
    cropped_img = output_img.crop((left, top, right, bottom))
    human_img_orig.paste(cropped_img, (int(left), int(top)))

    return human_img_orig

app = Flask(__name__)

@app.route('/virtual-dressing', methods=['POST'])
def virtual_dressing():
#    try:
        person_bt = request.files.get('person')
        cloth_bt = request.files.get('cloth')
        category = request.form.get("position")
        caption = request.form.get("caption")
        logo = request.form.get("logo")

        person = Image.open(person_bt)
        cloth = Image.open(cloth_bt)
        res_image = start_tryon(
            person_img=person,
            garm_img=cloth,
            category=category,
            dress=caption,
            garment_des="",
            parsing_model=parsing_model,
            openpose_model=openpose_model,
            pipe=pipe,
            tensor_transform=tensor_transform,
            denoise_steps=40,
            seed=42
        )

        if logo and logo.lower() == "yes":
            res_image = add_logo(res_image)

        res_image_base64 = image_to_base64(res_image)
        return jsonify({"res": res_image_base64, "code": 200}), 200

 #   except Exception as e:
  #      return jsonify({"res": f"Error: {str(e)}", "code": 500}), 500

if __name__ == '__main__':
    base_path = 'yisol/IDM-VTON'
    pipe, parsing_model, openpose_model, tensor_transform = initialize_models(base_path)
    app.run(host='0.0.0.0', port=5003, debug=True)

