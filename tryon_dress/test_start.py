import os
import sys
sys.path.append('./')
from flask import Flask, request, jsonify, flash

from base.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from base.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from base.unet_hacked_tryon import UNet2DConditionModel

from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
import base_net
from PIL import Image
from typing import List
import torch
from torchvision import transforms
from transformers import AutoTokenizer
from torchvision.transforms.functional import to_pil_image
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation

from diffusers import DDPMScheduler,AutoencoderKL

from dress_mask import get_mask_location
from pose_parsing.humanparsing.run_parsing import Parsing
from pose_parsing.openpose.run_openpose import OpenPose
from size_calculator_utils import calculate, file_storage_to_cv2
from tryon_dress.utils import memory_check, pil_to_binary_mask, add_logo, get_base_path, resize_with_padding_and_blur, get_category_coords, crop, image_to_base64

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

device0 = 'cuda:0' # if torch.cuda.is_available() else 'cpu'
device1 = 'cuda:1'
print(f"torch devices {torch.cuda.device_count()}")
print(torch.cuda.get_device_name(0))

def start_tryon(person_img, garm_img, garment_des, dress, category, is_checked=False, is_checked_crop=False, denoise_steps=32, seed=0):
    print(f"Caption dress: {dress}")
    if category == "lower":
        garment_des = dress + ' Skirts'
    elif category == "dress":
        garment_des = dress + ' dress'
    elif category == "upper":
        garment_des = dress + ' Top'
        
        
    garm_img= garm_img.convert("RGB").resize((768,1024))
    human_img_orig = person_img.convert("RGB")

    if is_checked_crop:
        human_img = resize_with_padding_and_blur(human_img_orig)
        # width, height = human_img.size
        # target_width = int(min(width, height * (3 / 4)))
        # target_height = int(min(height, width * (4 / 3)))
        # left = (width - target_width) / 2
        # top = (height - target_height) / 2
        # right = (width + target_width) / 2
        # bottom = (height + target_height) / 2
        # cropped_img = human_img.crop((left, top, right, bottom))
        # crop_size = cropped_img.size
        # human_img = cropped_img.resize((768,1024))
    else:
        human_img = human_img_orig.resize((768,1024))


    if is_checked:
        keypoints = openpose_model(human_img.resize((384,512)))
        model_parse, _ = parsing_model(human_img.resize((384,512)))
        mask, mask_gray = get_mask_location('dc', category, model_parse, keypoints)
        #mask, mask_gray = get_mask_location('hd', "tops", model_parse, keypoints)
        mask = mask.resize((768,1024))
    else:
        mask = pil_to_binary_mask(person_img['layers'][0].convert("RGB").resize((768, 1024)))
        # mask = transforms.ToTensor()(mask)
        # mask = mask.unsqueeze(0)

    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)


    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = base_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    print("____________args_____________")
    print(args)
    # verbosity = getattr(args, "verbosity", None)
    pose_img = args.func(args,human_img_arg)
    pose_img = pose_img[:,:,::-1]
    pose_img = Image.fromarray(pose_img).resize((768,1024))

    memory_check()

    with torch.no_grad():
        # Extract the images
        with torch.amp.autocast("cuda"):
            with torch.no_grad():
                prompt = "model is wearing " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                print(f"--------flag-0-{device1}---------")
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                        device = torch.device(device0)

                        )

                    prompt = "a photo of " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    print(f"___________flag-1-{device1}-----------")
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                            device = torch.device(device1)
                            )



                    pose_img =  tensor_transfrom(pose_img).unsqueeze(0).to(device0,torch.float16)
                    garm_tensor =  tensor_transfrom(garm_img).unsqueeze(0).to(device0,torch.float16)
                    generator = torch.Generator(device0).manual_seed(seed) if seed is not None else None
                    print(f"___________flag-2-{device0}-----------")
                    print("prompd embeds is cuda: ", prompt_embeds.is_cuda)
                    print("negative prompd embeds is cuda: ", negative_prompt_embeds.is_cuda)
                    print("pooled_prompt_embeds is cuda: ", pooled_prompt_embeds.is_cuda)
                    print("negative_pooled_prompt_embeds is cuda: ", negative_pooled_prompt_embeds.is_cuda)
                    print("prompt_embeds_c is cuda: ", prompt_embeds_c.is_cuda)
                    print("pose img is cuda: ", pose_img.is_cuda)

                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device0,torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(device0,torch.float16),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device0,torch.float16),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device0,torch.float16),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength = 1.0,
                        pose_img = pose_img.to(device0,torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(device0,torch.float16),
                        cloth = garm_tensor.to(device0,torch.float16),
                        mask_image=mask,
                        image=human_img,
                        height=1024,
                        width=768,
                        ip_adapter_image = garm_img.resize((768,1024)),
                        guidance_scale=2.0,
                        device=device0
                        )[0]

    #if is_checked_crop:
    # out_img = images[0].resize(crop_size)
    # human_img.paste(out_img, (int(left), int(top)))
    # return human_img
    #else:
    return images[0]
        #return images[0], mask_gray


if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    base_path = 'yisol/IDM-VTON' #"/base" #get_base_path()

    example_path = os.path.join(os.path.dirname(__file__), 'example')

    unet = UNet2DConditionModel.from_pretrained(
        base_path,
        subfolder="unet",
        torch_dtype=torch.float16,
    )
    unet.requires_grad_(False)
    tokenizer_one = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer_2",
        revision=None,
        use_fast=False,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

    text_encoder_one = CLIPTextModel.from_pretrained(
        base_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        base_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        base_path,
        subfolder="image_encoder",
        torch_dtype=torch.float16,
        )
    vae = AutoencoderKL.from_pretrained(base_path,
                                        subfolder="vae",
                                        torch_dtype=torch.float16,
    )
    memory_check()
    # "stabilityai/stable-diffusion-xl-base-1.0",
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
        base_path,
        subfolder="unet_encoder",
        torch_dtype=torch.float16,
        device=device0
        )
    memory_check()

    #my_unet = nn.DataParallel(UNet_Encoder, device_ids = [0,1])

    #exit()
    parsing_model = Parsing(gpu_id=1)
    memory_check()

    openpose_model = OpenPose(gpu_id=1)
    memory_check()

    UNet_Encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    memory_check()
    text_encoder_one.requires_grad_(False).to(device0)
    memory_check()

    text_encoder_two.requires_grad_(False).to(device1)
    memory_check()


    tensor_transfrom = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
        )
    pipe = TryonPipeline.from_pretrained(
            base_path,
            unet=unet,
            vae=vae,
            feature_extractor= CLIPImageProcessor(),
            text_encoder = text_encoder_one.to(device0,torch.float16),
            text_encoder_2 = text_encoder_two.to(device0,torch.float16),
            tokenizer = tokenizer_one,
            tokenizer_2 = tokenizer_two,
            scheduler = noise_scheduler,
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
            # unet_encoder = UNet_Encoder,
            device_map = device0
            )
    pipe.unet_encoder = UNet_Encoder

    #pipe.unet = pipe.unet.cuda()

    memory_check()

    pipe.unet_encoder = UNet_Encoder.cuda()
    memory_check()

    pipe = pipe.to(device1)
    memory_check()

    print("unet is cuda: ", next(pipe.unet.parameters()).is_cuda)
    print("unet-encoder is cuda: ", next(pipe.unet_encoder.parameters()).is_cuda)
    print("vae is cuda: ", next(pipe.vae.parameters()).is_cuda)
    print("text_encoder is cuda: ", next(pipe.text_encoder.parameters()).is_cuda)
    print("text_encoder_2 is cuda: ", next(pipe.text_encoder_2.parameters()).is_cuda)

    memory_check()

app = Flask(__name__)

@app.route('/virtual-dressing', methods=['POST'])
def virtual_dressing():
    try:
        person_bt = request.files.get('person')
        cloth_bt = request.files.get("cloth")
        person = Image.open(person_bt)
        cloth = Image.open(cloth_bt)
        category = request.form.get("position")    #position = request.form.get("position")
        caption = request.form.get("caption")
        logo = request.form.get("logo")
        caption = "knee-length" if caption == "knee_length" else caption
        print(f"Caption: {caption}")
        print("Starting tryon--------------------")
        res_image = start_tryon(person_img = person,
                                garm_img = cloth,
                                category = category,
                                dress = caption,
                                garment_des = "",
                                # dropdown = position,
                                is_checked = True,
                                is_checked_crop = True,
                                denoise_steps = 28,
                                seed = 42)
        
        if logo.lower() == "yes":
            res_image = add_logo(res_image)

        res_image_base64 = image_to_base64(res_image)

        return jsonify({"res": res_image_base64, "code": 200}), 200
    except Exception as e:
        print(e)
        return jsonify({"res": f"Error reason: {e}", "code": 500}), 500

@app.route('/calculate_measures', methods=['POST'])
def calculate_measures():
    try:
        if request.method == 'POST':
            if "height" not in request.form:
                flash('No height')
                return {"error": "Height is required"}
            if 'front' not in request.files:
                flash('No file part')
                return {"error": "Front image is required"}
            if 'side' not in request.files:
                flash('No file part')   
                return {"error": "Side image is required"}


            Height = (int)(request.form.get('height'))
            front = request.files['front']
            side = request.files['side']
            front_image = file_storage_to_cv2(front)
            side_image = file_storage_to_cv2(side)
            
            # front.save(front_path)
            # side.save(side_path)
            
            if front.filename == '':
                flash('No selected file')
            if side.filename == '':
                flash('No selected file')
        else:
            print("Shouldn't use GET method")
            return

        image_base_64, body_parts = calculate(front_image, side_image, Height)
        return {'image': image_base_64, "body_parts": body_parts}
       
    except Exception as e:
        print(e)
        return {'error': f"Error reason: {e}"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


    

