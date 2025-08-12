import argparse
from diffusers import DiffusionPipeline, FluxTransformer2DModel
import torch
import os
from utils import insert_community_flux_lora_to_unet
from huggingface_hub import login

record_content_loras = [
    "ginipick/flux-lora-eric-cat",
    "glif-loradex-trainer/antix82_flux_dev_marv_simplecap_v1",
    "glif-loradex-trainer/festerbitcoin_86601_cats",
    "glif-loradex-trainer/fabian3000_chillguy",
    "Keltezaa/LilyC_SoloLoRA_F1V1",
    "markury/brandon-rowland-flux",
    'playboy40k/flux-PiersonWodzynskiLora',#6
    'glif-loradex-trainer/Lamia-000_Rauschenberg',
    'glif-loradex-trainer/danyl3767_alara_character_lora',
    'glif-loradex-trainer/Chris-6f48_Carley',
    'glif-loradex-trainer/Scrilla_Zengar',
    'glif-loradex-trainer/fabian3000_optimus'
]
content_triggers = [
    "eric cat",
    "marv frog man marv",
    "Cat rule the world, ",
    "chillguy",
    "LilyC",
    "Brandon Rowland",
    "Pierson Wodzynski",
    'Rausch',
    'alara girl, anime girl',
    '[Carley]'
    'Zengar',
    'optimus'
]
content_lora_weight_names = [
    "flux-lora-eric-cat.safetensors",
    "flux_dev_marv_simplecap_v1.safetensors",
    "cats.safetensors",
    "chillguy.safetensors",
    'LilyC_SoloLoRA_F1V1.safetensors',
    "pytorch_lora_weights.safetensors",
    "Pierson_Wodzynski-000010.safetensors",
    'Rauschenberg.safetensors',
    'alara_character_lora.safetensors'
    'Carley.safetensors',
    'Zengar.safetensors',
    'optimus.safetensors'
    ]
record_style_loras = [
    "glif-loradex-trainer/bingbangboom_flux_surf",
    "glif-loradex-trainer/mindlywork_AcrylicWorld",
    "glif-loradex-trainer/an303042_Seiwert_Industrial_v1",
    "glif-loradex-trainer/maxxd4240_BlueDraw",
    "glif-loradex-trainer/maxxd4240_SketchOnWater",
    "glif-loradex-trainer/araminta_k_flux_dev_leonardlesliebrookes",
    "glif-loradex-trainer/araminta_k_flux_dev_karl_weiner",
    "glif-loradex-trainer/araminta_k_flux_dev_tarot_test_1",
    "glif-loradex-trainer/i12_appelsiensam_fanimals_v1",
    "glif-loradex-trainer/goldenark__WaterColorSketchStyle",
    "glif-loradex-trainer/fabian3000_henrymajor",
    "glif-loradex-trainer/fabian3000_impressionism2", #11
    "tarfandoon/farshchian_flux", 
    'glif-loradex-trainer/001_Detailed_Manga_Style-v2_1',
    'glif-loradex-trainer/Insectagon_Warhol_style1'
]
style_triggers = [
    "SRFNGV01",
    "Acryl!ck",
    "swrind",
    "BluD!!",
    "SkeWat, water color sketch style",
    "illustration style",
    "collage style",
    "illustration style",
    "FNMLS_PPLSNSM",
    "WaterColorSketchStyle",
    "henrymajorstyle",
    "impressionist", #11
    "FRSCHIAN",
    'X.M4ng4.S7y13',
    'warhol'
]
style_lora_weight_names = [
    "flux_surf.safetensors",
    "AcrylicWorld.safetensors",
    "Seiwert_Industrial_v1.safetensors",
    "BlueDraw.safetensors",
    "SketchOnWater.safetensors", #4
    "flux_dev_leonardlesliebrookes.safetensors",
    "flux_dev_karl_weiner.safetensors",
    "flux_dev_tarot_test_1.safetensors",
    "appelsiensam_fanimals_v1.safetensors",
    "WaterColorSketchStyle.safetensors",
    "henrymajor.safetensors",
    "impressionism2.safetensors", #11
    "flux-lora.safetensors", #12
    'Detailed_Manga_Style-v2_1.safetensors',
    'Warhol_style1.safetensors'
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default = "black-forest-labs/FLUX.1-dev",
        # default="/home/ubuntu/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44/",
        help="Pretrained model path",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Output folder path",
        default="/workspace",
    )
    parser.add_argument(
        "--content_index",
        type=str,
        default="8",
    )
    parser.add_argument(
        "--style_index",
        type=str,
        help="Output folder path",
        default="0",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Pattern for the image generation",
        default="s*",
    )
    return parser.parse_args()


args = parse_args()
pattern = args.pattern
if pattern == "s*":
    alpha = 1.5
    beta = alpha * 0.85
else:
    alpha = 1.5
    beta = 0.5
flux_diffuse_step = 28

content_lora = record_content_loras[int(args.content_index)]
style_lora = record_style_loras[int(args.style_index)]
content_trigger_word = content_triggers[int(args.content_index)]
style_trigger_word = style_triggers[int(args.style_index)]
content_lora_weight_name = content_lora_weight_names[int(args.content_index)]
style_lora_weight_name = style_lora_weight_names[int(args.style_index)]

pipe = DiffusionPipeline.from_pretrained(
    args.pretrained_model_name_or_path, torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

unet = insert_community_flux_lora_to_unet(
    unet=pipe,
    lora_weights_content_path=content_lora,
    lora_weights_style_path=style_lora,
    alpha=alpha,
    beta=beta,
    diffuse_step=flux_diffuse_step,
    content_lora_weight_name=content_lora_weight_name,
    style_lora_weight_name=style_lora_weight_name,
)

prompt = content_trigger_word + " in " + style_trigger_word + " style."
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device, dtype=torch.float16)
print(prompt)

def run():
    seeds = list(range(2))

    for index, seed in enumerate(seeds):
        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipe(prompt=prompt, generator=generator).images[0]
        output_path = os.path.join(args.output_folder, f"output_image_{index}.png")
        print(f"Saving output to {output_path}")
        image.save(output_path)


if __name__ == "__main__":
    run()
