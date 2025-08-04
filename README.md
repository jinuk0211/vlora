# vlora
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/home/ubuntu/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44/",
        help="Pretrained model path",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Output folder path",
        default="output/",
    )
    parser.add_argument(
        "--content_index",
        type=str,
        default="0",
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
