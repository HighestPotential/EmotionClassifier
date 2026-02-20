import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite
from gfpgan import GFPGANer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input",
        type=str,
        default="inputs/whole_imgs",
        help="Input image or folder"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="results",
        help="Output folder"
    )
    parser.add_argument(
        "-v", "--version",
        type=str,
        default="1.3",
        help="GFPGAN model version"
    )
    parser.add_argument(
        "-s", "--upscale",
        type=int,
        default=1,
        help="Upscale factor"
    )
    parser.add_argument(
        "--bg_upsampler",
        type=str,
        default="realesrgan",
        help="Background upsampler"
    )
    parser.add_argument(
        "--bg_tile",
        type=int,
        default=400,
        help="Tile size for background upsampler"
    )
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--only_center_face", action="store_true")
    parser.add_argument("--aligned", action="store_true")
    parser.add_argument(
        "--ext",
        type=str,
        default="auto",
        help="Image extension"
    )
    parser.add_argument(
        "-w", "--weight",
        type=float,
        default=0.5,
        help="Restoration weight"
    )

    args = parser.parse_args()

    # -------------------- INPUT IMAGES --------------------
    if args.input.endswith("/"):
        args.input = args.input[:-1]

    if os.path.isfile(args.input):
        img_list = [args.input]
    else:
        img_list = sorted(glob.glob(os.path.join(args.input, "*")))

    # -------------------- OUTPUT DIRS --------------------
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "restored_imgs112"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "cropped_faces112"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "restored_faces112"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "cmp"), exist_ok=True)

    # -------------------- BG UPSAMPLER --------------------
    bg_upsampler = None
    if args.bg_upsampler == "realesrgan" and torch.cuda.is_available():
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23,
            num_grow_ch=32, scale=2
        )

        bg_upsampler = RealESRGANer(
            scale=2,
            model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            model=model,
            tile=args.bg_tile,
            tile_pad=10,
            pre_pad=0,
            half=True
        )

    # -------------------- MODEL SETUP --------------------
    if args.version == "1.3":
        arch = "clean"
        channel_multiplier = 2
        model_name = "GFPGANv1.3"
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"

    elif args.version == '1.2':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANCleanv1-NoCE-C2'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
    else:
        raise ValueError("Unsupported GFPGAN version")

    model_path = os.path.join("experiments/pretrained_models", model_name + ".pth")
    if not os.path.isfile(model_path):
        model_path = os.path.join("gfpgan/weights", model_name + ".pth")
    if not os.path.isfile(model_path):
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        upscale=args.upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler
    )

    # ==================== MAIN LOOP ====================
    for img_path in img_list:
        img_name = os.path.basename(img_path)
        basename, ext = os.path.splitext(img_name)

        print(f"Processing {img_name} ...")

        input_img = cv2.imread(img_path)
        if input_img is None:
            print(f"⚠️ Skipping unreadable image: {img_path}")
            continue

        with torch.no_grad():
            cropped_faces, restored_faces, restored_img = restorer.enhance(
                input_img,
                has_aligned=args.aligned,
                only_center_face=args.only_center_face,
                paste_back=True,
                weight=args.weight
            )

        # ---------- SAVE FULL IMAGE ----------
        if restored_img is not None:
            extension = ext[1:] if args.ext == "auto" else args.ext

            if args.suffix:
                out_name = f"{basename}_{args.suffix}.{extension}"
            else:
                out_name = f"{basename}.{extension}"

            imwrite(
                restored_img,
                os.path.join(args.output, "restored_imgs112", out_name)
            )

        # ---------- SAVE FACES ----------
        cropped_faces = [cv2.resize(f, (112, 112)) for f in cropped_faces]
        restored_faces = [cv2.resize(f, (112, 112)) for f in restored_faces]

        for idx, (cf, rf) in enumerate(zip(cropped_faces, restored_faces)):
            imwrite(
                cf,
                os.path.join(args.output, "cropped_faces112", f"{basename}_{idx:02d}.png")
            )
            imwrite(
                rf,
                os.path.join(args.output, "restored_faces112", f"{basename}_{idx:02d}.png")
            )

            cmp = np.concatenate([cf, rf], axis=1)
            imwrite(
                cmp,
                os.path.join(args.output, "cmp", f"{basename}_{idx:02d}.png")
            )

    print(f"\n✅ Results saved in: {args.output}")


if __name__ == "__main__":
    main()
