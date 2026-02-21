
"""
# Clone GFPGAN and enter the GFPGAN folder
%cd /content
!rm -rf GFPGAN
!git clone https://github.com/TencentARC/GFPGAN.git
%cd GFPGA

# Set up the environment
# Install basicsr - https://github.com/xinntao/BasicSR
# We use BasicSR for both training and inference
#!pip install basicsr
# Install facexlib - https://github.com/xinntao/facexlib
# We use face detection and face restoration helper in the facexlib package
!pip install facexlib
# Install other depencencies
!pip install -r requirements.txt
!python setup.py develop
!pip install realesrgan  # used for enhancing the background (non-face) regions
# Download the pre-trained model
# !wget https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth -P experiments/pretrained_models
# Now we use the V1.3 model for the demo


# THAN:
# Replace 'functional_tensor' with 'functional' in the degradations.py file
!sed -i 's/functional_tensor/functional/' /usr/local/lib/python3.12/dist-packages/basicsr/data/degradations.py
"""
# ========================================================
# GFPGAN Batch Inference Script - Full Images 64x64
# ========================================================

# ========================================================
# GFPGAN Batch Inference Script - Full Images 64x64
# ========================================================

import os
import subprocess
from pathlib import Path
import cv2

# -------------------- CONFIG --------------------
GFPGAN_DIR = Path("/home/k/kienzlehagen/GFPGAN")  # Where GFPGAN repo lives
DATASETS_ROOT = Path("/home/k/kienzlehagen/version_3/latest_3_0_ready_to_use_datasets")
RESULTS_ROOT = GFPGAN_DIR / "results64x64"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# -------------------- CLONE GFPGAN IF MISSING --------------------
if not GFPGAN_DIR.exists():
    print("Cloning GFPGAN repository...")
    subprocess.run(["git", "clone", "https://github.com/TencentARC/GFPGAN.git", str(GFPGAN_DIR)], check=True)

# -------------------- OPTIONAL: PATCH BasicSR --------------------
try:
    import basicsr
    degradations_file = Path(basicsr.__path__[0]) / "data/degradations.py"
    if degradations_file.exists():
        with open(degradations_file, "r") as f:
            code = f.read()
        if "functional_tensor" in code and "functional" not in code:
            code = code.replace("functional_tensor", "functional")
            with open(degradations_file, "w") as f:
                f.write(code)
            print("Patched 'functional_tensor' -> 'functional' in basicsr.")
except ImportError:
    print("Warning: basicsr not installed. Make sure your Conda environment has it.")

# -------------------- RUN GFPGAN INFERENCE --------------------
# Walk nested folders: dataset/split/class/leaf
for dataset_dir in DATASETS_ROOT.iterdir():
    if not dataset_dir.is_dir():
        continue
    for split_dir in dataset_dir.iterdir():
        if not split_dir.is_dir():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            # Prepare output folder
            output_dir = RESULTS_ROOT / dataset_dir.name / split_dir.name / class_dir.name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Build command to run GFPGAN
            cmd = [
                "python",  # Use the current env's Python
                str(GFPGAN_DIR / "inference_gfpganCIP6464.py"),
                "-i", str(class_dir),
                "-o", str(output_dir),
                "-v", "1.3",            # GFPGAN version
                "-s", "1",              # Keep original size
                "--aligned",            # Only if images are aligned
                "--bg_upsampler", "realesrgan"
            ]

            print(f"Running GFPGAN on {class_dir} ...")
            subprocess.run(cmd, cwd=str(GFPGAN_DIR), check=True)

            # -------------------- POST-PROCESS: RESIZE TO 64X64------------------
            restored_imgs_dir = output_dir / "restored_imgs64"
            if restored_imgs_dir.exists():
                for img_path in restored_imgs_dir.glob("*"):
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.resize(img, (64,64))
                        cv2.imwrite(str(img_path), img)

print("✅ All images processed and resized to 64x64.")




# ===========================================================================
# The printing -> Do it probably in another file as a Jupyter notbeook
# import os
# import glob
# import cv2
# import matplotlib.pyplot as plt
# import time

# time.sleep(2)


# input_folder = r"D:\documentos\Studium\3Semester\compVisionDL\project\latest_1_0_ready_to_use_datasets\Nova_pasta\jaffeFormated\train\fear"
# result_folder = r"D:\documentos\Studium\3Semester\compVisionDL\CNNs\GFPGANData\GFPGAN\results\jaffeFormated\train\fear"
# input_list = sorted(glob.glob(os.path.join(input_folder, '*')))
# output_list = sorted(glob.glob(os.path.join(result_folder, '*')))


# def imread(img_path):
#   img = cv2.imread(img_path)
#   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#   img = cv2.resize(img, (64,64))
#   return img



# input_sample = []

# output_sample = []

# # take pictures to two separated lists ( or it has to be arrays?)
# for i in range(6):
#   img_input = imread(input_list[i])
#   img_output = imread(output_list[i])
#   input_sample.append(img_input)
#   output_sample.append(img_output)

# n = len(input_sample)

# fig, axes = plt.subplots(2, n, figsize=(30,10))

# for i in range(n):
#     # Originals (row 0)
#     axes[0, i].imshow(input_sample[i], cmap="gray")
#     axes[0, i].set_title(f"Original {i+1}")
#     axes[0, i].axis("on")

#     # Restored (row 1)
#     axes[1, i].imshow(output_sample[i], cmap="gray")
#     axes[1, i].set_title(f"Restored 2x {i+1}")
#     axes[1, i].axis("on")

# plt.tight_layout()
# plt.show()




# =======================================================


"""
# Include download steps directly after inference
!ls results/train/anger # List the contents of the output directory for verification
print('Download results')
import os
from google.colab import files

os.system('zip -r download.zip results')
files.download("download.zip")
"""