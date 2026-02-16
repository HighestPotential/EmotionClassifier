
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
import shutil
import subprocess
import matplotlib.pyplot as plt



# # Process the uploaded files
# for filename in input_image_folder:
#     # If not a zip, assume it's an image and move it to the input folder
#     dst_path = os.path.join(input_image_folder, filename)
#     print(f'Moving {filename} to {dst_path}')
#     shutil.move(filename, dst_path)

# print(f"Images prepared in: {input_image_folder}")



# =======================================



# Now we use the GFPGAN to restore the above low-quality images
# We use [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for enhancing the background (non-face) regions
# You can find the different models in https://github.com/TencentARC/GFPGAN#european_castle-model-zoo
import subprocess
from pathlib import Path

# -------------------- CONFIG --------------------
GFPGAN_DIR = Path("/home/k/kienzlehagen")  # Folder where inference_gfpgan.py lives
DATASETS_ROOT = Path("/home/k/kienzlehagen/version_3/latest_3_0_ready_to_use_datasets")
RESULTS_ROOT = GFPGAN_DIR / "resultsnew"

# -------------------- WALK DATASET FOLDERS --------------------
# Only process the **last-level subfolders** (6 subfolders inside the 3rd-level folder)
for last_level_folder in DATASETS_ROOT.rglob("*/*/*/*"):  # 4 levels down: dataset/split/class/leaf
    if not last_level_folder.is_dir():
        continue

    # Get all image files in this folder
    image_files = [f for f in last_level_folder.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    if not image_files:
        continue  # skip empty folders

    for img_path in image_files:
        # Determine output path
        relative_path = img_path.relative_to(DATASETS_ROOT)
        output_dir = RESULTS_ROOT / relative_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build GFPGAN command
        cmd = [
            "python",
            str(GFPGAN_DIR / "inference_gfpgan.py"),
            "-i", str(img_path),
            "-o", str(output_dir),
            "-v", "1.2",
            "-s", "2",
            "--aligned",
            "--bg_upsampler", "realesrgan"
        ]

        print(f"\nRunning GFPGAN on: {img_path}")
        print("Command:", " ".join(cmd))

        # Run inference with try/except
        try:
            subprocess.run(cmd, cwd=str(GFPGAN_DIR), check=True)
            print(f"✅ Finished: {img_path}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Inference failed for {img_path}. Skipping this image.")
            print("Error:", e)


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