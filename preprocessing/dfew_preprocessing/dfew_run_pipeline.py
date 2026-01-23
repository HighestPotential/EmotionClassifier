from pathlib import Path
from preprocessing import dataset_runner
from preprocessing.pipeline import Pipeline
from preprocessing.lighting_filter import LightingFilter
from preprocessing.blurred_images import RemoveBlurredFaces
from preprocessing.resizing_64 import ResizingTo64
from preprocessing.to_rgb import ToRGB

def main():
    project_root = Path(__file__).resolve().parents[2]
    dfew_stage1 = Path(__file__).resolve().parent / "dfew_stage1"
    out_root = project_root / "preprocessed_dataset" / "DFEW"

    steps = [
        LightingFilter(),
        RemoveBlurredFaces(),
        ResizingTo64(),
        ToRGB(),
    ]

    pipeline = Pipeline(steps)

    dataset_runner.run_folder(
        pipeline,
        str(dfew_stage1),
        str(out_root),
        keep_structure=True,
        max_files=None,
        log_every=100,
    )

    print("Input :", dfew_stage1)
    print("Output:", out_root)

if __name__ == "__main__":
    main()