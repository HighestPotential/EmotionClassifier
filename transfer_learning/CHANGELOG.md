# Changes to `ir50.py`

## 1. Restored 4th Stage (Layers 124-152 equivalent)
**Why:** The original file had the 4th stage of the ResNet architecture commented out. This stage is crucial as it contains the most semantic, high-level features learned from the large-scale face dataset.
**Change:** 
- Updated `get_blocks` to return the full list of 4 block definitions for ResNet-50.
- Updated `Backbone.__init__` to instantiate `self.body4` using these blocks.
- Updated `Backbone.forward` to pass data through `self.body4`.

## 2. Added Classification Head for 6 Emotions
**Why:** The user requested the model to output exactly 6 classes (Anger, Disgust, Fear, Happiness, Sadness, Surprise) instead of the original thousands of identities or just raw features.
**Change:** 
- Modified `Backbone.__init__` to accept a `num_classes` argument (defaulting to 6).
- Extended `self.output_layer`. The original ended at a 512-dimensional embedding (BatchNorm1d).
- Added:
    - `ReLU` activation.
    - `Linear(512, num_classes)`: This projects the 512 feature vector down to 6 logits.
- **Result:** The model now outputs a tensor of shape `(batch_size, 6)`, ready for `CrossEntropyLoss`.

## 3. Forward Pass Update
**Why:** To connect the new layers and return the predictions.
**Change:** 
- `forward` now flows through `body1 -> body2 -> body3 -> body4 -> output_layer`.
- Returns the final logits.

## 4. Weight Loading Compatibility
**Note:** When loading the pretrained `ir50.pth` (which has ~8000 classes), the new `Backbone` has a generic `output_layer` that technically includes `Linear(512, 6)`.
- The pretrained weights for the *backbone* (bodies 1-4) and the *embedding layer* (Linear 512*7*7 -> 512) will match and load correctly.
- The final `Linear(512, 6)` layer is **new** and initialized randomly. It must be trained (fine-tuned).
- The `load_pretrained_weights` function handles this by ignoring keys that don't match (i.e., the old classifier weights won't overlap with our new 6-class layer).
