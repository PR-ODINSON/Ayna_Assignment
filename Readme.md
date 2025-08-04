# Polygon Colorization using Conditional UNet
### Author: Prithviraj Verma

---

## Problem Statement

Design and train a **UNet** model to generate a polygon image filled with a specified color. The model should take as input:
- A **grayscale polygon image** (shape: triangle, square, etc.)
- A **color name** (e.g., "blue", "red", "yellow")

And output a **colored polygon image** where the shape is filled using the specified color.

---

## Dataset Structure

The dataset follows the structure below:

```
dataset/
│
├── training/
│   ├── inputs/         # Grayscale polygon images
│   ├── outputs/        # Colored polygon images (ground truth)
│   └── data.json       # Maps input → color → output
│
├── validation/
│   ├── inputs/
│   ├── outputs/
│   └── data.json
```

Each JSON file contains mappings like:
```json
{
  "input_polygon": "triangle_001.png",
  "colour": "blue",
  "output_image": "triangle_001_colored.png"
}
```

---

## Model Architecture

- **Model Used:** `UNet` (from scratch)
- **Input Channels:** `4`  
  - 1 channel for grayscale polygon image  
  - 3 channels for one-hot encoded color tensor
- **Output Channels:** `3` (RGB)
- **Skip Connections:** Yes (typical UNet structure)
- **Dropout:** Used in double convolution blocks for regularization

### Architecture Flow:

```
[Grayscale Image (1xHxW)] + [Color Tensor (3xHxW)] → concat → 4xHxW  
→ Encoder (3 blocks: 64→128→256)  
→ Bottleneck: 512  
→ Decoder with skip connections  
→ Final Conv: 3-channel RGB Output  
```

---

## Preprocessing Steps

1. **Image Conversion:**
   - Input: Converted to grayscale (`L`) → 1 channel
   - Output: Converted to RGB → 3 channels

2. **Color Conditioning:**
   - A dictionary maps color names to RGB vectors.
   - The RGB vector is converted to a tensor and **expanded** to shape `(3, H, W)`.
   - This tensor is **concatenated** with the grayscale polygon → `4-channel` input.

3. **Tensor Final Shape:**
   ```python
   input_tensor = torch.cat([gray_img_tensor, color_tensor], dim=0)  # Shape: [4, H, W]
   ```

---

## Training Setup

| Parameter       | Value              |
|----------------|--------------------|
| Model           | UNet               |
| Input Channels  | 4                  |
| Output Channels | 3 (RGB)            |
| Loss Function   | L1 Loss            |
| Optimizer       | Adam (lr = 1e-3)   |
| Scheduler       | StepLR (γ=0.5, step_size=10) |
| Epochs          | 20                 |
| Batch Size      | 8                  |
| Device          | GPU (if available) |

---

## Data Augmentation

- **Horizontal Flip**
- **Vertical Flip**
- Implemented via `torchvision.transforms.Compose`

---

## Tracking Tools

**wandb (Weights & Biases)**

- Project: `ayna-polygon-color`
- Run Name: `unet-augmented`
- Tracked:
  - `train_loss` and `val_loss` per epoch
  - Logs, metrics, system usage
  - All visualizations and comparisons

Code snippet:
```python
wandb.init(project="ayna-polygon-color", name="unet-augmented")
wandb.log({"train_loss": train_loss, "val_loss": val_loss})
```

---

## Training Logs

The model was trained for **20 epochs**.  
Trends observed:
- **Steady convergence** of training and validation loss.
- Scheduler effectively reduced learning rate every 10 epochs.
- No overfitting noticed due to effective regularization (Dropout + Augmentations).

---

## Sample Inference Explanation

Each inference includes:
- Input: Grayscale polygon + color name
- Output: RGB-colored polygon

In `visualize(model, dataloader)`:
- `subplot[0]`: Raw grayscale polygon
- `subplot[1]`: Ground truth colored image
- `subplot[2]`: Model prediction (RGB)

Visuals indicate the model learns sharp edges and accurate color fills.

---

## Failure Modes Observed

- **Edge bleeding:** Colors slightly overflowing outside shape in early epochs.
- **Incorrect fill:** Rarely, model fills partial area inside the polygon.
- **Color confusion:** Occurred with similar tones like *cyan* vs *blue*.

---

## Fixes Implemented

- Added **Dropout** layers in all convolution blocks to reduce overfitting.
- Used **color expansion (3xHxW)** instead of naive one-hot to match shape.
- Applied **data augmentation** for better generalization.
- Tuned learning rate schedule and batch size for stability.

---

## Possible Future Improvements

- Integrate **Dice Loss or perceptual loss** for better boundary accuracy.
- Use **Vision Transformer blocks** or attention modules for refinement.
- Introduce **color similarity embeddings** for improved conditioning.
- Train on **higher-resolution images** for real-world deployment.
- Perform **multi-color fill** conditioning for complex patterns.

---

## Key Learnings from the Project

- How to condition a UNet model using both image and semantic (text) inputs.
- Importance of properly reshaping and expanding tensors for fusion.
- Utility of experiment tracking with wandb in debugging ML models.
- The value of regularization and data augmentation in small datasets.

---

## Submission Deliverables

| Item | Description |
|------|-------------|
| Jupyter Notebook | For model inference and visualization |
| README.md        | This file |
| wandb Project    | [Click to view](https://wandb.ai/) *(replace with actual link)* |

---
