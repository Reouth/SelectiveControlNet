# SelectiveControlNet


This project implements selective image transformation using [ControlNet](https://github.com/lllyasviel/ControlNet), where transformations are guided by spatial masks and user prompts. The goal is to selectively alter specific regions of an image (e.g., foreground or background) without affecting others, using pretrained models only.

---

## Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GIp7hUOhyuvrrk1tLNIqXPb7KFIDK9Fu?usp=sharing)

---

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/Reouth/SelectiveControlNet.git
cd SelectiveControlNet
pip install -r requirements.txt
```
---
## Usage (Colab Notebook)

The notebook is split into two parts:

**Part 1**: Foreground-Only Transformation
Uses rembg to mask the subject

* Applies a single prompt only to the foreground

**Part 2**: Dual Prompt (Foreground + Background)
Separates foreground and background with masks

* Generates each region using a different prompt
* Merges results via alpha blending

---
## Folder Structure

``` bash
SelectiveControlNet/
│
├── notebooks/
│   └── SelectiveControlNet.ipynb        # Main notebook
│   └── SelectiveVideoControlnet.ipynb   # Bonus notebook
│
├── src/
│   ├── masking.py                      # Foreground/background masking and normalization
│   ├── image_utils.py                  # Image loading, resizing, utilities
│   ├── control_pipeline.py             # ControlNet setup and inference functions
│   ├── merge.py                        # Output blending and validation
│
├── config.py                           
├── requirements.txt
└── README.md

```
---

## Answers to Assignment Questions

### Q1 – Approach and Key Factors
For Part 1, I used rembg to extract a binary foreground mask, applied it to the input image, and passed the masked result through a lineart detector to generate the ControlNet conditioning. The blacked-out background produced a near-zero signal in the control image, limiting guidance to the foreground.

This approach was chosen because:

* It leverages ControlNet’s spatial conditioning as intended

* The masked lineart preserves subject structure while ignoring the background

* It works with the pretrained diffusers pipeline without model modifications

For Part 2, I reused the same lineart control image (from the foreground-masked input), but created two masked versions of the image: one for the foreground, one for the background. I applied separate prompts, generated two outputs, and merged the results using the original mask.

Chosen because:

* Enables prompt-specific generation for different regions

* Shared control ensures structural consistency

* Works cleanly in the same pipeline
* Avoids architectural changes via modular post-processing

---
### Q2 – Challenges
The main challenge was simulating region-specific generation without model support for prompt-region mapping or binary masks.

In Part 1, lineart control localized the effect, but lacked texture/color, leading to altered skin tone or facial details. I tested StableDiffusionControlNetImg2ImgPipeline with the masked image and lineart, which preserved identity better, but failed to generate any background due to how ControlNet interprets black/noisy regions literally.

In Part 2, using two prompts and images enabled semantic separation, but caused shading mismatches and compositing artifacts even with blending.

Technical limitations:

* **No mask-based conditioning**: ControlNet doesn't accept binary masks to restrict spatial influence, so masking must be handled indirectly. 
* **No region-specific prompt control**: Prompts affect the full image — there's no native way to assign different tokens to different regions. 
* **Masked lineart may cause edge artifacts**: Using masked inputs to generate lineart can produce artifacts near transitions, which affect generation quality. 
* **Separate passes for each region**: Multi-region control requires multiple inference runs and manual merging, increasing runtime and complexity. 
* **No native compositing/fusion support**: The pipeline lacks tools for blending or combining outputs, especially in latent space.
---
### Q3 – Improvements With More Time or Resources
1. **Pre- and Post-Processing**
   * **Use segmentation-based masks**: Replace rembg with SAM or DeepLab for more accurate and semantically meaningful region separation
   * **Apply soft edge blending**: Dilate and feather mask edges to smooth transitions and reduce compositing artifacts 
   * **Improve prompt design**: Use clearer phrasing or apply token weighting to better control visual emphasis in specific regions

2. **Architectural Enhancements (no retraining)**
   * **Latent space fusion**: Merge foreground and background in latent space instead of pixel space for improved lighting and shading coherence
   * **Attention-based prompt routing**: Use spatial masks to guide attention layers, so that different prompts apply to different regions without retraining 
   * **Multi-control guidance**: Stack or interleave multiple control images (e.g. lineart for foreground, depth for background) with masks to enable region-specific conditioning
3. **Training-Based Approaches** 
   * **Mask-aware ControlNet**: Train a ControlNet variant that explicitly takes both a control image and binary mask as inputs for true spatial control 
   * **Prompt-to-region supervision**: Train models to map parts of the prompt to image regions using attention or segmentation alignment 
   * **Video extension with temporal control**: Adapt the pipeline for consistent multi-frame generation using temporal-aware diffusion models

---
### Bonus Challenge: Consistent Transformation Across Video Frames
#### Approach and Reasoning

To address the bonus challenge of generating two consistently transformed frames using a single prompt, I used the Control-A-Video pipeline.
I extracted two frames from a video and applied the same preprocessing as in Part 1: a binary foreground mask (via rembg) was used to remove the background.
The masked frames were then converted into a two-frame .mp4 video and passed into the model using --control_mode canny.

Chosen because:
* It maintains consistency with the masked input format used earlier.
* Control-A-Video handles temporal coherence across frames. 
* It works with the existing pipeline using a single shared prompt and no architectural changes.

#### Challenges

The main challenge was adapting a multi-frame video model to run on just two frames. This required formatting the frames into a proper .mp4 and tuning parameters like --num_sample_frames=2 and --each_sample_frame=2.
An additional challenge was environment compatibility. conflicting versions of huggingface_hub, jax, and diffusers caused runtime issues, which I resolved by pinning specific package versions during setup.

####  Improvements With More Time or Resources

* Add support for lineart control in the video pipeline to match Part 1 more closely.
* Extend to longer sequences with more advanced temporal smoothing.
* Experiment with latent-space fusion to improve coherence across frame boundaries.