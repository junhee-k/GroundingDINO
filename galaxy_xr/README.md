# Galaxy XR Test Images

This directory contains sample images for testing GroundingDINO with Galaxy XR HEIF/HEIC format images.

## Installation

After cloning the repository and checking out the galaxy-xr-custom branch:

1. Create the conda environment:
   ```bash
   conda env create -f environment_linux.yaml
   conda activate dino
   ```

2. Install GroundingDINO from source:
   ```bash
   pip install -e .
   ```

3. Install HEIF support:
   ```bash
   pip install pillow-heif
   ```

## Contents
- Sample HEIC images from Galaxy XR devices
- Test images for object detection with GroundingDINO

## Usage
Use with the modified `demo/inference_on_a_image.py` which includes HEIF support via `pillow_heif`.

## Requirements
- pillow_heif (install via: `pip install pillow-heif`)
