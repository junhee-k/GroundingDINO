import argparse
import os
import sys
import time
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

# Support for Galaxy XR HEIF/HEIC images
import pillow_heif
pillow_heif.register_heif_opener()

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_path, input_resolution):
    """
    1. Simulates taking the photo at 'input_resolution'.
    2. Applies standard GroundingDINO preprocessing (Resize to 800/1333).
    """
    image_pil = Image.open(image_path).convert("RGB") 

    # 1. Simulate source resolution (Simulating the camera capture size)
    image_pil = image_pil.resize((input_resolution, input_resolution), resample=Image.BICUBIC)

    # 2. Standard GroundingDINO Preprocessing
    # Resizes short edge to 800, max long edge to 1333.
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(f"Model loaded: {load_res}")
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshold and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    
    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                # Appending confidence score to the phrase string
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # (Handling for token_spans if provided)
        positive_maps = create_positive_map_from_span(
            model.tokenizer(caption),
            token_span=token_spans
        ).to(image.device) 

        logits_for_phrases = positive_maps @ logits.T 
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            filt_mask = logit_phr > box_threshold
            all_boxes.append(boxes[filt_mask])
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases

    return boxes_filt, pred_phrases


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO Performance Test", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--image_path", "-i", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
    
    # Defaults for Galaxy XR testing
    parser.add_argument("--test_sizes", type=str, default="2560,1280,640,320", 
                        help="Comma separated list of resolutions to test. Defaults to '2560,1280,640,320'.")

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--token_spans", type=str, default=None, help="Token spans for specific phrase detection")
    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    
    args = parser.parse_args()

    # Parse args
    sizes_to_test = [int(x) for x in args.test_sizes.split(",")]
    config_file = args.config_file
    checkpoint_path = args.checkpoint_path
    image_path = args.image_path
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = args.token_spans

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    if token_spans is not None:
        text_threshold = None
        print("Using token_spans. Set the text_threshold to None.")

    # --- WARMUP PHASE ---
    print(f"\n{'='*20} WARMUP PHASE {'='*20}")
    print("Running a dummy inference to initialize CUDA and load kernels...")
    try:
        # Load a dummy image at standard size for warmup
        warmup_img_pil, warmup_img = load_image(image_path, input_resolution=800)
        _ = get_grounding_output(
            model, warmup_img, text_prompt, box_threshold, text_threshold, 
            cpu_only=args.cpu_only, token_spans=eval(f"{token_spans}")
        )
        _ = get_grounding_output(
            model, warmup_img, text_prompt, box_threshold, text_threshold, 
            cpu_only=args.cpu_only, token_spans=eval(f"{token_spans}")
        )
        _ = get_grounding_output(
            model, warmup_img, text_prompt, box_threshold, text_threshold, 
            cpu_only=args.cpu_only, token_spans=eval(f"{token_spans}")
        )
        if not args.cpu_only:
            torch.cuda.synchronize()
        print("Warmup complete.\n")
    except Exception as e:
        print(f"Warmup failed (non-fatal, but timing might be off): {e}\n")


    # --- BENCHMARKING LOOP ---
    benchmark_results = []

    print(f"{'='*60}")
    print(f"Starting Performance Test on Image: {os.path.basename(image_path)}")
    print(f"Simulating Capture Resolutions: {sizes_to_test}")
    print(f"{'='*60}\n")

    for size in sizes_to_test:
        print(f"--- Simulating Capture Resolution: {size}x{size} ---")
        
        try:
            # Load, Simulate Resolution, and Apply Standard Transform
            image_pil, image = load_image(image_path, input_resolution=size)
            
            # Print tensor shape to confirm standard preprocessing is working (should be ~800x800)
            c, h, w = image.shape
            
            # Start Timer
            if not args.cpu_only:
                torch.cuda.synchronize()
            start = time.time()
            
            # Run Inference
            boxes_filt, pred_phrases = get_grounding_output(
                model, image, text_prompt, box_threshold, text_threshold, 
                cpu_only=args.cpu_only, token_spans=eval(f"{token_spans}")
            )
            
            # Stop Timer
            if not args.cpu_only:
                torch.cuda.synchronize()
            end = time.time()
            latency = end - start
            
            print(f"Inference Latency: {latency:.4f}s")
            print(f"Detected {len(pred_phrases)} phrases: {pred_phrases}")
            
            benchmark_results.append({
                "size": size,
                "tensor_dims": f"{h}x{w}",
                "latency": latency,
                "phrases": pred_phrases, # List of strings like "dog(0.95)"
                "error": None
            })

            # Visualization
            pred_dict = {
                "boxes": boxes_filt,
                "size": [image_pil.size[1], image_pil.size[0]],  # H,W
                "labels": pred_phrases,
            }
            
            viz_image = image_pil.copy()
            image_with_box = plot_boxes_to_image(viz_image, pred_dict)[0]
            save_name = f"pred_source_{size}.jpg"
            image_with_box.save(os.path.join(output_dir, save_name))

        except Exception as e:
            print(f"Error at {size}: {e}")
            benchmark_results.append({
                "size": size,
                "tensor_dims": "N/A",
                "latency": -1,
                "phrases": [],
                "error": str(e)
            })

    # --- FINAL SUMMARY TABLE ---
    print(f"\n\n{'='*30} FINAL SUMMARY {'='*30}")
    # Header
    print(f"{'Resolution':<12} | {'Latency':<10} | {'Count':<6} | {'Detected Phrases & Confidence'}")
    print("-" * 100)
    
    for res in benchmark_results:
        if res['error']:
            status = "FAILED"
            latency_str = "N/A"
            phrases_str = f"Error: {res['error']}"
        else:
            status = "Success"
            latency_str = f"{res['latency']:.4f}s"
            # Join all phrases into a comma-separated string
            res['phrases'] = sorted(res['phrases'])
            phrases_str = ", ".join(res['phrases'])
            
        print(f"{res['size']:<12} | {latency_str:<10} | {len(res['phrases']):<6} | {phrases_str}")
        
    print(f"{'='*80}\n")