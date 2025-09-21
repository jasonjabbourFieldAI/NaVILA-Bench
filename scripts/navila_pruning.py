# ---- Monkey Patch for SparseGPT ---------
import transformers.modeling_utils as modeling_utils

def safe_unwrap_model(model, *args, **kwargs):
    return model  # skip DeepSpeed unwrap

modeling_utils.unwrap_model = safe_unwrap_model
# -----------------------------------------

from llmcompressor.modifiers.pruning import WandaPruningModifier, MagnitudePruningModifier
from llmcompressor.modifiers.obcq import SparseGPTModifier
from llmcompressor import oneshot

import torch
import argparse

from transformers import AutoModelForCausalLM
from transformers.data import default_data_collator
from transformers import AutoProcessor

import torch
from PIL import Image
import io
import os
import random
from transformers import AutoModelForCausalLM
import pdb
from navila_utils import VLMServer


if not hasattr(torch, "OutOfMemoryError"):
    class _OOM(RuntimeError): pass
    torch.OutOfMemoryError = _OOM

from datasets import IterableDataset
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from tfrecord.reader import tfrecord_loader
import base64



PRETRAINED_CHECKPOINT = f"/home/ubuntu/pruning_vlas/models/navila_models/navila-llama3-8b-8f"
TFRECORD_DIR = f"/home/ubuntu/pruning_vlas/data/navila_data"

PRUNING_MODIFIER = "Wanda"  # ["Wanda", "Magnitude", or "SparseGPT"]

# Only one of these should be True at a time
PRUNE_VISION_BACKBONE = False
PRUNE_LANGUAGE_MODEL = False
PRUNE_FULL_MODEL = True

IGNORE_SPECIFIC_LANGUAGE_LAYERS = False 
# Half are: list(range(0, 16)) or list(range(16, 32))
LANGUAGE_LAYERS_TO_IGNORE = list(range(16, 32))
NUM_CALIB_SAMPLES = 10000

assert sum([PRUNE_VISION_BACKBONE, PRUNE_LANGUAGE_MODEL, PRUNE_FULL_MODEL]) == 1, \
    "Only one of PRUNE_* flags can be True at a time."

# Determine what parts to prune
if PRUNE_FULL_MODEL:
    ignore = ["re:^llm\\.lm_head\\."] # or None
elif PRUNE_VISION_BACKBONE:
    ignore = [
        "re:^llm\\.",
        "re:^projector\\.",
        "re:^llm\\.lm_head\\."
    ]
elif PRUNE_LANGUAGE_MODEL:
    ignore = [
        "re:^vision_tower\\.",
        "re:^projector\\.",
        "re:^llm\\.lm_head\\."
    ]
else:
    raise ValueError("No pruning target selected!")

if IGNORE_SPECIFIC_LANGUAGE_LAYERS:
    ignore += [f"re:^llm\\.model\\.layers\\.{i}\\." for i in LANGUAGE_LAYERS_TO_IGNORE]

if not hasattr(torch, "OutOfMemoryError"):
    class _OOM(RuntimeError): pass
    torch.OutOfMemoryError = _OOM


def decode_jpeg(jpeg_bytes):
    """Decode JPEG bytes into RGB image as np.uint8 array."""
    image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    return image

def build_calibration_dataset_from_examples(
    tfrecord_paths,
    device,
    navila_server,
    num_samples,
    seed=42,
):
    calibration_set = []
    rng = random.Random(seed)
    paths = list(tfrecord_paths)
    rng.shuffle(paths)

    description = {
        "episode_id": "int",
        "step_id": "int",
        "language_instruction": "byte",
        "images": "byte",
    }

    for tfrecord_path in paths:
        print(f"[*] Scanning: {tfrecord_path}")
        for record in tfrecord_loader(tfrecord_path, None, description):
            if len(calibration_set) >= num_samples:
                break

            try:
                instruction = record["language_instruction"].decode()
                img_bytes_list = record["images"]

                # ensure list
                if isinstance(img_bytes_list, (bytes, bytearray)):
                    img_bytes_list = [img_bytes_list]

                # decode images
                images = [decode_jpeg(img_bytes) for img_bytes in img_bytes_list]

                input_ids, image_tensor, _ = navila_server.get_navila_inputs(images, instruction)

                calibration_set.append({
                    "input_ids": input_ids,
                    "images": image_tensor,
                })


            except Exception as e:
                print(f"[!] Skipping record due to error: {e}")
                continue

        if len(calibration_set) >= num_samples:
            break

    print(f"[✓] Collected {len(calibration_set)} calibration examples.")
    return calibration_set



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='localhost', help="Host to bind the server")
    parser.add_argument("--port", type=int, default=54321, help="Port to bind the server")
    parser.add_argument("--model_path", type=str, default=PRETRAINED_CHECKPOINT, help="Path to the model checkpoint")
    parser.add_argument("--precision", type=str, default="W16A16", help="compute precision")
    parser.add_argument("--conv_mode", type=str, default="llama_3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_video_frames", type=int, default=8)
    args = parser.parse_args()

    print("[*] Loading full model...")
    navila_server = VLMServer(args)
    model = navila_server.model

    # After loading NaViLA model
    if not hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    tfrecord_paths = [
        os.path.join(TFRECORD_DIR, f)
        for f in sorted(os.listdir(TFRECORD_DIR))
        if ".tfrecord" in f
    ]

    calib_data = build_calibration_dataset_from_examples(
        tfrecord_paths=tfrecord_paths,
        device=model.device,
        num_samples=NUM_CALIB_SAMPLES,
        navila_server=navila_server,
    )

    print("Number of Calibration Samples", len(calib_data))

    ds = Dataset.from_list(calib_data).with_format("torch")
    del calib_data
    print(len(ds), ds.column_names)        # sanity-check
    
    # Create the pruner
    if PRUNING_MODIFIER == "Wanda":
        pruner = WandaPruningModifier(
            targets="Linear",
            sparsity=0.5,
            sequential_targets=["llm.model.layers", "vision_tower", "mm_projector"],  # For navila
            mask_structure="2:4",
            ignore=ignore,
        )
    elif PRUNING_MODIFIER == "Magnitude":
        pruner = MagnitudePruningModifier(
            targets="Linear",
            init_sparsity=0.0,
            final_sparsity=0.5,
            mask_structure="unstructured",  # Does not support "2:4" mask structure
            ignore=ignore,
        )
    elif PRUNING_MODIFIER == "SparseGPT":
        pruner = SparseGPTModifier(
            targets="Linear",
            sparsity=0.5,
            sequential_targets=["llm", "vision_tower", "mm_projector"],
            mask_structure="2:4", 
            ignore=ignore,
        )
    else:
        raise ValueError(f"Unknown PRUNING_MODIFIER: {PRUNING_MODIFIER}")

    oneshot(model=model,
            recipe=pruner,
            processor=None,
            dataset=ds,
            num_calibration_samples=NUM_CALIB_SAMPLES,
            pipeline='basic',
            save_compressed=False,
            output_dir=None)

    if PRUNE_FULL_MODEL:
        pruned_scope = "full_model"
    elif PRUNE_VISION_BACKBONE:
        pruned_scope = "vision_backbone"
    elif PRUNE_LANGUAGE_MODEL:
        pruned_scope = "language_backbone"
    else:
        raise ValueError("No pruning target selected!")


    save_dir = f"NaVILA-pruned-2_4-{PRUNING_MODIFIER}-{pruned_scope}"

    if IGNORE_SPECIFIC_LANGUAGE_LAYERS:
        save_dir += f"-ignore-lang-layers-{min(LANGUAGE_LAYERS_TO_IGNORE)}-{max(LANGUAGE_LAYERS_TO_IGNORE)}"
    
    total_params = 0
    zero_params  = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            W = module.weight.data
            total_params += W.numel()
            zero_params  += (W == 0).sum().item()
    print(f"Global zero fraction: {zero_params/total_params:.3%}")


    # save in compressed form
    model.save_pretrained(
        save_dir,
        safe_serialization=True,          # safetensors
        # save_compressed=True,             # write bit-masks instead of dense tensors
        # compression_scheme="sparse-24-bitmask",   # this string matters!
        disable_sparse_compression=True,
    )

    pdb.set_trace()


#################################################3

# ## Sanity Check: View Architecture of Model

# ###############################################

# for name, mod in model.named_modules():
#     linear = False
#     if isinstance(mod, torch.nn.Linear):
#         linear = True
#     print(f"Layer: {name} Type: {type(mod)} Linear: {linear}")


# ###############################################

# ## Sanity Check Visualization of Pruned Weights

# ################################################
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap, BoundaryNorm

# # pick a layer, e.g. the first Linear in your language model
# # layer = model.language_model.model.layers[0].self_attn.q_proj 
# # layer = model.vision_backbone.featurizer.blocks[0].mlp.fc1
# layer = model.vision_backbone.featurizer.blocks[0].attn.qkv

# W = layer.weight.data.cpu().numpy()   # shape (out_dim, in_dim)
## W = layer.weight.data.to(torch.float32).cpu().numpy()

# # make a binary mask: 0 where W==0, 1 everywhere else
# binary = (W != 0).astype(int)

# # define a 2-color map: zeros in light gray, non-zeros in navy
# cmap = ListedColormap(["#FFFFFF", "#F15A29"])
# norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)  # boundaries at -0.5→0.5→1.5

# plt.figure(figsize=(8, 6))
# plt.imshow(binary, aspect="auto", cmap=cmap, norm=norm)
# plt.colorbar(ticks=[0, 1],label="Zero vs Non-Zero",format=plt.FuncFormatter(lambda val, loc: "zero" if val == 0 else "non-zero"))
# plt.title("Binary mask of Vision Backbone attn.qkv weight (layer 0)")
# plt.xlabel("Input dimension")
# plt.ylabel("Output dimension")

# # save high-res copy before showing
# plt.savefig("vision_backbone_attn_qkv.png", dpi=300, bbox_inches="tight")


# # --------- Zoomed In View of Matrix

# # slice the first 100×100
# W_small = W[:20, :20]

# # make a binary mask: 0 where W_small==0, 1 where !=0
# binary_small = (W_small != 0).astype(int)

# plt.figure(figsize=(6, 6))
# plt.imshow(binary_small, aspect="equal", cmap=cmap, norm=norm)
# plt.colorbar(ticks=[0, 1],label="Zero vs Non-Zero",format=plt.FuncFormatter(lambda val, loc: "zero" if val == 0 else "non-zero"))
# plt.title("Binary mask of Vision Backbone attn.qkv weight")
# plt.xlabel("Input dim (0–19)")
# plt.ylabel("Output dim (0–19)")

# # save hi-res
# plt.savefig("vision_backbone_attn_qkv_20x20.png", dpi=300, bbox_inches="tight")
