import torch
import pandas as pd
import pdb
import json
import csv
import argparse
from torch import nn
from transformers import AutoModelForCausalLM

from navila_utils import VLMServer


RANK = 500  # Number of components to keep for nudging
# IGNORE_SPECIFIC_LANGUAGE_LAYERS = True
# LANGUAGE_LAYERS_TO_IGNORE = list(range(0, 16))
CHOOSE_SINGULAR_VALUES_BY = 'Magnitude' # 'Magnitude' or 'Random'
TOTALLY_REPLACE_PRUNED_WEIGHTS = False  # Whether to replace pruned weights with nudged weights or add them
SAVE_SINGULAR_VALUES_SEPARATELY = False  # Whether to save singular values separately
SAVE_SINGULAR_VALUES_TO_CSV = False 
SAVE_RANDOM_INDICES = False
SWAP_WEIGHTS = True                      # Place Specific Dense Layers into Pruned Model


SVD_FACTORS = {}  # Dictionary to hold singular value factors for each layer
SKIP_LAYERS = ['lm_head', "projector", "vision_tower"]  # Layers to skip during nudging


if SAVE_SINGULAR_VALUES_TO_CSV:
    # CSV setup
    csv_path = "singular_values.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["layer","singular_values"])
        writer.writeheader()
elif SAVE_RANDOM_INDICES:
    # CSV setup for random indices
    csv_path = "singular_values_chosen_indices.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["layer", "chosen_indices"])
        writer.writeheader()

def nudge_and_save(pruned_model, dense_model, save_dir='pruned_model_nudged', tau=0.8, device='cuda'):
    global SVD_FACTORS

    # Move models to appropriate devices
    pruned_model.eval()
    dense_sd = dense_model.state_dict()

    device_gpu = torch.device(device)
    device_cpu = torch.device('cpu')

    # Perform patch under no_grad to avoid in-place grad errors
    with torch.no_grad():
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear): 

                if any(skip_layer in name for skip_layer in SKIP_LAYERS):
                    continue

                # if IGNORE_SPECIFIC_LANGUAGE_LAYERS: 
                #     if "language_model" in name:
                #         layer_id = int(name.split("language_model.model.layers.")[1].split(".")[0])
                #         if layer_id in LANGUAGE_LAYERS_TO_IGNORE:
                #             continue
                #         else:
                #             pass 
                # else:
                #     pass

                # Move ONLY this submodule to GPU  
                module.to(device_gpu)

                Wp = module.weight
                Wd = dense_sd[name + ".weight"].to(device_gpu)

                if SWAP_WEIGHTS:
                    dense_layers_to_place_into_pruned = ["." + str(i) + "." for i in range(0,6)]
                    if any(swap_layer in name for swap_layer in dense_layers_to_place_into_pruned):
                        module.weight.data = Wd
                        print(f"Dense {name} placed into pruned model.")
                    continue

                if not TOTALLY_REPLACE_PRUNED_WEIGHTS:
                    print(f"Checking Safety Gap for {name}")
                    # Compute the gap
                    V = Wd - Wp
                else:
                    V = Wd

                # Convert to float64 for SVD computation
                V = V.to(torch.float64)

                # SVD: get top singular component
                U, S, Vh = torch.linalg.svd(V)
                print("SVD singular values:", S)
                r = RANK  # number of components to keep, must be << d

                if CHOOSE_SINGULAR_VALUES_BY == 'Random':
                    n_vals = S.size(0)
                    # pick r random, non-repeating indices
                    rand_idx = torch.randperm(n_vals, device=S.device)[:r]

                    # extract those components
                    U_r  = U[:, rand_idx]        # (d_out × r)
                    S_r  = S[rand_idx]           # (r,)
                    Vh_r = Vh[rand_idx, :]       # (r × d_in)

                elif CHOOSE_SINGULAR_VALUES_BY == 'Magnitude':
                    # Slice off the top-r singular triplets
                    U_r   = U[:, :r]        # (d_out × r)
                    S_r   = S[:r]           # (r,)
                    Vh_r  = Vh[:r, :]       # (r × d_in)
                else:
                    raise ValueError("CHOOSE_SINGULAR_VALUES_BY must be 'Magnitude' or 'Random'")

                # Sum up each rank-1 piece
                delta_W = sum(
                    S_r[i] * torch.ger(U_r[:, i], Vh_r[i, :])
                    for i in range(r)
                )

                # print(f"The SVD", delta_W[:8, :8])  # Print the top-left 8x8 block of delta_W
                # print(f"The Wp", Wp[:8, :8])  # Print the top-left 8x8 block of Wp

                # Rank-1 patch
                # delta_W = sigma1 * torch.ger(u1, v1)
                
                if SAVE_SINGULAR_VALUES_SEPARATELY:

                    U_scaled = U_r * S_r.unsqueeze(0)  # (d_out, r)
                    U_T_r = U_scaled.t().contiguous()  # (r, d_out)
                    V_r   = Vh_r.t().contiguous()      # (d_in, r)

                    # Save low-rank factors without changing weight
                    SVD_FACTORS[name] = {
                        "V": V_r.cpu(),
                        "U_T": U_T_r.cpu()
                    }

                    del U_scaled, U_T_r, V_r

                elif not TOTALLY_REPLACE_PRUNED_WEIGHTS:
                    print(f" -> Adding nudged weights to pruned weights for {name}")
                    # Update weight in-place on its .data buffer
                    module.weight.data.add_(delta_W)

                    # mask = (Wp != 0).to(Wp.dtype)            # 1s where Wp is nonzero, 0s elsewhere
                    # delta_masked = delta_W * mask            # zero out all entries outside the original support

                    # # print(f"Mask", delta_masked[:8, :8])  # Print the top-left 8x8 block of delta_masked
                    # module.weight.data.add_(delta_masked)    # now W_new = Wp + Δ only on the mask


                else:
                    print(f" -> Replacing pruned weights with SVD weights for {name}")
                    module.weight.data = delta_W


                # Move this submodule back to CPU
                module.to(device_cpu)

                if SAVE_SINGULAR_VALUES_TO_CSV:
                    # append to CSV
                    with open(csv_path, "a", newline="") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=["layer","singular_values"])
                        writer.writerow({
                            "layer": name,
                            "singular_values": json.dumps(S.tolist())
                        })
                elif SAVE_RANDOM_INDICES:
                    # save the random indices we picked
                    with open(csv_path, "a", newline="") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=["layer", "chosen_indices"])
                        writer.writerow({
                            "layer": name,
                            "chosen_indices": json.dumps(rand_idx.tolist())
                        })
    
    if SAVE_SINGULAR_VALUES_SEPARATELY:
        torch.save(SVD_FACTORS, f"svd_factors_rank_{RANK}.pt")
    else:
        # Save the patched model
        pruned_model.save_pretrained(save_dir)
        print(f"Patched model saved to {save_dir}")


parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default='localhost', help="Host to bind the server")
parser.add_argument("--port", type=int, default=54321, help="Port to bind the server")
parser.add_argument("--model_path", type=str, default="", help="Path to the model checkpoint")
parser.add_argument("--precision", type=str, default="W16A16", help="compute precision")
parser.add_argument("--conv_mode", type=str, default="llama_3")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--num_video_frames", type=int, default=8)
args = parser.parse_args()

print("[*] Loading pruned NaVILA model...")
pruned_model_path = "/home/ubuntu/pruning_vlas/models/navila_models/NaVILA-pruned-2_4-Wanda-language_backbone"
args.model_path = pruned_model_path
navila_server = VLMServer(args)
pruned_model = navila_server.model
pruned_model = pruned_model.to("cpu")
pruned_model = pruned_model.to(torch.float16)


print("[*] Loading dense NaVILA model...")
dense_model = "/home/ubuntu/pruning_vlas/models/navila_models/navila-llama3-8b-8f"
# Update the args path 
args.model_path = dense_model
navila_server = VLMServer(args)
dense_model = navila_server.model
dense_model = dense_model.to("cpu")
dense_model = dense_model.to(torch.float16)

nudge_and_save(pruned_model, dense_model, device='cuda')

pdb.set_trace()

