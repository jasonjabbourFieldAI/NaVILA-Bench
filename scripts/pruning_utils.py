import torch
from torch import nn
import torch.nn.functional as F
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
import gc

SparseSemiStructuredTensor._FORCE_CUTLASS = True


# class SparseSVDFusedWrap(nn.Module):
#     def __init__(self, linear, V, U_T):
#         super().__init__()
#         self.linear = linear
#         self.register_buffer("V", V.contiguous())
#         self.register_buffer("U_T", U_T.contiguous())

#     # @torch._dynamo.disable()
#     # def _clone_outside_compile(self, t: torch.Tensor) -> torch.Tensor:
#     #     return t.clone()
    
#     def forward(self, x):
#         y = F.linear(x, self.linear.weight, self.linear.bias)
#         tmp = (x @ self.V) @ self.U_T
#         return torch.add(y, tmp)  
#         # out = y + tmp
#         # return self._clone_outside_compile(out)
    

def quantize_tensor(t: torch.Tensor, num_bits=8):
    """
    Symmetric linear quantization of a tensor.
    Supports int8.
    """
    qmin = -(2**(num_bits-1))
    qmax = (2**(num_bits-1)) - 1

    # Compute scale (per tensor; can be per-channel for better accuracy)
    max_val = t.abs().max()
    scale = max_val / float(qmax)

    # Quantize to integers
    q = torch.clamp((t / scale).round(), qmin, qmax)

    q = q.to(torch.int8)

    return q, scale

class SparseSVDFusedWrap(nn.Module):
    def __init__(self, linear, V_q, V_scale, U_T_q, U_T_scale):
        super().__init__()
        self.linear = linear
        self.register_buffer("V_q", V_q.contiguous())
        self.register_buffer("U_T_q", U_T_q.contiguous())
        self.register_buffer("V_scale", torch.tensor(V_scale, dtype=torch.float32))
        self.register_buffer("U_T_scale", torch.tensor(U_T_scale, dtype=torch.float32))

    def forward(self, x):
        y = F.linear(x, self.linear.weight, self.linear.bias)

        # dequantize on the fly
        V = (self.V_q.float() * self.V_scale).to(x.dtype)
        U_T = (self.U_T_q.float() * self.U_T_scale).to(x.dtype)

        tmp = (x @ V) @ U_T
        # y.add_(tmp)
        return torch.add(y, tmp)  


def is_layer_pruning_enabled(module, layer_name, filter_for=None, skip_layers=None):
    """
    Check if layer pruning is enabled for a specific layer.

    filter_for: If provided, only check layers that contain this string in their name.
        Possible Values: 'language_model', 'vision_backbone', etc.
    skip_layers: If provided, skip layers that contain any of these strings in their name.
        Possible Values: ['vision_backbone', 'lm_head', 'projector']
    """
    if isinstance(module, nn.Linear): #and "layer" in fqn:
        if (filter_for is not None) and any(f in layer_name for f in filter_for):
            if skip_layers and any(skip_layer in layer_name for skip_layer in skip_layers):
                print("Skipping layer:", layer_name)
                return False
            if module.weight.shape[0] % 32 == 0 and module.weight.shape[1] % 64 == 0:
                return True
    return False


def attach_sparse_kernel(model, filter_for=None, skip_layers=None):

    print("[*] Attaching sparse kernel to model...")

    for layer_name, module in model.named_modules():
        if is_layer_pruning_enabled(module, layer_name, filter_for, skip_layers):
            print(f"Converting {layer_name} to sparse semi-structured")
            # module.weight = nn.Parameter(to_sparse_semi_structured(module.weight))
            old_weight = module.weight.detach()
            sparse_weight = to_sparse_semi_structured(old_weight)
            module.weight = nn.Parameter(sparse_weight)
            del old_weight, sparse_weight
            torch.cuda.empty_cache()
            gc.collect()
        
    print("[✓] Attached sparse kernel to model.")
    return model

def wrap_linears_with_svd(model, svd_factors_path, filter_for=None, skip_layers=None, dtype=torch.float16, device="cuda"):
    """
    Replaces nn.Linear layers with SparseSVDFusedWrap if factors are available.
    """

    print("[*] Wrapping Linear layers with SparseSVDFusedWrap...")

    svd_factors = torch.load(svd_factors_path, map_location="cpu")

    for name, module in model.named_modules():

        if is_layer_pruning_enabled(module, name, filter_for, skip_layers):
            if name not in svd_factors:
                # print(f"Skipping {name} as no SVD factors found.")
                continue

            print("SVD Wrapping layer:", name)

            # # Grab factors for this layer
            # V = svd_factors[name]["V"].to(device=device, dtype=dtype)
            # U_T = svd_factors[name]["U_T"].to(device=device, dtype=dtype)

            V = svd_factors[name]["V"].to("cpu")  # load in float
            U_T = svd_factors[name]["U_T"].to("cpu")

            V_q, V_scale = quantize_tensor(V, num_bits=8)
            U_T_q, U_T_scale = quantize_tensor(U_T, num_bits=8)

            V_q = V_q.to(device)
            U_T_q = U_T_q.to(device)

            # Replace module in parent
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name)

            setattr(parent, child_name,
                    SparseSVDFusedWrap(module, V_q, V_scale, U_T_q, U_T_scale))
            # print(f"[+] Wrapped {name} with SparseSVDFusedWrap")

    print("[✓] Wrapped all applicable Linear layers with SparseSVDFusedWrap.")
    return model


def compile_linears(model):

    print("[*] Compiling Linear layers...")
    for name, module in model.named_modules():
        if isinstance(module, SparseSVDFusedWrap) or isinstance(module, nn.Linear):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name)

            setattr(parent, child_name, torch.compile(module, mode="max-autotune"))
            # print(f"[✓] Compiled layer: {name}")

    print("[✓] Compiled all applicable Linear layers.")
    return model




# ------------ int4 SVD Support:

# def quantize_tensor(t: torch.Tensor, precision='int8'):
#     """
#     Symmetric linear quantization of a tensor.
#     Supports int8.
#     """
#     if precision == 'int8':
#         num_bits = 8
#         qmin = -(2**(num_bits-1))
#         qmax = (2**(num_bits-1)) - 1
#     elif precision == 'int4':
#         qmin = -8
#         qmax = 7
#     else:
#         raise ValueError("Unsupported precision")

#     # Compute scale (per tensor; can be per-channel for better accuracy)
#     max_val = t.abs().max()
#     scale = max_val / float(qmax)

#     # Quantize to integers
#     q = torch.clamp((t / scale).round(), qmin, qmax).to(torch.int8)

#     if precision == 'int4':
#         q_flat = q.flatten()
#         # ensure even length
#         if q_flat.numel() % 2 != 0:
#             q_flat = torch.cat([q_flat, torch.zeros(1, dtype=q_flat.dtype, device=q.device)])
#         # pack two int4 values into one int8
#         q_packed = (q_flat[::2] & 0x0F) | ((q_flat[1::2] & 0x0F) << 4)
#         return q_packed, scale, t.shape  # save original shape
    
#     return q, scale, t.shape

# class SparseSVDFusedWrap(nn.Module):
#     def __init__(self, linear, V_q, V_scale, U_T_q, U_T_scale, precision='int8', V_shape=None, U_T_shape=None):
#         super().__init__()
#         self.linear = linear
#         self.register_buffer("V_q", V_q.contiguous())
#         self.register_buffer("U_T_q", U_T_q.contiguous())
#         self.register_buffer("V_scale", torch.tensor(V_scale, dtype=torch.float32))
#         self.register_buffer("U_T_scale", torch.tensor(U_T_scale, dtype=torch.float32))
#         self.precision = precision
#         self.V_shape = V_shape
#         self.U_T_shape = U_T_shape

#     def _dequantize(self, q, scale, precision, original_shape):
#         if precision == "int8":
#             return (q.float() * scale)
#         elif precision == "int4":
#             # total number of real elements
#             numel = int(np.prod(original_shape))

#             # unpack two int4s from each int8
#             q_full = torch.zeros(numel + (numel % 2), dtype=torch.int8, device=q.device)
#             q_full[::2] = q & 0x0F
#             q_full[1::2] = (q >> 4) & 0x0F

#             # two's complement fix
#             q_full = q_full - (q_full > 7) * 16  

#             # truncate padded element (if any), then reshape
#             q_full = q_full[:numel]
#             return (q_full.float() * scale).reshape(original_shape)
    
#         else:
#             raise ValueError(f"Unsupported precision {precision}")
        
#     def forward(self, x):
#         y = F.linear(x, self.linear.weight, self.linear.bias)

#         # # dequantize on the fly
#         # V = (self.V_q.float() * self.V_scale).to(x.dtype)
#         # U_T = (self.U_T_q.float() * self.U_T_scale).to(x.dtype)

#         V = self._dequantize(self.V_q, self.V_scale, self.precision, self.V_shape).to(x.dtype)
#         U_T = self._dequantize(self.U_T_q, self.U_T_scale, self.precision, self.U_T_shape).to(x.dtype)

#         tmp = (x @ V) @ U_T
#         # y.add_(tmp)
#         return torch.add(y, tmp)  


# def is_layer_pruning_enabled(module, layer_name, filter_for=None, skip_layers=None):
#     """
#     Check if layer pruning is enabled for a specific layer.

#     filter_for: If provided, only check layers that contain this string in their name.
#         Possible Values: 'language_model', 'vision_backbone', etc.
#     skip_layers: If provided, skip layers that contain any of these strings in their name.
#         Possible Values: ['vision_backbone', 'lm_head', 'projector']
#     """
#     if isinstance(module, nn.Linear): #and "layer" in fqn:
#         if (filter_for is not None) and (filter_for in layer_name):
#             if skip_layers and any(skip_layer in layer_name for skip_layer in skip_layers):
#                 print("Skipping layer:", layer_name)
#                 return False
#             if module.weight.shape[0] % 32 == 0 and module.weight.shape[1] % 64 == 0:
#                 return True
#     return False


# def attach_sparse_kernel(model, filter_for=None, skip_layers=None):

#     print("[*] Attaching sparse kernel to model...")

#     for layer_name, module in model.named_modules():
#         if is_layer_pruning_enabled(module, layer_name, filter_for, skip_layers):
#             print(f"Converting {layer_name} to sparse semi-structured")
#             # module.weight = nn.Parameter(to_sparse_semi_structured(module.weight))
#             old_weight = module.weight.detach()
#             sparse_weight = to_sparse_semi_structured(old_weight)
#             module.weight = nn.Parameter(sparse_weight)
#             del old_weight, sparse_weight
#             torch.cuda.empty_cache()
#             gc.collect()
        
#     print("[✓] Attached sparse kernel to model.")
#     return model

# def wrap_linears_with_svd(model, svd_factors_path, filter_for=None, skip_layers=None, dtype=torch.float16, device="cuda"):
#     """
#     Replaces nn.Linear layers with SparseSVDFusedWrap if factors are available.
#     """

#     print("[*] Wrapping Linear layers with SparseSVDFusedWrap...")

#     svd_factors = torch.load(svd_factors_path, map_location="cpu")

#     for name, module in model.named_modules():

#         if is_layer_pruning_enabled(module, name, filter_for, skip_layers):
#             if name not in svd_factors:
#                 # print(f"Skipping {name} as no SVD factors found.")
#                 continue

#             # # Grab factors for this layer
#             # V = svd_factors[name]["V"].to(device=device, dtype=dtype)
#             # U_T = svd_factors[name]["U_T"].to(device=device, dtype=dtype)

#             V = svd_factors[name]["V"].to("cpu")  # load in float
#             U_T = svd_factors[name]["U_T"].to("cpu")

#             precision = 'int4'  # 'int8' or 'int4'

#             V_q, V_scale, V_shape = quantize_tensor(V, precision=precision)
#             U_T_q, U_T_scale, U_T_shape = quantize_tensor(U_T, precision=precision)

#             V_q = V_q.to(device)
#             U_T_q = U_T_q.to(device)

#             # Replace module in parent
#             parent_name = ".".join(name.split(".")[:-1])
#             child_name = name.split(".")[-1]
#             parent = model.get_submodule(parent_name)

#             setattr(parent, child_name,
#                     SparseSVDFusedWrap(module, V_q, V_scale, U_T_q, U_T_scale, precision=precision, V_shape=V_shape, U_T_shape=U_T_shape)
#                     )
#             # print(f"[+] Wrapped {name} with SparseSVDFusedWrap")

#     print("[✓] Wrapped all applicable Linear layers with SparseSVDFusedWrap.")
#     return model


# def compile_linears(model):

#     print("[*] Compiling Linear layers...")
#     for name, module in model.named_modules():
#         if isinstance(module, SparseSVDFusedWrap):
#             parent_name = ".".join(name.split(".")[:-1])
#             child_name = name.split(".")[-1]
#             parent = model.get_submodule(parent_name)

#             setattr(parent, child_name, torch.compile(module, mode="max-autotune"))
#             # print(f"[✓] Compiled layer: {name}")

#     print("[✓] Compiled all applicable Linear layers.")
#     return model