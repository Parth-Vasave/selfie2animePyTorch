"""Convert MXNet .params weights to PyTorch .pth format for CycleGAN."""

import struct
import sys
import numpy as np
import torch
from pix2pix_gan import ResnetGenerator


def parse_mxnet_params(filepath):
    """Parse MXNet NDArray .params file and return dict of name -> numpy array."""
    with open(filepath, 'rb') as f:
        data = f.read()

    num_entries = struct.unpack('<Q', data[16:24])[0]

    # Find all NDArray magic positions
    magic_bytes = b'\xc9\xfa\x93\xf9'
    positions = []
    pos = 0
    while True:
        pos = data.find(magic_bytes, pos)
        if pos == -1:
            break
        positions.append(pos)
        pos += 1

    assert len(positions) == num_entries

    # Parse each NDArray
    arrays = []
    for p in positions:
        offset = p + 4  # skip magic
        offset += 4  # skip reserved
        num_dims = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        shape = []
        for _ in range(num_dims):
            dim = struct.unpack('<Q', data[offset:offset + 8])[0]
            shape.append(dim)
            offset += 8
        offset += 8  # skip dev_type + dev_id
        num_elements = 1
        for d in shape:
            num_elements *= d
        arr = np.frombuffer(data[offset:offset + num_elements * 4], dtype=np.float32).copy().reshape(shape)
        arrays.append(arr)

    # Find key section (starts with num_entries as u64)
    marker = struct.pack('<Q', num_entries)
    key_offset = data.rfind(marker)
    remaining = data[key_offset + 8:]
    keys = []
    offset = 0
    for _ in range(num_entries):
        key_len = struct.unpack('<Q', remaining[offset:offset + 8])[0]
        offset += 8
        key = remaining[offset:offset + key_len].decode('ascii')
        offset += key_len
        keys.append(key)

    return {k: v for k, v in zip(keys, arrays)}


def convert(mxnet_params_path, output_path):
    """Convert MXNet params to PyTorch state dict (without spectral norm)."""
    mx_params = parse_mxnet_params(mxnet_params_path)

    print(f"Loaded {len(mx_params)} MXNet parameters")
    for k, v in mx_params.items():
        print(f"  {k}: {v.shape}")

    # Build model WITHOUT spectral norm for inference.
    # MXNet's _weight is the raw weight, and spectral norm state (u/v vectors)
    # doesn't transfer cleanly between frameworks. For inference this is fine —
    # spectral norm is a training regularizer and doesn't affect inference quality.
    model = ResnetGenerator(use_spectral_norm=False)
    target_sd = model.state_dict()

    # Collect weights and u vectors separately, then normalize
    raw_weights = {}  # base_key -> weight tensor
    u_vectors = {}    # base_key -> u tensor (1, out_channels)
    new_sd = {}

    for mx_key, mx_arr in mx_params.items():
        # Strip leading underscore
        clean = mx_key.lstrip('_')
        # Replace ._net. with .net.
        clean = clean.replace('._net.', '.net.')

        if clean.endswith('._u'):
            base = clean[:-3]  # remove ._u
            u_vectors[base] = torch.from_numpy(mx_arr)  # keep as (1, N)
        elif clean.endswith('._weight'):
            base = clean[:-8]  # remove ._weight
            raw_weights[base] = torch.from_numpy(mx_arr)
        else:
            # CAM layers: cam._gap_linear.weight -> cam.gap_linear.weight
            clean = clean.replace('._', '.')
            new_sd[clean] = torch.from_numpy(mx_arr)

    # Apply spectral normalization: W_normalized = W / sigma
    # MXNet stores the raw (unnormalized) weight and u vector.
    # MXNet's forward does exactly 1 power iteration step using the saved u,
    # so we replicate that here to bake the normalization into the weight.
    print("\nApplying spectral normalization to weights...")
    for base_key in raw_weights:
        weight = raw_weights[base_key]
        if base_key in u_vectors:
            u = u_vectors[base_key]  # (1, out_channels)
            w_mat = weight.reshape(weight.shape[0], -1)  # (out, in*k*k)
            # MXNet 1-step power iteration:
            # v = L2Norm(u @ W)
            v = torch.mm(u, w_mat)
            v = v / v.norm()
            # u_new = L2Norm(v @ W^T)
            u_new = torch.mm(v, w_mat.t())
            u_new = u_new / u_new.norm()
            # sigma = sum(u_new @ W * v)
            sigma = torch.sum(torch.mm(u_new, w_mat) * v)
            weight = weight / sigma
            print(f"  {base_key}: sigma={sigma.item():.4f}")
        new_sd[base_key + '.weight'] = weight

    # Verify all keys match
    missing = set(target_sd.keys()) - set(new_sd.keys())
    extra = set(new_sd.keys()) - set(target_sd.keys())
    if missing:
        print(f"\nMISSING keys in converted weights: {missing}")
    if extra:
        print(f"\nEXTRA keys not in model: {extra}")

    # Shape check
    for k in new_sd:
        if k in target_sd:
            if new_sd[k].shape != target_sd[k].shape:
                print(f"  SHAPE MISMATCH: {k}: converted {new_sd[k].shape} vs expected {target_sd[k].shape}")

    torch.save(new_sd, output_path)
    print(f"\nSaved PyTorch weights to {output_path}")

    # Validation
    model.load_state_dict(new_sd)
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 256, 256)
        out, cam_out = model(dummy)
        print(f"Validation: input {dummy.shape} -> output {out.shape}, cam {cam_out.shape}")
    print("Conversion successful!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_weights.py <input.params> [output.pth]")
        print("Example: python convert_weights.py model/selfie2anime.gen_ab.params model/selfie2anime.gen_ab.pth")
        sys.exit(1)

    input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = input_path.replace('.params', '.pth')

    convert(input_path, output_path)
