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
    """Convert MXNet params to PyTorch state dict WITH spectral norm.

    Uses PyTorch's spectral_norm so sigma is computed dynamically during
    forward, exactly matching MXNet's behavior.
    """
    mx_params = parse_mxnet_params(mxnet_params_path)

    print(f"Loaded {len(mx_params)} MXNet parameters")
    for k, v in mx_params.items():
        print(f"  {k}: {v.shape}")

    # Build model WITH spectral norm (default)
    model = ResnetGenerator(use_spectral_norm=True)
    target_sd = model.state_dict()

    # Collect MXNet params by type
    raw_weights = {}
    u_vectors = {}
    cam_params = {}

    for mx_key, mx_arr in mx_params.items():
        clean = mx_key.lstrip('_').replace('._net.', '.net.')

        if clean.endswith('._u'):
            base = clean[:-3]
            u_vectors[base] = torch.from_numpy(mx_arr)
        elif clean.endswith('._weight'):
            base = clean[:-8]
            raw_weights[base] = torch.from_numpy(mx_arr)
        else:
            clean = clean.replace('._', '.')
            cam_params[clean] = torch.from_numpy(mx_arr)

    # Build PyTorch state dict
    new_sd = dict(cam_params)

    print("\nMapping weights with spectral norm...")
    for base_key, weight in raw_weights.items():
        new_sd[base_key + '.weight_orig'] = weight

        if base_key in u_vectors:
            u_mx = u_vectors[base_key]  # (1, out_channels)
            u = u_mx.flatten()  # (out_channels,)

            # Compute v with 1 power iteration step (matching MXNet)
            w_mat = weight.reshape(weight.shape[0], -1)
            v = torch.mv(w_mat.t(), u)
            v = v / v.norm()
            # Re-derive u for consistency
            u = torch.mv(w_mat, v)
            u = u / u.norm()

            new_sd[base_key + '.weight_u'] = u
            new_sd[base_key + '.weight_v'] = v

            sigma = torch.dot(u, torch.mv(w_mat, v))
            print(f"  {base_key}: sigma={sigma.item():.4f}")

    # Verify all keys match
    missing = set(target_sd.keys()) - set(new_sd.keys())
    extra = set(new_sd.keys()) - set(target_sd.keys())
    if missing:
        print(f"\nMISSING keys: {missing}")
    if extra:
        print(f"\nEXTRA keys: {extra}")

    for k in new_sd:
        if k in target_sd and new_sd[k].shape != target_sd[k].shape:
            print(f"  SHAPE MISMATCH: {k}: {new_sd[k].shape} vs {target_sd[k].shape}")

    torch.save(new_sd, output_path)
    print(f"\nSaved PyTorch weights to {output_path}")

    # Validation
    model.load_state_dict(new_sd)
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 256, 256)
        out, cam_out = model(dummy)
        print(f"Validation: input {dummy.shape} -> output {out.shape}, cam {cam_out.shape}")
        print(f"Output range: [{out.min():.4f}, {out.max():.4f}], mean={out.mean():.4f}, std={out.std():.4f}")
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
