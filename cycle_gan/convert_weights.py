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
    """Convert MXNet params to PyTorch state dict."""
    mx_params = parse_mxnet_params(mxnet_params_path)

    print(f"Loaded {len(mx_params)} MXNet parameters")
    for k, v in mx_params.items():
        print(f"  {k}: {v.shape}")

    # Build MXNet key -> PyTorch key mapping
    # MXNet: _enc.1._weight -> PyTorch: enc.1.weight_orig
    # MXNet: _enc.1._u      -> PyTorch: enc.1.weight_u (reshape from [1, N] to [N])
    # MXNet: _cam._gap_linear.weight -> PyTorch: cam.gap_linear.weight (no spectral norm)
    # MXNet: _cam._out.weight -> PyTorch: cam.out.weight
    # MXNet: _cam._out.bias  -> PyTorch: cam.out.bias

    # Create model to get target state dict structure
    model = ResnetGenerator()
    target_sd = model.state_dict()

    new_sd = {}

    for mx_key, mx_arr in mx_params.items():
        # Strip leading underscore
        clean = mx_key.lstrip('_')
        # Replace ._net. with .net.
        clean = clean.replace('._net.', '.net.')

        if clean.endswith('._u'):
            # Spectral norm u vector: enc.3._u -> enc.3.weight_u
            base = clean[:-3]  # remove ._u
            pt_key = base + '.weight_u'
            new_sd[pt_key] = torch.from_numpy(mx_arr.flatten())
        elif clean.endswith('._weight'):
            # Spectral norm weight: enc.3._weight -> enc.3.weight_orig
            base = clean[:-8]  # remove ._weight
            pt_key = base + '.weight_orig'
            new_sd[pt_key] = torch.from_numpy(mx_arr)
        else:
            # CAM layers etc: cam._gap_linear.weight -> cam.gap_linear.weight
            # Clean remaining ._ prefixes on sub-module names
            clean = clean.replace('._', '.')
            pt_key = clean
            new_sd[pt_key] = torch.from_numpy(mx_arr)

    # Compute missing weight_v vectors via power iteration
    print("\nComputing spectral norm weight_v vectors...")
    for key in list(target_sd.keys()):
        if key.endswith('.weight_v') and key not in new_sd:
            # Get corresponding weight_orig and weight_u
            base = key[:-9]  # remove .weight_v
            w_key = base + '.weight_orig'
            u_key = base + '.weight_u'
            if w_key in new_sd and u_key in new_sd:
                weight = new_sd[w_key]
                u = new_sd[u_key]
                # Reshape weight to 2D: (out_features, in_features*k*k)
                w_mat = weight.reshape(weight.shape[0], -1)
                # v = W^T u / ||W^T u||
                v = torch.mv(w_mat.t(), u)
                v = v / v.norm()
                new_sd[key] = v

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

    # Quick validation
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
