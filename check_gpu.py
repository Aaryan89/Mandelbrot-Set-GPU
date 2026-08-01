import sys

def check_gpu():
    print("=" * 65)
    print("  NVIDIA GPU & CUDA Availability Diagnostic Tool")
    print("=" * 65)
    
    cuda_detected = False
    
    # 1. Check Numba CUDA (Primary Kernel Engine)
    try:
        from numba import cuda
        print("\n[+] Probing Numba CUDA...")
        if cuda.is_available():
            cuda_detected = True
            device = cuda.get_current_device()
            name = device.name.decode('utf-8') if isinstance(device.name, bytes) else str(device.name)
            cc = device.compute_capability
            print(f"    - Numba CUDA Status : AVAILABLE")
            print(f"    - Device Name       : {name}")
            print(f"    - Compute Capability: {cc[0]}.{cc[1]}")
        else:
            print("    - Numba CUDA Status : Driver/Toolkit NOT detected by Numba.")
    except Exception as e:
        print(f"    - Numba CUDA Check  : Not ready ({e})")

    # 2. Check CuPy (Alternative CUDA Fallback Engine)
    try:
        import cupy as cp
        print("\n[+] Probing CuPy...")
        cuda_detected = True
        dev = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props['name'].decode('utf-8') if isinstance(props['name'], bytes) else str(props['name'])
        cc = (props['major'], props['minor'])
        print(f"    - CuPy Status       : AVAILABLE")
        print(f"    - Device Name       : {name}")
        print(f"    - Compute Capability: {cc[0]}.{cc[1]}")
        print(f"    - Total VRAM        : {props['totalGlobalMem'] / (1024**3):.2f} GB")
    except Exception as e:
        print(f"    - CuPy Check        : Not installed/configured ({e})")

    # 3. Check PyTorch CUDA (Alternative GPU Info & Tensor Fallback)
    try:
        import torch
        print("\n[+] Probing PyTorch CUDA...")
        if torch.cuda.is_available():
            cuda_detected = True
            dev_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"    - PyTorch CUDA      : AVAILABLE")
            print(f"    - Device Name       : {dev_name}")
            print(f"    - Compute Capability: {capability[0]}.{capability[1]}")
            print(f"    - Total VRAM        : {vram:.2f} GB")
            print(f"    - CUDA Version      : {torch.version.cuda}")
        else:
            print("    - PyTorch CUDA      : PyTorch installed without CUDA support.")
    except Exception as e:
        print(f"    - PyTorch Check     : Not installed ({e})")

    print("\n" + "=" * 65)
    if cuda_detected:
        print(" SUCCESS: CUDA GPU acceleration is available!")
    else:
        print(" WARNING: No CUDA acceleration backend detected.")
        print(" Please verify NVIDIA driver installation or install dependencies.")
    print("=" * 65)

if __name__ == "__main__":
    check_gpu()
