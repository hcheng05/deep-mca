import shutil

import torch


def check_env():
    # check pytorch
    try:
        print("PyTorch is installed")
        if torch.cuda.is_available():
            print(f"gpu available {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            print("gpu available apple mps ")
        else:
            print("no gpu lol")
    except ImportError:
        raise Exception("PyTorch is not installed")

    # check llvm
    mc_path = shutil.which("llvm-mc")
    mca_path = shutil.which("llvm-mca")
    objdump_path = shutil.which("llvm-objdump")

    if mc_path:
        print("llvm-mc was found")
    else:
        raise Exception("llvm-mc is not installed")

    if mca_path:
        print("llvm-mca was found")
    else:
        raise Exception("llvm-mca is not installed")

    if objdump_path:
        print("llvm-objdump was found")
    else:
        raise Exception("llvm-objdump is not installed")

    print("Environment check passed")


if __name__ == "__main__":
    check_env()
