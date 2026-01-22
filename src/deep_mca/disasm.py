"""
This module is adapted from disasm from bhive, but placed here so it can be more easily imported.
"""

import subprocess


def disassemble(hex_str: str, output_intel_syntax: bool = False) -> str:
    args = []
    for i in range(0, len(hex_str), 2):
        byte = hex_str[i : i + 2]
        args.append("0x" + byte)

    syntax_id = 1 if output_intel_syntax else 0
    cmd = "echo {} | llvm-mc -disassemble -output-asm-variant={}".format(
        " ".join(args),
        syntax_id,
    )
    stdout = subprocess.check_output(cmd, shell=True)
    return stdout.decode("utf8")
