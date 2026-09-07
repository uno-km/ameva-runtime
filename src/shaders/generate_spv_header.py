import struct
import os

def spv_to_header(spv_path: str, header_path: str, array_name: str):
    with open(spv_path, "rb") as f:
        data = f.read()
    
    assert len(data) % 4 == 0, f"SPV size ({len(data)}) must be a multiple of 4"
    words = struct.unpack(f"<{len(data)//4}I", data)
    assert words[0] == 0x07230203, f"Invalid SPIR-V magic number: 0x{words[0]:08x}"
    
    lines = []
    lines.append(f"// Auto-generated from {os.path.basename(spv_path)} via glslangValidator. DO NOT EDIT.")
    lines.append("#pragma once")
    lines.append("#include <cstdint>")
    lines.append("#include <cstddef>")
    lines.append("")
    lines.append("namespace ameva {")
    lines.append("namespace shaders {")
    lines.append("")
    lines.append(f"alignas(4) static const uint32_t {array_name}[] = {{")
    
    row = []
    for i, w in enumerate(words):
        row.append(f"0x{w:08x}")
        if len(row) == 8:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ", ".join(row) + ",")
        
    lines.append("};")
    lines.append(f"static const size_t {array_name}ByteSize = {len(data)};")
    lines.append(f"static const size_t {array_name}WordCount = {len(words)};")
    lines.append("")
    lines.append("} // namespace shaders")
    lines.append("} // namespace ameva")
    lines.append("")
    
    with open(header_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Successfully generated {header_path} ({len(data)} bytes, {len(words)} words)")

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    spv_to_header(os.path.join(cur_dir, "bitnet_gemv_i2_s.spv"), os.path.join(cur_dir, "bitnet_gemv_i2_s_spv.h"), "kBitnetGemvI2SSpv")
    spv_to_header(os.path.join(cur_dir, "swiglu_silu.spv"), os.path.join(cur_dir, "swiglu_silu_spv.h"), "kSwigluSiluSpv")
    spv_to_header(os.path.join(cur_dir, "rmsnorm.spv"), os.path.join(cur_dir, "rmsnorm_spv.h"), "kRmsNormSpv")
