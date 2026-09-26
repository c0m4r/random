# .tools — offline verification dependencies

This machine has no GPU device nodes (`/dev/dri` and `/dev/nvidia*` are absent), so the
project is verified against a **software** Vulkan implementation. Nothing here is needed
to build or run `room2` on a real GPU — it exists purely so the renderer can be exercised
and validated in this environment.

Everything was downloaded from the Manjaro `extra` repository and extracted locally; no
root access is required and nothing outside this directory is modified.

| Path | Contents |
|---|---|
| `lvp-root/` | `vulkan-swrast` (Mesa **lavapipe**), a CPU Vulkan 1.4 implementation |
| `vvl-root/` | `vulkan-validation-layers` (Khronos validation) |
| `icd/lvp_icd.json` | ICD manifest pointing at the extracted lavapipe driver |
| `layers/VkLayer_khronos_validation.json` | layer manifest pointing at the extracted validation layer |
| `env.sh` | exports `VK_DRIVER_FILES` and `VK_LAYER_PATH` for both |

Usage:

```sh
source .tools/env.sh
./build/room2 --headless --screenshot shot.png
```

`scripts/verify.sh` does this automatically.

**On a normal machine with a working GPU you should not source `env.sh`** — doing so would
force the software driver and hide your real hardware.
