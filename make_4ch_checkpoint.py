# -------------------------------------------------
# make_4ch_checkpoint.py
# Build a 4‑channel version of a YOLO‑v11 pretrained checkpoint.
# -------------------------------------------------
import torch, pathlib, copy

def make_4ch(pretrained_pt: pathlib.Path, out_pt: pathlib.Path):
    # -----------------------------------------------------------------
    # 1️⃣ Load the original checkpoint (3‑channel)
    # -----------------------------------------------------------------
    ckpt = torch.load(pretrained_pt, map_location="cpu")

    # Ultralytics >= 8.2 stores the weights under "model_state_dict",
    # older versions keep them under "model".
    state = ckpt.get("model_state_dict", ckpt["model"].float().state_dict())

    # -----------------------------------------------------------------
    # 2️⃣ Find the weight tensor of the *first* convolution
    # -----------------------------------------------------------------
    first_key = None
    for k in state.keys():
        if k.startswith("model.0") and k.endswith("conv.weight"):
            first_key = k
            break
    if first_key is None:
        raise RuntimeError("⚠️ Could not locate the first Conv weight in the checkpoint.")

    w3 = state[first_key]                 # shape = (out_c, 3, k, k)
    out_c, _, k, _ = w3.shape

    # -----------------------------------------------------------------
    # 3️⃣ Build a new weight tensor that has 4 input channels
    # -----------------------------------------------------------------
    w4 = torch.zeros((out_c, 4, k, k), dtype=w3.dtype)   # new (out_c, 4, k, k)

    # copy the three pretrained channels
    w4[:, :3, :, :] = w3

    # initialise the 4‑th channel – we use the *mean* of the three
    # keepdim=False → shape becomes (out_c, k, k), which matches the slice
    w4[:, 3, :, :] = w3.mean(dim=1)          # <-- fixed line

    # -----------------------------------------------------------------
    # 4️⃣ Replace the old weight with the new one in the state dict
    # -----------------------------------------------------------------
    state[first_key] = w4

    # -----------------------------------------------------------------
    # 5️⃣ Write a new checkpoint that mimics the original format
    # -----------------------------------------------------------------
    new_ckpt = copy.deepcopy(ckpt)
    new_ckpt["model_state_dict"] = state
    torch.save(new_ckpt, out_pt)

    print(f"✅ 4‑channel checkpoint written to: {out_pt}")

# -----------------------------------------------------------------
# 6️⃣ Run the script directly (you can also import `make_4ch` elsewhere)
# -----------------------------------------------------------------
if __name__ == "__main__":
    # -------------------------------------------------
    # Adjust ONLY the two paths below if your files live elsewhere
    # -------------------------------------------------
    PRE = pathlib.Path(r"C:\Users\HiWi\.ultralytics\models\yolo11s.pt")      # original 3‑ch weights
    OUT = pathlib.Path(r"C:\Users\HiWi\Desktop\Group 1 Project Folder\datasets\yolo11s_4ch.pt")
    make_4ch(PRE, OUT)