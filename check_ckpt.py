import torch

ckpt_path = "transforms_reduction/trainset_50pct/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/ooo-finetuning-epoch19-val_overall_loss0.00.ckpt"
print("Loading checkpoint from:", ckpt_path)
ckpt = torch.load(ckpt_path, map_location="cpu")
print("\nCheckpoint top-level keys:", list(ckpt.keys()))

# Check if state_dict exists
if 'state_dict' in ckpt:
    sd = ckpt['state_dict']
    print("\nState_dict keys:")
    for k in sd.keys():
        print(" ", k, "| shape:", sd[k].shape)
    
    # Show stats for each possible transform
    if 'transform_w' in sd:
        print("transform_w stats: mean=%.5f std=%.5f min=%.5f max=%.5f" % (
            sd['transform_w'].mean(), sd['transform_w'].std(), sd['transform_w'].min(), sd['transform_w'].max()
        ))
    if 'transform_b' in sd:
        print("transform_b stats: mean=%.5f std=%.5f min=%.5f max=%.5f" % (
            sd['transform_b'].mean(), sd['transform_b'].std(), sd['transform_b'].min(), sd['transform_b'].max()
        ))
else:
    print("No state_dict found in checkpoint!")
