# mycode/fast3r_wrapper.py
import torch
import numpy as np
import os
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference
from mycode.utils.cuda_utils import get_available_gpu

class Fast3RPredictor:
    def __init__(self, model_id="jedyang97/Fast3R_ViT_Large_512", device="cuda"):
        gpu_id = get_available_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[INFO] Using GPU {gpu_id}")

        self.device = torch.device(device)
        self.model = Fast3R.from_pretrained(model_id).to(self.device)
        self.lit_module = MultiViewDUSt3RLitModule.load_for_inference(self.model)
        self.model.eval()
        self.lit_module.eval()

    def predict(self, image_paths: list[str], max_images=256, filter_confidence=True, sample_points=200000):
        """
        image_paths: list of str (file paths to color images)
        return: Nx3 numpy array of 3D points
        """
        print("[INFO] Using GPU:", torch.cuda.current_device(), torch.cuda.get_device_name())

        image_paths = image_paths[:max_images]
        images = load_images(image_paths, size=512, verbose=False)

        with torch.no_grad():
            output_dict = inference(
                images,
                self.model,
                self.device,
                dtype=torch.float32,
                verbose=False,
                profiling=False
            )


        all_points = []
        for pred in output_dict["preds"]:
            pts = pred["pts3d_in_other_view"][0]  # (H, W, 3)

            if filter_confidence and "mask" in pred:
                mask = pred["mask"][0] > 0  # (H, W), bool
                pts = pts[mask]  # (N, 3)
            else:
                pts = pts.reshape(-1, 3)

            all_points.append(pts.cpu().numpy())

        merged_points = np.concatenate(all_points, axis=0)

        # Random sampling to reduce file size
        if sample_points and merged_points.shape[0] > sample_points:
            idx = np.random.choice(merged_points.shape[0], sample_points, replace=False)
            merged_points = merged_points[idx]

        return merged_points
