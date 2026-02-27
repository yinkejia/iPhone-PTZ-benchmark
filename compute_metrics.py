from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from metrics import mPSNR, mLPIPS, FIDCalculator
import torch
import argparse

def compute_metrics(gt, pred, masks=None):
    device = "cuda"
    psnr_metric = mPSNR().to(device)
    lpips_metric = mLPIPS().to(device)

    for i in range(len(gt)):
        gt_img = torch.from_numpy(np.array(gt[i])).to(device) / 255.0
        pred_img = torch.from_numpy(np.array(pred[i])).to(device) / 255.0
        if masks is not None:
            mask = torch.from_numpy(np.array(masks[i])).to(device) / 255.0
        else:
            mask = torch.ones_like(pred_img[..., 0])
        psnr_metric.update(gt_img, pred_img, mask)
        lpips_metric.update(gt_img[None], pred_img[None], mask[None])
    mpsnr = psnr_metric.compute().item()
    mlpips = lpips_metric.compute().item()
    print(f"NV mPSNR: {mpsnr:.4f}")
    print(f"NV mLPIPS: {mlpips:.4f}")
    return mpsnr, mlpips

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate metrics.")
    parser.add_argument("--gt_folder", type=str, default="./evaluation_benchmarks")
    parser.add_argument("--results_folder", type=str, default="./output_frames")
    parser.add_argument("--output_folder", type=str, default="./evaluated_metrics")
    args = parser.parse_args()

    datasets = ["iPhone", "iPhone-PTZ"]
    results_folder = args.results_folder
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    fid_calculator = FIDCalculator(device="cuda")

    gt_dir = args.gt_folder
    camdirector_dir = os.path.join(results_folder, "camdirector")
    for dataset in datasets:
        if dataset == "iPhone":
            scenes = ["apple", "block", "paper-windmill", "spin", "teddy"]
        else:
            scenes = ["container", "corner", "door2", "exercise2", "library", "mural4", "playground", "sit", "stairs", "taiji2"]
            
        gt = {}
        src = {}
        camdirector = {}

        for scene in tqdm(scenes, desc="Loading frames"):
            frame_names = sorted(os.listdir(os.path.join(results_folder, "camdirector", scene)))
            
            gt[scene] = [Image.open(os.path.join(gt_dir, dataset, scene, "target_imgs", image_name)).convert("RGB") for image_name in frame_names]
            camdirector[scene] = [Image.open(os.path.join(camdirector_dir, scene, image_name)).convert("RGB") for image_name in frame_names]

        psnr_camdirector, lpips_camdirector = [], []

        for scene in tqdm(scenes, desc="Evaluate metrics on all frames"):
            mpsnr, mlpips = compute_metrics(gt[scene], camdirector[scene], None)
            psnr_camdirector.append(mpsnr)
            lpips_camdirector.append(mlpips)
        fid_camdirector = fid_calculator.compute_fid([frame for frames in gt.values() for frame in frames], [frame for frames in camdirector.values() for frame in frames])

        if dataset == "iPhone":
            output_file = os.path.join(output_folder, "metrics_iPhone.txt")
        else:
            output_file = os.path.join(output_folder, "metrics_iPhone-PTZ.txt")

        methods = [
            ("CamDirector", psnr_camdirector, lpips_camdirector, fid_camdirector),
        ]

        # Header
        header = f"{'Method':<15}{'Scene':<20}{'PSNR':>8}{'LPIPS':>10}\n"
        header += "-" * (15 + 20 + 8 + 10) + "\n"
        lines = [header]

        # Per-scene results
        for i, scene in enumerate(scenes):
            for method_name, psnr_list, lpips_list, fid_score in methods:
                psnr = psnr_list[i]
                lpips = lpips_list[i]
                lines.append(f"{method_name:<15}{scene:<20}{psnr:>8.3f}{lpips:>10.4f}\n")

        # Separator
        lines.append("\n" + "=" * (15 + 20 + 8 + 10 + 16) + "\n")
        lines.append(f"{'Method':<15}{'Mean PSNR':>12}{'Mean LPIPS':>14}{'FID':>12}\n")
        lines.append("-" * (15 + 12 + 14 + 12) + "\n")

        # Mean values per method
        for method_name, psnr_list, lpips_list, fid_score in methods:
            mean_psnr = np.mean(psnr_list)
            mean_lpips = np.mean(lpips_list)
            lines.append(f"{method_name:<15}{mean_psnr:>12.3f}{mean_lpips:>14.4f}{fid_score:>12.3f}\n")

        # Save to file
        with open(output_file, "w") as f:
            f.writelines(lines)

        print(f"✅ Results table saved to: {output_file}")

        psnr_camdirector, lpips_camdirector = [], []


        for scene in tqdm(scenes, desc="Evaluate metrics on first 41 frames"):
            mpsnr, mlpips = compute_metrics(gt[scene][:41], camdirector[scene][:41], None)
            psnr_camdirector.append(mpsnr)
            lpips_camdirector.append(mlpips)
        fid_camdirector = fid_calculator.compute_fid([frame for frames in gt.values() for frame in frames[:41]], [frame for frames in camdirector.values() for frame in frames[:41]])


        if dataset == "iPhone":
            output_file = os.path.join(output_folder, "metrics_iPhone_first_41.txt")
        else:
            output_file = os.path.join(output_folder, "metrics_iPhone-PTZ_first_41.txt")
        methods = [
            ("CamDirector", psnr_camdirector, lpips_camdirector, fid_camdirector),
        ]

        # Header
        header = f"{'Method':<15}{'Scene':<20}{'PSNR':>8}{'LPIPS':>10}\n"
        header += "-" * (15 + 20 + 8 + 10) + "\n"
        lines = [header]

        # Per-scene results
        for i, scene in enumerate(scenes):
            for method_name, psnr_list, lpips_list, fid_score in methods:
                psnr = psnr_list[i]
                lpips = lpips_list[i]
                lines.append(f"{method_name:<15}{scene:<20}{psnr:>8.3f}{lpips:>10.4f}\n")

        # Separator
        lines.append("\n" + "=" * (15 + 20 + 8 + 10 + 16) + "\n")
        lines.append(f"{'Method':<15}{'Mean PSNR':>12}{'Mean LPIPS':>14}{'FID':>12}\n")
        lines.append("-" * (15 + 12 + 14 + 12) + "\n")

        # Mean values per method
        for method_name, psnr_list, lpips_list, fid_score in methods:
            mean_psnr = np.mean(psnr_list)
            mean_lpips = np.mean(lpips_list)
            lines.append(f"{method_name:<15}{mean_psnr:>12.3f}{mean_lpips:>14.4f}{fid_score:>12.3f}\n")

        # Save to file
        with open(output_file, "w") as f:
            f.writelines(lines)

        print(f"✅ Results table saved to: {output_file}")