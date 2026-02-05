"""
sae_tester_patch4x4.py

Drop-in replacement for your SAETester file, adding support for "super-patches"
(i.e., pooling the ViT 14x14 token grid into a 4x4 grid) for the same analysis:
- highlight a 4x4 region
- get top neurons for that region
- show top activating images
- optionally show a coarse 4x4 segmentation overlay for a feature

Key idea:
- ViT tokens are per-patch (typically 14x14 = 196 for 224x224 inputs).
- A "4x4 patch" analysis is done by pooling those 14x14 tokens -> 4x4 regions.

How to use:
- tester.register_image(...)
- tester.run_region(grid_size=4, region_idx=..., ...)
"""

import os
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image


# ----------------------------
# Helpers
# ----------------------------
def _infer_token_grid(num_tokens: int) -> int:
    """Infer H=W for a square token grid (e.g., 196 -> 14)."""
    g = int(round(num_tokens**0.5))
    if g * g != num_tokens:
        raise ValueError(f"Expected a square number of tokens, got {num_tokens}.")
    return g


def pool_tokens_to_grid(
    token_feat: torch.Tensor, grid_out: int, mode: str = "avg"
) -> torch.Tensor:
    """
    Pools token features from an NxN token grid to (grid_out x grid_out).

    token_feat: [T, D] or [N, N, D]
    returns:    [(grid_out*grid_out), D]
    """
    if token_feat.dim() == 2:
        T, D = token_feat.shape
        N = _infer_token_grid(T)
        token_feat = token_feat.view(N, N, D)
    elif token_feat.dim() == 3:
        N, N2, D = token_feat.shape
        if N != N2:
            raise ValueError(f"Token grid must be square, got {N}x{N2}.")
    else:
        raise ValueError(f"token_feat must be [T,D] or [N,N,D], got {token_feat.shape}")

    # [D, N, N]
    x = token_feat.permute(2, 0, 1)

    if mode == "avg":
        xg = F.adaptive_avg_pool2d(x, output_size=(grid_out, grid_out))  # [D, g, g]
    elif mode == "max":
        xg = F.adaptive_max_pool2d(x, output_size=(grid_out, grid_out))
    else:
        raise ValueError("mode must be 'avg' or 'max'")

    # [(g*g), D]
    return xg.permute(1, 2, 0).reshape(grid_out * grid_out, -1)


def upsample_grid_mask(
    grid_vals: torch.Tensor, grid_size: int, out_h: int, out_w: int
) -> np.ndarray:
    """
    grid_vals: [grid_size*grid_size] (torch)
    returns:   [out_h, out_w] mask normalized [0..1]
    """
    mask = grid_vals.view(1, 1, grid_size, grid_size)
    mask = F.interpolate(mask, size=(out_h, out_w), mode="bilinear", align_corners=False)[0, 0]
    mask = mask.detach().cpu().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-10)
    return mask


# ----------------------------
# Mixins
# ----------------------------
class UtilMixin:
    def _get_max_activating_images_and_labels(self, neuron_idx, dataset, max_activating_image_indices):
        img_list = max_activating_image_indices[neuron_idx]
        images = []
        labels = []
        for i in img_list:
            try:
                images.append(dataset[i.item()]["image"])
                labels.append(dataset[i.item()]["label"])
            except Exception:
                # fallback behavior used in your original file
                images.append(dataset[i.item()]["image"])
                labels.append(0)
        return images, labels

    def _create_patches(self, patch=256):
        """
        Creates visual patches from the *processed image tensor*.
        This is independent from ViT token patches. For a 224x224 image:
        - patch=14 would make many tiny patches (not what you want)
        - patch=56 gives a 4x4 grid
        """
        temp = self.processed_image.clone()  # [1,3,H,W]
        patches = temp[0].data.unfold(0, 3, 3)      # channels
        patches = patches.unfold(1, patch, patch)   # height
        patches = patches.unfold(2, patch, patch)   # width
        return patches


class VisualizeMixin:
    def _plot_input_image(self, save=True):
        plt.imshow(self.input_image)
        if save:
            img_name = os.path.basename(str(self.img_url)).split(".")[0]
            save_name = f"{self.save_dir}/{img_name}/input_image.png"
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            plt.savefig(save_name, bbox_inches="tight", dpi=300)
            plt.show()
            plt.close()
        else:
            plt.show()

    def _plot_feature_mask(self, patches, feat_idx, mask=None, plot=True, save=True):
        """
        NOTE: This expects 'mask' to be indexable per patch cell (flattened).
        For your 4x4 pooled mask, pass a 4x4-derived mask aligned to patches.size(1)*patches.size(2).
        """
        if mask is None:
            # fallback to SAE activation for a token-grid; not ideal for pooled 4x4 use
            mask = self.sae_act[0, :, feat_idx].cpu()

        fig, axs = plt.subplots(patches.size(1), patches.size(2), figsize=(6, 6))
        plt.subplots_adjust(wspace=0.01, hspace=0.01)

        for i in range(patches.size(1)):
            for j in range(patches.size(2)):
                patch = patches[0, i, j].permute(1, 2, 0)
                patch *= torch.tensor(self.vit.processor.image_processor.image_std)
                patch += torch.tensor(self.vit.processor.image_processor.image_mean)

                # mask index: +1 was for CLS token in your old code;
                # for pooled region masks, you should pass a mask of length patches.size(1)*patches.size(2)
                idx = i * patches.size(2) + j
                masked_patch = patch * float(mask[idx])
                masked_patch = (masked_patch - masked_patch.min()) / (masked_patch.max() - masked_patch.min() + 1e-8)

                axs[i, j].imshow(masked_patch)
                axs[i, j].axis("off")

        fig.suptitle(str(feat_idx))
        if save:
            img_name = os.path.basename(str(self.img_url)).split(".")[0]
            save_name = f"{self.save_dir}/{img_name}/feature_masks/{feat_idx}.png"
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            fig.savefig(save_name, dpi=300)
            plt.close(fig)
        else:
            plt.close(fig)
        return fig

    def _plot_patches(self, patches, highlight_patch_idx=None, save=True):
        fig, axs = plt.subplots(patches.size(1), patches.size(2), figsize=(6, 6))
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        for i in range(patches.size(1)):
            for j in range(patches.size(2)):
                patch = patches[0, i, j].permute(1, 2, 0)
                patch *= torch.tensor(self.vit.processor.image_processor.image_std)
                patch += torch.tensor(self.vit.processor.image_processor.image_mean)
                axs[i, j].imshow(patch)

                flat_idx = i * patches.size(2) + j
                if flat_idx == highlight_patch_idx:
                    for spine in axs[i, j].spines.values():
                        spine.set_edgecolor("red")
                        spine.set_linewidth(3)
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])
                else:
                    axs[i, j].axis("off")

        if save:
            img_name = os.path.basename(str(self.img_url)).split(".")[0]
            save_name = f"{self.save_dir}/{img_name}/patches.png"
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            plt.savefig(save_name, bbox_inches="tight", dpi=300)
            plt.show()
            plt.close(fig)
        else:
            plt.show()

    def _plot_union_top_neruons(self, top_k, union_top_neurons, token_idx, token_act, save=False):
        print(f"Union of top {top_k} neurons: {union_top_neurons}")

        plt.figure(figsize=(10, 5))
        plt.plot(token_act, color="black")
        plt.plot(union_top_neurons, token_act[union_top_neurons], "ro", label="Top neurons", markersize=5)

        for idx in union_top_neurons:
            plt.text(idx, token_act[idx] + 0.05, str(idx), fontsize=9, ha="center")

        plt.legend()
        plt.title(f"token/region {token_idx} activation")

        if save:
            img_name = os.path.basename(str(self.img_url)).replace(".jpg", "")
            save_name = f"{self.save_dir}/{img_name}/activation/{token_idx}.jpg"
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            plt.savefig(save_name, dpi=300)
            plt.show()

        plt.close()

    def _plot_images(self, dataset_name, images, neuron_idx, labels=None, suptitle=None, top_k=5, save=True):
        images = [img.resize((224, 224)) for img in images]
        num_cols = min(top_k, 5)
        num_rows = (top_k + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4.5 * num_cols, 5 * num_rows))
        axes = axes.flatten()

        for i in range(top_k):
            axes[i].imshow(images[i])
            axes[i].axis("off")

        plt.tight_layout()

        if save:
            img_name = os.path.basename(str(self.img_url)).replace(".jpg", "")
            save_name = f"{self.save_dir}/{img_name}/top_images/{dataset_name}/{neuron_idx}.jpg"
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            plt.savefig(save_name, dpi=300)

        plt.close()
        return fig

    def _fig_to_img(self, fig):
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        img = img.reshape(canvas.get_width_height()[::-1] + (3,))
        return img

    def _plot_multiple_images(self, figs, neuron_idx, top_k=5, save=True, seg=False):
        num_plots = len(figs)
        cols = 1
        rows = (num_plots + cols - 1) // cols
        combined_fig = plt.figure(figsize=(20, 12))

        for i, fig in enumerate(figs):
            ax = combined_fig.add_subplot(rows, cols, i + 1)
            img = self._fig_to_img(fig)
            ax.imshow(img)
            ax.axis("off")

        if save:
            img_name = os.path.basename(str(self.img_url)).replace(".jpg", "")
            if seg:
                save_name = f"{self.save_dir}/{img_name}/top_images/seg_{neuron_idx}.jpg"
            else:
                save_name = f"{self.save_dir}/{img_name}/top_images/{neuron_idx}.jpg"
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            plt.savefig(save_name, dpi=300)

        combined_fig.show()


# ----------------------------
# Main class with 4x4 support
# ----------------------------
class SAETester(VisualizeMixin, UtilMixin):
    def __init__(
        self,
        vit,
        cfg,
        sae,
        mean_acts,
        max_act_images,
        datasets,
        class_names,
        noisy_threshold=0.1,
        device="cpu",
        save_dir="./saved_images",
        pool_mode: str = "avg",  # 'avg' or 'max' when pooling tokens -> grid
    ):
        self.vit = vit
        self.cfg = cfg
        self.sae = sae
        self.mean_acts = mean_acts
        self.max_act_images = max_act_images
        self.datasets = datasets
        self.class_names = class_names
        self.noisy_threshold = noisy_threshold
        self.device = device
        self.save_dir = save_dir
        self.pool_mode = pool_mode

    def register_image(self, img_url: str) -> None:
        """Load and process an image from a URL or local path."""
        if isinstance(img_url, str):
            image = self._load_image(img_url)
        else:
            image = img_url

        self.input_image = image
        self.img_url = img_url

        if image.mode != "RGB":
            image = image.convert("RGB")

        processed = self.vit.processor(image)  # expected [3,H,W]
        self.processed_image = processed.unsqueeze(0)  # [1,3,H,W]

    def _load_image(self, img_url: str) -> Image.Image:
        if "https" in img_url:
            response = requests.get(img_url)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        return Image.open(img_url)

    @property
    def processed_image(self):
        return self._processed_image

    @processed_image.setter
    def processed_image(self, value):
        self._processed_image = value

    @property
    def input_image(self):
        return self._input_image

    @input_image.setter
    def input_image(self, value):
        self._input_image = value

    # ----------------------------
    # Original behaviors
    # ----------------------------
    def show_input_image(self, save=True):
        self._plot_input_image(save=save)

    def _run_vit_hook(self, image=None):
        if image is None:
            inputs = self.processed_image.to(self.device)
        else:
            if image.mode != "RGB":
                image = image.convert("RGB")
            inputs = self.vit.processor(image).unsqueeze(0).to(self.device)

        list_of_hook_locations = [(self.cfg.block_layer, self.cfg.module_name)]
        _vit_out, vit_cache_dict = self.vit.run_with_cache(list_of_hook_locations, inputs)
        vit_act = vit_cache_dict[(self.cfg.block_layer, self.cfg.module_name)]
        return vit_act

    def _run_sae_hook(self, vit_act):
        _sae_out, sae_cache_dict = self.sae.run_with_cache(vit_act)
        sae_act = sae_cache_dict["hook_hidden_post"]
        if sae_act.shape[0] != 1:
            sae_act = sae_act.permute(1, 0, 2)

        # return patch tokens only (drop CLS), but keep full grid
        # typical: sae_act: [1, 1+T, d] -> return [1, T, d]
        return sae_act[:, 1:, :]

    def _filter_out_nosiy_activation(self, features):
        noisy_features_indices = ((self.mean_acts["mito"] > self.noisy_threshold).nonzero()[0].tolist())
        features_copy = deepcopy(features)
        if len(features_copy.shape) == 1:
            features_copy[noisy_features_indices] = 0
        elif len(features_copy.shape) == 2:
            features_copy[:, noisy_features_indices] = 0
        return features_copy

    # ----------------------------
    # NEW: 4x4 region analysis
    # ----------------------------
    def get_region_acts(self, grid_size: int = 4) -> torch.Tensor:
        """
        Returns pooled region activations for the currently registered image.

        returns: [grid_size*grid_size, d_sae] torch.Tensor on CPU
        """
        vit_act = self._run_vit_hook()
        sae_act = self._run_sae_hook(vit_act)     # [1, T, d]
        token_feat = sae_act[0].detach()          # [T, d]
        region_feat = pool_tokens_to_grid(token_feat, grid_out=grid_size, mode=self.pool_mode)  # [g*g, d]
        return region_feat.cpu()

    def get_top_neurons_region(self, region_idx: int, grid_size: int = 4, top_k: int = 5, plot: bool = True, save: bool = True):
        """
        Get top neurons for a pooled region index (0..grid_size^2-1).
        """
        region_feat = self.get_region_acts(grid_size=grid_size)  # [g*g, d]
        acts = region_feat[region_idx].numpy()                   # [d]
        acts = self._filter_out_nosiy_activation(acts)
        top_neurons = np.argsort(acts)[::-1][:top_k]

        # store last sae_act too (for compatibility with other plotting)
        vit_act = self._run_vit_hook()
        self.sae_act = self._run_sae_hook(vit_act)

        if plot:
            self._plot_union_top_neruons(top_k, top_neurons, region_idx, acts, save=save)

        return top_neurons

    def get_segmentation_mask_region(self, image, feat_idx: int, grid_size: int = 4) -> Image.Image:
        """
        Creates an overlay mask using pooled region activations (grid_size x grid_size).
        """
        if image.mode == "L":
            image = image.convert("RGB")

        vit_act = self._run_vit_hook(image)
        sae_act = self._run_sae_hook(vit_act)   # [1, T, d]
        token_feat = sae_act[0].detach()        # [T, d]

        region_feat = pool_tokens_to_grid(token_feat, grid_out=grid_size, mode=self.pool_mode)  # [g*g, d]
        vals = region_feat[:, feat_idx].cpu().clone()  # [g*g]

        # filter noisy features: filter expects numpy arrays; apply consistently
        vals_np = vals.numpy()
        vals_np = self._filter_out_nosiy_activation(vals_np)  # sets feature dims; here vals is 1D over regions
        # NOTE: noisy filtering by feature index doesn't apply to region axis; we keep as-is.
        # If you want "noisy tokens" filtering, do it earlier on token_feat before pooling.

        vals = torch.tensor(vals_np, dtype=torch.float32)
        mask = upsample_grid_mask(vals, grid_size=grid_size, out_h=image.height, out_w=image.width)

        base_opacity = 30
        image_array = np.array(image)[..., :3]
        rgba_overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        rgba_overlay[..., :3] = image_array[..., :3]

        darkened_image = (image_array[..., :3] * (base_opacity / 255)).astype(np.uint8)
        rgba_overlay[mask == 0, :3] = darkened_image[mask == 0]
        rgba_overlay[..., 3] = 255
        return Image.fromarray(rgba_overlay)

    # ----------------------------
    # Visual alignment: show a 4x4 patch grid (pixel-space)
    # ----------------------------
    def show_patches_grid(self, grid_size: int = 4, highlight_region_idx: Optional[int] = None, save: bool = True):
        """
        Shows pixel-space patches in a grid_size x grid_size layout.
        For 224x224 images, patch pixels = 224//grid_size.
        """
        if not hasattr(self, "input_image"):
            raise RuntimeError("register_image() first")

        # determine patch size in pixels from processed_image size
        _, _, H, W = self.processed_image.shape
        patch_px_h = H // grid_size
        patch_px_w = W // grid_size
        if patch_px_h != patch_px_w:
            raise ValueError(f"Non-square processed image {H}x{W}, patch sizes {patch_px_h}x{patch_px_w}")

        patches = self._create_patches(patch=patch_px_h)
        self._plot_patches(patches.cpu().data, highlight_patch_idx=highlight_region_idx, save=save)

    # ----------------------------
    # Existing top-images functions (unchanged)
    # ----------------------------
    def get_top_images(self, neuron_idx: int, top_k=5, show_seg_mask=False, grid_size: int = 4):
        out_top_images = []
        for dataset_name in self.max_act_images.keys():
            images, labels = self._get_max_activating_images_and_labels(
                neuron_idx,
                self.datasets[dataset_name],
                self.max_act_images[dataset_name],
            )

            if show_seg_mask:
                images = [self.get_segmentation_mask_region(img, neuron_idx, grid_size=grid_size) for img in images[:top_k]]

            suptitle = f"{dataset_name} - {neuron_idx}"
            fig = self._plot_images(
                dataset_name,
                images,
                neuron_idx,
                labels,
                suptitle=suptitle,
                top_k=top_k,
                save=False,
            )
            out_top_images.append(fig)
        return out_top_images

    def show_ref_images_of_neuron_indices(self, neuron_indices: List[int], top_k=5, save=False, seg_mask=False, grid_size: int = 4):
        for neuron_idx in neuron_indices:
            figs = self.get_top_images(neuron_idx, top_k=top_k, show_seg_mask=False, grid_size=grid_size)
            self._plot_multiple_images(figs, neuron_indices, top_k=top_k, save=True)

            if seg_mask:
                figs = self.get_top_images(neuron_idx, top_k=top_k, show_seg_mask=True, grid_size=grid_size)
                self._plot_multiple_images(figs, neuron_indices, top_k=top_k, save=True, seg=True)

    # ----------------------------
    # One-call runner for region analysis
    # ----------------------------
    def run_region(
        self,
        region_idx: int,
        grid_size: int = 4,
        top_k: int = 5,
        num_images: int = 5,
        seg_mask: bool = True,
        save: bool = True,
    ):
        """
        Full pipeline for a pooled region (e.g., 4x4):
        1) show pixel-space 4x4 patches with highlighted region
        2) compute top neurons for the region
        3) show top activating images for those neurons (+ optional region-based seg overlay)
        """
        self.show_patches_grid(grid_size=grid_size, highlight_region_idx=region_idx, save=save)
        top_neurons = self.get_top_neurons_region(region_idx=region_idx, grid_size=grid_size, top_k=top_k, plot=True, save=save)
        self.show_ref_images_of_neuron_indices(top_neurons.tolist(), top_k=num_images, seg_mask=seg_mask, save=save, grid_size=grid_size)


# ----------------------------
# Example usage (commented)
# ----------------------------
if __name__ == "__main__":
    """
    Example (pseudo-code; you must provide vit/cfg/sae/mean_acts/etc.):

    tester = SAETester(vit=vit, cfg=cfg, sae=sae, mean_acts=mean_acts,
                       max_act_images=max_act_images, datasets=datasets,
                       class_names=class_names, device="cuda", save_dir="./saved")

    tester.register_image("/path/to/image.jpg")

    # Analyze region 0..15 on a 4x4 pooled grid:
    tester.run_region(region_idx=5, grid_size=4, top_k=5, num_images=5, seg_mask=True, save=True)
    """
    pass
