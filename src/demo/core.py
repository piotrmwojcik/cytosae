import os
from copy import deepcopy
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image


# =====================================================================
#                               UTIL MIXIN
# =====================================================================

class UtilMixin:
    def _get_max_activating_images_and_labels(
        self, neuron_idx, dataset, max_activating_image_indices
    ):
        img_list = max_activating_image_indices[neuron_idx]
        images, labels = [], []

        for i in img_list:
            try:
                sample = dataset[i.item()]
                images.append(sample["image"])
                labels.append(sample.get("label", 0))
            except:
                images.append(dataset[i.item()]["image"])
                labels.append(0)
        return images, labels

    # -------------------------------------------------------------
    #               CORRECT PATCH EXTRACTION FOR VIT
    # -------------------------------------------------------------
    def _create_patches(self, patch=16):
        """
        Extract non-overlapping patches from a 448×448 processed image.

        Input: self.processed_image → [1, C, H, W]
        Output: patches → [1, n_h, n_w, C, patch, patch]
        """
        temp = self.processed_image  # [1, C, H, W]
        assert temp.dim() == 4, f"Expected [B,C,H,W], got {temp.shape}"
        B, C, H, W = temp.shape
        assert B == 1

        assert H % patch == 0 and W % patch == 0, \
            f"Image size ({H},{W}) not divisible by patch {patch}"

        # unfold over H
        patches = temp.unfold(2, patch, patch)      # [1,C,n_h,patch,W]
        # unfold over W
        patches = patches.unfold(3, patch, patch)   # [1,C,n_h,patch,n_w,patch]

        # reorder → [1,n_h,n_w,C,patch,patch]
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        return patches


# =====================================================================
#                           VISUALIZATION MIXIN
# =====================================================================

class VisualizeMixin:

    def _plot_input_image(self, save=True):
        plt.imshow(self.input_image)
        plt.axis("off")
        if save:
            img_name = os.path.basename(self.img_url).split(".")[0]
            save_name = f"{self.save_dir}/{img_name}/input_image.png"
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            plt.savefig(save_name, bbox_inches="tight", dpi=300)
        plt.close()

    def _plot_patches(self, patches, highlight_patch_idx=None, save=True):
        n_h = patches.size(1)
        n_w = patches.size(2)

        fig, axs = plt.subplots(n_h, n_w, figsize=(6, 6))
        plt.subplots_adjust(wspace=0.01, hspace=0.01)

        for i in range(n_h):
            for j in range(n_w):
                patch = patches[0, i, j]  # [C, patch, patch]
                patch = patch.permute(1, 2, 0).cpu().numpy()  # → HWC
                axs[i, j].imshow(patch)
                axs[i, j].axis("off")

                if highlight_patch_idx == i * n_w + j:
                    for spine in axs[i, j].spines.values():
                        spine.set_edgecolor("red")
                        spine.set_linewidth(3)

        if save:
            img_name = os.path.basename(self.img_url).split(".")[0]
            save_name = f"{self.save_dir}/{img_name}/patches.png"
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            plt.savefig(save_name, dpi=300)

        plt.close(fig)

    def _plot_feature_mask(self, patches, feat_idx, mask=None, plot=True, save=True):
        if mask is None:
            mask = self.sae_act[0, :, feat_idx].cpu()

        n_h = patches.size(1)
        n_w = patches.size(2)

        fig, axs = plt.subplots(n_h, n_w, figsize=(6, 6))
        plt.subplots_adjust(wspace=0.01, hspace=0.01)

        for i in range(n_h):
            for j in range(n_w):
                patch = patches[0, i, j].permute(1, 2, 0)
                patch = patch.cpu().numpy()

                m = mask[i * n_w + j + 1].item()
                masked_patch = patch * m
                masked_patch = (masked_patch - masked_patch.min()) / (
                    masked_patch.max() - masked_patch.min() + 1e-8
                )

                axs[i, j].imshow(masked_patch)
                axs[i, j].axis("off")

        if save:
            img_name = os.path.basename(self.img_url).split(".")[0]
            save_name = f"{self.save_dir}/{img_name}/feature_masks/{feat_idx}.png"
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            fig.savefig(save_name, dpi=300)

        plt.close(fig)
        return fig

    def _fig_to_img(self, fig):
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        img = img.reshape(canvas.get_width_height()[::-1] + (3,))
        return img

    def _plot_images(
        self,
        dataset_name,
        images,
        neuron_idx,
        labels=None,
        suptitle=None,
        top_k=5,
        save=True,
    ):
        # Resize to 448x448
        images = [img.resize((448, 448)) for img in images]

        num_cols = min(top_k, 5)
        num_rows = (top_k + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4.5 * num_cols, 5 * num_rows))
        axes = axes.flatten()

        for i in range(top_k):
            axes[i].imshow(images[i])
            axes[i].axis("off")

        if save:
            img_name = os.path.basename(self.img_url).split(".")[0]
            save_name = f"{self.save_dir}/{img_name}/top_images/{dataset_name}/{neuron_idx}.png"
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            plt.savefig(save_name, dpi=300)

        plt.close(fig)
        return fig


# =====================================================================
#                               SAE TESTER
# =====================================================================

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

    # -------------------------------------------------------------
    #                       IMAGE LOADING
    # -------------------------------------------------------------

    def show_input_image(self, save=True):
        """Show (and optionally save) the currently registered input image."""
        self._plot_input_image(save=save)

    def register_image(self, img_url: str):
        if isinstance(img_url, str):
            image = self._load_image(img_url)
        else:
            image = img_url

        if image.mode != "RGB":
            image = image.convert("RGB")

        self.input_image = image
        self.img_url = img_url

        # Resize to 448×448 because ViT patching expects fixed size
        resized = image.resize((448, 448))

        processed = self.vit.processor(resized)
        self.processed_image = processed.unsqueeze(0)

    def _load_image(self, img_url):
        if img_url.startswith("http"):
            response = requests.get(img_url)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        return Image.open(img_url)

    # -------------------------------------------------------------
    #                   CORE ACTIVATION EXTRACTION
    # -------------------------------------------------------------

    def _run_vit_hook(self, image=None):
        if image is None:
            inputs = self.processed_image.to(self.device)
        else:
            if image.mode != "RGB":
                image = image.convert("RGB")
            resized = image.resize((448, 448))
            inputs = self.vit.processor(resized).unsqueeze(0).to(self.device)

        list_of_hook_locations = [(self.cfg.block_layer, self.cfg.module_name)]
        vit_out, vit_cache_dict = self.vit.run_with_cache(
            list_of_hook_locations, inputs
        )
        return vit_cache_dict[(self.cfg.block_layer, self.cfg.module_name)]

    def _run_sae_hook(self, vit_act):
        sae_out, sae_cache_dict = self.sae.run_with_cache(vit_act)
        sae_act = sae_cache_dict["hook_hidden_post"]
        if sae_act.shape[0] != 1:  # batch dimension mismatch
            sae_act = sae_act.permute(1, 0, 2)
        return sae_act[:, 1:, :]  # remove CLS token

    # -------------------------------------------------------------
    #                   PATCH DISPLAY & SEG MASK
    # -------------------------------------------------------------

    def show_patches(self, highlight_patch_idx=None, patch_size=16, save=True):
        patches = self._create_patches(patch=patch_size)
        self._plot_patches(patches.cpu(), highlight_patch_idx, save)

    # -------------------------------------------------------------
    #                   SEGMENTATION MASK
    # -------------------------------------------------------------

    def get_segmentation_mask(self, image, feat_idx):
        if image.mode != "RGB":
            image = image.convert("RGB")

        vit_act = self._run_vit_hook(image)
        sae_act = self._run_sae_hook(vit_act)
        token_act = sae_act[0].detach().cpu().numpy()

        temp = token_act[:, feat_idx]  # shape [patches]

        num_patches = temp.shape[0]
        grid = int(np.sqrt(num_patches))
        assert grid * grid == num_patches, "Not a square grid of patches."

        mask = torch.Tensor(temp.reshape(grid, grid)).view(1, 1, grid, grid)
        mask = torch.nn.functional.interpolate(mask, (image.height, image.width))[0][0].numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-10)

        img_arr = np.array(image)
        opacity = 30

        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[..., :3] = img_arr[..., :3]
        dark = (img_arr[..., :3] * (opacity / 255)).astype(np.uint8)
        rgba[mask == 0, :3] = dark[mask == 0]
        rgba[..., 3] = 255

        return Image.fromarray(rgba)
