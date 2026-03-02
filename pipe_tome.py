import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from torch.nn import functional as F


from diffusers.image_processor import PipelineImageInput

from diffusers.utils import (
    deprecate,
    logging,
)

from diffusers.pipelines.stable_diffusion_xl.pipeline_output import (
    StableDiffusionXLPipelineOutput,
)


from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline

from utils.ptp_utils import AttentionStore, aggregate_attention, register_self_time
from torchvision import transforms as T


logger = logging.get_logger(__name__)


def token_merge(
    prompt_embeds: torch.Tensor, idx_merge: List[List[int]]
) -> torch.Tensor:
    """
    prompt_embeds: 77 dim
    idx_merge: [ [[1],[2]],[[3],[4]] ]
    """

    for idxs in idx_merge:
        noun_idx = idxs[0][0]
        alpha = 1.1
        prompt_embeds[noun_idx] = alpha * prompt_embeds[idxs[0]].sum(
            dim=0
        ) + 1.2 * prompt_embeds[idxs[1]].sum(dim=0)
        if len(idxs[0]) > 1:
            prompt_embeds[idxs[0][1:]] = 0
        prompt_embeds[idxs[1]] = 0

    return prompt_embeds


def get_centroid(attn_map: torch.Tensor) -> torch.Tensor:
    """
    attn_map: h*w*token_len
    """
    h, w, seq_len = attn_map.shape

    attn_x, attn_y = attn_map.sum(0), attn_map.sum(1)  # w|h seq_len
    x = torch.linspace(0, 1, w).to(attn_map.device).reshape(w, 1)
    y = torch.linspace(0, 1, h).to(attn_map.device).reshape(h, 1)

    centroid_x = (x * attn_x).sum(0) / attn_x.sum(0)  # seq_len
    centroid_y = (y * attn_y).sum(0) / attn_y.sum(0)  # bs seq_len
    centroid = torch.stack((centroid_x, centroid_y), -1)  # (seq_len, 2)
    return centroid


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed].
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class tomePipeline(StableDiffusionXLPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion XL.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion XL uses the text portion of
            [CLIP], specifically
            the [clip-vit-large-patch14]variant.
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP],
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k]
            variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer].
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer].
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        add_watermarker (`bool`, *optional*):
            Whether to use the [invisible_watermark library] to
            watermark output images. If not defined, it will default to True if the package is installed, otherwise no
            watermarker will be used.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "feature_extractor",
    ]
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
    ]

    def _entropy_loss(
        self,
        attention_store: AttentionStore,
        indices_to_alter: List[int],
        attention_res: int = 16,
        pose_loss: bool = False,
    ):
        """Aggregates the attention for each token and computes the max activation value for each token to alter."""
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0,
        )  # h w 77

        loss = 0

        prompt = self.prompt[0] if isinstance(self.prompt, list) else self.prompt
        last_idx = len(self.tokenizer(prompt)["input_ids"]) - 1

        attention_for_text = attention_maps[:, :, 1:last_idx]
        attention_for_text = torch.nn.functional.softmax(
            attention_for_text / 0.5, dim=-1
        )

        # get pos idx and calculate pos loss
        indices = []
        for i in range(len(indices_to_alter)):
            curr_idx = indices_to_alter[i][0][0]
            indices.append(curr_idx)

        indices = [i - 1 for i in indices]
        cross_map = attention_for_text[:, :, indices]  # 32,32 seq_len
        cross_map = (cross_map - cross_map.amin(dim=(0, 1), keepdim=True)) / (
            cross_map.amax(dim=(0, 1), keepdim=True)
            - cross_map.amin(dim=(0, 1), keepdim=True)
        )
        cross_map = cross_map / cross_map.sum(dim=(0, 1), keepdim=True)

        loss = loss - 2 * (cross_map * torch.log(cross_map + 1e-5)).sum()
        if pose_loss:
            idx = 0
            for subject_idx, subject_idx2 in [indices]:
                # Shift indices since we removed the first token
                curr_map = attention_for_text[
                    :, :, [subject_idx, subject_idx2]
                ]  # h w k

                vis_map = curr_map.permute(2, 0, 1)  # k h w
                sub_map, sub_map2 = vis_map[0], vis_map[1]

                sub_map = (sub_map - sub_map.min()) / (sub_map.max() - sub_map.min())
                sub_map2 = (sub_map2 - sub_map2.min()) / (
                    sub_map2.max() - sub_map2.min()
                )

                curr_map = torch.stack([sub_map, sub_map2])  # k h w
                curr_map = curr_map.permute(1, 2, 0)  # h w k
                pair_pos = get_centroid(curr_map) * 32  # (2, 2) k 2

                pos1 = torch.tensor([10.0, 16]).to("cuda")

                pos2 = torch.tensor([25.0, 16]).to("cuda")

                loss = loss + (0.2 * (pair_pos[0] - pos1) ** 2).mean()
                loss = loss + (0.2 * (pair_pos[1] - pos2) ** 2).mean()

                T.ToPILImage()(sub_map.reshape(1, 32, 32)).save("mask_left.png")
                T.ToPILImage()(sub_map2.reshape(1, 32, 32)).save("mask_right.png")
        return loss

    @staticmethod
    def _update_latent(
        latents: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [latents], retain_graph=True
        )[0]
        latents = latents - 0.5 * step_size * grad_cond
        return latents

    @staticmethod
    def _update_text(
        text_embeddings: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [text_embeddings], retain_graph=True
        )[0]
        text_embeddings = text_embeddings - step_size * grad_cond
        return text_embeddings

    def _perform_iterative_refinement_step(
        self,
        latents: torch.Tensor,
        indices_to_alter: List[Tuple[int, int]],
        threshold: float,
        text_embeddings: torch.Tensor,
        attention_store: AttentionStore,
        step_size: float,
        t: int,
        attention_res: int = 32,
        max_refinement_steps: List[int] = [3, 3],
        pose_loss: bool = False,
    ):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code and text embedding according to our loss objective until the given threshold is reached for all tokens.
        """
        threshold = threshold / 2 * len(indices_to_alter)
        threshold -= 2
        ratio = t / 1000
        if ratio > 0.9:
            max_refinement_steps = max_refinement_steps[0]
        if ratio <= 0.9:
            max_refinement_steps = max_refinement_steps[1]
        iteration = 0
        while True:
            iteration += 1
            torch.cuda.empty_cache()
            latents = latents.clone().detach().requires_grad_(True)
            text_embeddings = text_embeddings.clone().detach().requires_grad_(True)

            noise_pred_text = self.unet(
                latents,
                t,
                encoder_hidden_states=text_embeddings[1].unsqueeze(0),
                timestep_cond=self.timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=self.added_cond_kwargs2,
            ).sample

            loss = self._entropy_loss(
                attention_store, indices_to_alter, attention_res, pose_loss=pose_loss
            )
            if loss != 0:  # and t/1000 > 0.8:
                latents = self._update_latent(latents, loss, step_size)
                text_embeddings = self._update_text(text_embeddings, loss, step_size)

            if loss < threshold:
                break
            if iteration >= max_refinement_steps:
                print(
                    f"Entropy loss optimization Exceeded max number of iterations ({max_refinement_steps}) "
                )
                break

        return latents, loss, text_embeddings.detach()

    @staticmethod
    def _update_stoken(
        stoken: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update the merged token according to the computed loss."""
        loss = loss * step_size
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [stoken])[0]
        stoken = stoken - grad_cond
        return stoken

    def opt_token(self, latents: torch.Tensor, t, stoken, prompt_anchor, iter_num=3):
        """
        latents: 128 128 4
        stoken: dim
        prompt_anchor: 77 dim
        """
        stoken.requires_grad_(True)

        latents = latents.clone().detach().unsqueeze(0)
        iteration = 0

        with torch.no_grad():
            noise_pred_anchor = self.unet(
                latents,
                t,
                encoder_hidden_states=prompt_anchor,
                timestep_cond=self.timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=self.added_cond_kwargs2,
            ).sample
        while True:
            iteration += 1
            noise_pred_token = self.unet(
                latents,
                t,
                encoder_hidden_states=stoken.unsqueeze(0).unsqueeze(0),
                timestep_cond=self.timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=self.added_cond_kwargs2,
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred_anchor, noise_pred_token)

            stoken = self._update_stoken(stoken, loss, 10000)
            if iteration >= iter_num:
                print(
                    f"Semantic binding loss optimization Exceeded max number of iterations ({iter_num}) "
                )
                break

        with torch.no_grad():
            noise_pred_null = self.unet(
                latents,
                t,
                encoder_hidden_states=self.negative_prompt_embeds,
                timestep_cond=self.timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=self.added_cond_kwargs2,
            ).sample

            noise_pred = noise_pred_null + self.guidance_scale * (
                noise_pred_null - noise_pred_anchor
            )

            noise_pred = rescale_noise_cfg(
                noise_pred,
                noise_pred_anchor,
                guidance_rescale=self.guidance_rescale,
            )
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            self.scheduler._step_index -= 1
        return stoken, latents[0]

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):

        callback = None
        callback_steps = None

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        attention_store = kwargs.get("attention_store")
        indices_to_alter = kwargs.get("indices_to_alter")
        attention_res = kwargs.get("attention_res")
        run_standard_sd = kwargs.get("run_standard_sd")
        thresholds = kwargs.get("thresholds")
        scale_factor = kwargs.get("scale_factor")
        scale_range = kwargs.get("scale_range")
        smooth_attentions = kwargs.get("smooth_attentions")
        sigma = kwargs.get("sigma")
        kernel_size = kwargs.get("kernel_size")
        prompt_anchor = kwargs.get("prompt_anchor")
        prompt3 = kwargs.get("prompt3")
        prompt_length = kwargs.get("prompt_length")
        token_refinement_steps = kwargs.get("token_refinement_steps")
        attention_refinement_steps = kwargs.get("attention_refinement_steps")
        tome_control_steps = kwargs.get("tome_control_steps")
        eot_replace_step = kwargs.get("eot_replace_step")
        use_pose_loss = kwargs.get("use_pose_loss")
        use_token_merge = kwargs.get("use_token_merge", True)
        use_ets = kwargs.get("use_ets", True)

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        self.prompt = prompt
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        panchors = []
        for panchor in prompt_anchor:
            (
                prompt_anchor_emb,
                _,
                _,
                _,
            ) = self.encode_prompt(
                prompt=panchor,
                prompt_2=panchor,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
            )
            panchors.append(prompt_anchor_emb)

        (
            prompt_anchor3,
            _,
            _,
            _,
        ) = self.encode_prompt(
            prompt=prompt3,
            prompt_2=prompt3,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # stoken1, stoken2 = prompt_embeds[0,2], prompt_embeds[0,6]
        # -----------------------------------
        # token merge
        if not run_standard_sd and token_refinement_steps and use_token_merge:
            prompt_embeds[0] = token_merge(prompt_embeds[0], indices_to_alter)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, add_text_embeds], dim=0
            )
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(
            batch_size * num_images_per_prompt, 1
        )

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 8. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        # 8.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(
                list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps))
            )
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self.timestep_cond = timestep_cond
        self._num_timesteps = len(timesteps)
        self.timesteps = timesteps

        scale_range = np.linspace(
            scale_range[0], scale_range[1], len(self.scheduler.timesteps)
        )

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            added_cond_kwargs["image_embeds"] = image_embeds

        # added_cond_kwargs2 = {"text_embeds": add_text_embeds[1:], "time_ids": add_time_ids[1:]}

        added_cond_kwargs2 = {
            "text_embeds": torch.zeros_like(add_text_embeds[1:]),
            "time_ids": add_time_ids[1:],
        }

        self.added_cond_kwargs2 = added_cond_kwargs2
        self.negative_prompt_embeds = negative_prompt_embeds
        self.pos = None

        # del self.text_encoder, self.text_encoder_2
        prompt_embeds2 = None
        latent_anchor = None
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                register_self_time(self, None)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_anchor = (
                    torch.cat([latents] * len(panchors))
                    if latent_anchor is None
                    else latent_anchor
                )

                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                latent_anchor = self.scheduler.scale_model_input(latent_anchor, t)

                latents_up = (
                    latent_model_input[1:].clone().detach()
                )  # .requires_grad_(True)

                prompt_embeds2 = (
                    prompt_embeds if prompt_embeds2 is None else prompt_embeds2
                )

                with torch.enable_grad():
                    if not run_standard_sd:
                        token_control, attention_control = tome_control_steps
                        # EOT replace
                        if i == eot_replace_step and use_ets:
                            prompt_embeds2[1, prompt_length + 1 :] = prompt_anchor3[0][
                                prompt_length + 1 :]
                        # semantic binding loss for token refinement
                        if i < token_control:
                            for idx, panchor in enumerate(panchors):
                                stoken = (
                                    prompt_embeds2[1, indices_to_alter[idx][0][0]]
                                    .detach()
                                    .clone()
                                )
                                stoken, latent_anchor[idx] = self.opt_token(
                                    latent_anchor[idx],
                                    t,
                                    stoken,
                                    panchor,
                                    token_refinement_steps,
                                )
                                prompt_embeds2[1, indices_to_alter[idx][0][0]] = stoken
                        # entropy loss for attention refinement
                        if i < attention_control:
                            latents_up, loss, prompt_embeds2 = (
                                self._perform_iterative_refinement_step(
                                    latents=latents_up,
                                    indices_to_alter=indices_to_alter,
                                    threshold=thresholds[i],
                                    text_embeddings=prompt_embeds2,
                                    attention_store=attention_store,
                                    step_size=scale_factor * scale_range[i],
                                    t=t,
                                    attention_res=attention_res,
                                    max_refinement_steps=attention_refinement_steps,
                                    pose_loss=use_pose_loss,
                                )
                            )

                            print(f"Iteration {i} | Loss: {loss:0.4f}")

                latent_model_input = (
                    torch.cat([latents_up] * 2)
                    if self.do_classifier_free_guidance
                    else latents_up
                )
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds2,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = (
                self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            )

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(
                    next(iter(self.vae.post_quant_conv.parameters())).dtype
                )

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = (
                hasattr(self.vae.config, "latents_mean")
                and self.vae.config.latents_mean is not None
            )
            has_latents_std = (
                hasattr(self.vae.config, "latents_std")
                and self.vae.config.latents_std is not None
            )
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents = (
                    latents * latents_std / self.vae.config.scaling_factor
                    + latents_mean
                )
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)
