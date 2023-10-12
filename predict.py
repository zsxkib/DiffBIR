# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from pprint import pprint
from cog import BasePredictor, Input, Path
from typing import List, Tuple, Optional

import os
import math
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import pytorch_lightning as pl

from ldm.xformers_state import auto_xformers_status
from model.cldm import ControlLDM
from utils.common import instantiate_from_config, load_state_dict
from utils.file import get_file_name_parts
from utils.image import auto_resize, pad
from utils.file import load_file_from_url
from utils.face_restoration_helper import FaceRestoreHelper

import einops

from model.spaced_sampler import SpacedSampler
from model.cond_fn import MSEGuidance
from dataclasses import asdict, dataclass


pretrained_models = {
    "general_v1": {
        "ckpt_url": "https://huggingface.co/lxq007/DiffBIR/resolve/main/general_full_v1.ckpt",
        "swinir_url": "https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt",
    },
    "face_v1": {
        "ckpt_url": "https://huggingface.co/lxq007/DiffBIR/resolve/main/face_full_v1.ckpt"
    },
}


@dataclass
class Arguments:
    # model
    ckpt: str = "weights/face_full_v1.ckpt"
    config: str = "configs/model/cldm.yaml"  # This was not in predict parameters
    reload_swinir: bool = False
    swinir_ckpt: str = "general_swinir_v1"

    # input and preprocessing
    input: str = None
    steps: int = 50
    sr_scale: int = 4
    image_size: int = 512
    repeat_times: int = 1
    disable_preprocess_model: bool = False

    # patch-based sampling
    tiled: bool = False
    tile_size: int = 512
    tile_stride: int = 256

    # latent image guidance
    use_guidance: bool = False
    g_scale: float = 0.0
    g_t_start: int = 1001
    g_t_stop: int = -1
    g_space: str = "latent"
    g_repeat: int = 5

    # face related
    has_aligned: bool = False
    only_center_face: bool = False
    detection_model: str = "retinaface_resnet50"

    # background upsampler
    bg_upsampler: str = "RealESRGAN"
    bg_tile: int = 400
    bg_tile_stride: int = 200

    # postprocessing and saving
    color_fix_type: str = "wavelet"
    output: str = "."
    show_lq: bool = False
    skip_if_exist: bool = False

    # miscellaneous
    seed: int = 231
    device: str = "cuda"


@torch.no_grad()
def process(
    model: ControlLDM,
    control_imgs: List[np.ndarray],
    steps: int,
    strength: float,
    color_fix_type: str,
    disable_preprocess_model: bool,
    cond_fn: Optional[MSEGuidance],
    tiled: bool,
    tile_size: int,
    tile_stride: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Apply DiffBIR model on a list of low-quality images.

    Args:
        model (ControlLDM): Model.
        control_imgs (List[np.ndarray]): A list of low-quality images (HWC, RGB, range in [0, 255]).
        steps (int): Sampling steps.
        strength (float): Control strength. Set to 1.0 during training.
        color_fix_type (str): Type of color correction for samples.
        disable_preprocess_model (bool): If specified, preprocess model (SwinIR) will not be used.
        cond_fn (Guidance | None): Guidance function that returns gradient to guide the predicted x_0.
        tiled (bool): If specified, a patch-based sampling strategy will be used for sampling.
        tile_size (int): Size of patch.
        tile_stride (int): Stride of sliding patch.

    Returns:
        preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 255]).
        stage1_preds (List[np.ndarray]): Outputs of preprocess model (HWC, RGB, range in [0, 255]).
            If `disable_preprocess_model` is specified, then preprocess model's outputs is the same
            as low-quality inputs.
    """
    n_samples = len(control_imgs)
    sampler = SpacedSampler(model, var_type="fixed_small")
    control = torch.tensor(
        np.stack(control_imgs) / 255.0, dtype=torch.float32, device=model.device
    ).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()

    if not disable_preprocess_model:
        control = model.preprocess_model(control)
    model.control_scales = [strength] * 13

    if cond_fn is not None:
        cond_fn.load_target(2 * control - 1)

    height, width = control.size(-2), control.size(-1)
    shape = (n_samples, 4, height // 8, width // 8)
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)
    if not tiled:
        samples = sampler.sample(
            steps=steps,
            shape=shape,
            cond_img=control,
            positive_prompt="",
            negative_prompt="",
            x_T=x_T,
            cfg_scale=1.0,
            cond_fn=cond_fn,
            color_fix_type=color_fix_type,
        )
    else:
        samples = sampler.sample_with_mixdiff(
            tile_size=tile_size,
            tile_stride=tile_stride,
            steps=steps,
            shape=shape,
            cond_img=control,
            positive_prompt="",
            negative_prompt="",
            x_T=x_T,
            cfg_scale=1.0,
            cond_fn=cond_fn,
            color_fix_type=color_fix_type,
        )
    x_samples = samples.clamp(0, 1)
    x_samples = (
        (einops.rearrange(x_samples, "b c h w -> b h w c") * 255)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )
    control = (
        (einops.rearrange(control, "b c h w -> b h w c") * 255)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )

    preds = [x_samples[i] for i in range(n_samples)]
    stage1_preds = [control[i] for i in range(n_samples)]

    return preds, stage1_preds


def build_diffbir_model(model_config, ckpt, swinir_ckpt=None):
    """'
    model_config: model architecture config file.
    ckpt: checkpoint file path of the main model.
    swinir_ckpt: checkpoint file path of the swinir model.
        load swinir from the main model if set None.
    """
    weight_root = os.path.dirname(ckpt)
    swinir_url = pretrained_models["general_v1"]["swinir_url"]

    # download ckpt automatically if ckpt not exist in the local path
    if "general_full_v1" in ckpt:
        ckpt_url = pretrained_models["general_v1"]["ckpt_url"]
        if swinir_ckpt is None:
            swinir_ckpt = f"{weight_root}/general_swinir_v1.ckpt"
            swinir_url = pretrained_models["general_v1"]["swinir_url"]
    elif "face_full_v1" in ckpt:
        # swinir ckpt is already included in the main model
        ckpt_url = pretrained_models["face_v1"]["ckpt_url"]
    else:
        # define a custom diffbir model
        raise NotImplementedError("undefined diffbir model type!")

    if not os.path.exists(ckpt):
        ckpt = load_file_from_url(ckpt_url, weight_root)
    if swinir_ckpt is not None and not os.path.exists(swinir_ckpt):
        swinir_ckpt = load_file_from_url(swinir_url, weight_root)

    model: ControlLDM = instantiate_from_config(OmegaConf.load(model_config))
    load_state_dict(model, torch.load(ckpt), strict=True)
    # reload preprocess model if specified
    if swinir_ckpt is not None:
        if not hasattr(model, "preprocess_model"):
            raise ValueError(f"model don't have a preprocess model.")
        print(f"reload swinir model from {swinir_ckpt}")
        load_state_dict(model.preprocess_model, torch.load(swinir_ckpt), strict=True)
    model.freeze()
    return model


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.config = "configs/model/cldm.yaml"
        self.model: ControlLDM = instantiate_from_config(OmegaConf.load(self.config))
        self.mode = None
        self.switch_model(mode="FULL", args=None)

    def switch_model(self, mode, args=None):
        # Check if mode hasn't changed
        if self.mode == mode:
            print(f"Mode hasn't changed, remains at '{self.mode}'. No need to switch.")
            return

        print(f"Switching from mode '{self.mode}' to '{mode}'...")
        self.mode = mode
        if self.mode == "FULL":
            print("Loading 'FULL' mode model...")
            load_state_dict(
                self.model,
                torch.load("weights/general_full_v1.ckpt", map_location="cpu"),
                strict=True,
            )

            # reload preprocess model if specified
            if args is not None:
                if args.reload_swinir:
                    if not hasattr(self.model, "preprocess_model"):
                        raise ValueError(f"model don't have a preprocess model.")
                    print(f"Reloading swinir model from '{args.swinir_ckpt}'...")
                    load_state_dict(
                        self.model.preprocess_model,
                        torch.load(args.swinir_ckpt, map_location="cpu"),
                        strict=True,
                    )
            print("Freezing the 'FULL' mode model and moving to the desired device...")
            self.model.freeze()
            self.model.to("cuda")
        else:  # FACE
            print("Building and loading 'FACE' mode model...")
            self.model = build_diffbir_model(
                self.config, args.ckpt, args.swinir_ckpt
            ).to("cuda")
        auto_xformers_status("cuda")
        print(f"Model successfully switched to '{mode}' mode.")

    def full_pipeline(self, args):
        args_dict = asdict(args)
        pprint(args_dict)
        pl.seed_everything(args.seed)

        file_path = args.input

        lq = Image.open(file_path).convert("RGB")
        if args.sr_scale != 1:
            lq = lq.resize(
                tuple(math.ceil(x * args.sr_scale) for x in lq.size), Image.BICUBIC
            )
        if not args.tiled:
            lq_resized = auto_resize(lq, 512)
        else:
            lq_resized = auto_resize(lq, args.tile_size)
        x = pad(np.array(lq_resized), scale=64)

        for i in range(args.repeat_times):
            save_path = os.path.join(
                args.output, os.path.relpath(file_path, args.input)
            )
            parent_path, stem, _ = get_file_name_parts(save_path)
            save_path = os.path.join(parent_path, f"{stem}_{i}.png")

            # No need to check if the file exists, just ensure the directory exists and save
            os.makedirs(parent_path, exist_ok=True)

            # initialize latent image guidance
            if args.use_guidance:
                cond_fn = MSEGuidance(
                    scale=args.g_scale,
                    t_start=args.g_t_start,
                    t_stop=args.g_t_stop,
                    space=args.g_space,
                    repeat=args.g_repeat,
                )
            else:
                cond_fn = None

            preds, stage1_preds = process(
                self.model,
                [x],
                steps=args.steps,
                strength=1,
                color_fix_type=args.color_fix_type,
                disable_preprocess_model=args.disable_preprocess_model,
                cond_fn=cond_fn,
                tiled=args.tiled,
                tile_size=args.tile_size,
                tile_stride=args.tile_stride,
            )
            pred, stage1_pred = preds[0], stage1_preds[0]

            # remove padding
            pred = pred[: lq_resized.height, : lq_resized.width, :]
            stage1_pred = stage1_pred[: lq_resized.height, : lq_resized.width, :]

            if args.show_lq:
                pred = np.array(Image.fromarray(pred).resize(lq.size, Image.LANCZOS))
                stage1_pred = np.array(
                    Image.fromarray(stage1_pred).resize(lq.size, Image.LANCZOS)
                )
                lq = np.array(lq)
                images = (
                    [lq, pred]
                    if args.disable_preprocess_model
                    else [lq, stage1_pred, pred]
                )
                Image.fromarray(np.concatenate(images, axis=1)).save(save_path)
            else:
                Image.fromarray(pred).resize(lq.size, Image.LANCZOS).save(save_path)
            print(f"save to {save_path}")
            yield Path(save_path)

    def face_pipeline(self, args):
        args_dict = asdict(args)
        pprint(args_dict)

        img_save_ext = "png"
        pl.seed_everything(args.seed)

        # ------------------ set up FaceRestoreHelper -------------------
        face_helper = FaceRestoreHelper(
            device="cuda",
            upscale_factor=1,
            face_size=args.image_size,
            use_parse=True,
            det_model=args.detection_model,
        )

        # set up the backgrouns upsampler
        if args.bg_upsampler == "DiffBIR":
            # Loading two DiffBIR models consumes huge GPU memory capacity.
            bg_upsampler = build_diffbir_model(
                self.config, "weights/general_full_v1.pth"
            )
            bg_upsampler = bg_upsampler.to("cuda")
        elif args.bg_upsampler == "RealESRGAN":
            from utils.realesrgan.realesrganer import set_realesrgan

            # support official RealESRGAN x2 & x4 upsample model.
            # Using x2 upsampler as default if scale is not specified as 4.
            bg_upscale = int(args.sr_scale) if int(args.sr_scale) in [2, 4] else 2
            print(
                f"Loading RealESRGAN_x{bg_upscale}plus.pth for background upsampling..."
            )
            bg_upsampler = set_realesrgan(args.bg_tile, "cuda", bg_upscale)
        else:
            bg_upsampler = None

        file_path = args.input

        # read image
        lq = Image.open(file_path).convert("RGB")

        if args.sr_scale != 1:
            lq = lq.resize(
                tuple(math.ceil(x * args.sr_scale) for x in lq.size), Image.BICUBIC
            )
        lq_resized = auto_resize(lq, args.image_size)
        x = pad(np.array(lq_resized), scale=64)

        face_helper.clean_all()
        if args.has_aligned:
            # the input faces are already cropped and aligned
            face_helper.cropped_faces = [x]
        else:
            face_helper.read_image(x)
            # get face landmarks for each face
            face_helper.get_face_landmarks_5(
                only_center_face=args.only_center_face,
                resize=640,
                eye_dist_threshold=5,
            )
            face_helper.align_warp_face()

        parent_dir, img_basename, _ = get_file_name_parts(file_path)
        rel_parent_dir = os.path.relpath(parent_dir, args.input)
        output_parent_dir = os.path.join(args.output, rel_parent_dir)
        cropped_face_dir = os.path.join(output_parent_dir, "cropped_faces")
        restored_face_dir = os.path.join(output_parent_dir, "restored_faces")
        restored_img_dir = os.path.join(output_parent_dir, "restored_imgs")
        if not args.has_aligned:
            os.makedirs(cropped_face_dir, exist_ok=True)
            os.makedirs(restored_img_dir, exist_ok=True)
        os.makedirs(restored_face_dir, exist_ok=True)
        for i in range(args.repeat_times):
            basename = f"{img_basename}_{i}" if i else img_basename
            restored_img_path = os.path.join(
                restored_img_dir, f"{basename}.{img_save_ext}"
            )
            if os.path.exists(restored_img_path) or os.path.exists(
                os.path.join(restored_face_dir, f"{basename}.{img_save_ext}")
            ):
                if args.skip_if_exist:
                    print(f"Exists, skip face image {basename}...")
                    continue
                else:
                    raise RuntimeError(f"Image {basename} already exist")

            try:
                preds, stage1_preds = process(
                    self.model,
                    face_helper.cropped_faces,
                    steps=args.steps,
                    strength=1,
                    color_fix_type=args.color_fix_type,
                    disable_preprocess_model=args.disable_preprocess_model,
                    cond_fn=None,
                    tiled=False,
                    tile_size=None,
                    tile_stride=None,
                )
            except RuntimeError as e:
                # Avoid cuda_out_of_memory error.
                print(f"{file_path}, error: {e}")
                continue

            for restored_face in preds:
                # unused stage1 preds
                # face_helper.add_restored_face(np.array(stage1_restored_face))
                face_helper.add_restored_face(np.array(restored_face))

            # paste face back to the image
            if not args.has_aligned:
                # upsample the background
                if bg_upsampler is not None:
                    print(
                        f"upsampling the background image using {args.bg_upsampler}..."
                    )
                    if args.bg_upsampler == "DiffBIR":
                        bg_img, _ = process(
                            bg_upsampler,
                            [x],
                            steps=args.steps,
                            color_fix_type=args.color_fix_type,
                            strength=1,
                            disable_preprocess_model=args.disable_preprocess_model,
                            cond_fn=None,
                            tiled=False,
                            tile_size=None,
                            tile_stride=None,
                        )
                        bg_img = bg_img[0]
                    elif args.bg_upsampler == "RealESRGAN":
                        # resize back to the original size
                        w, h = x.shape[:2]
                        input_size = (
                            int(w / args.sr_scale),
                            int(h / args.sr_scale),
                        )
                        x = Image.fromarray(x).resize(input_size, Image.LANCZOS)
                        bg_img = bg_upsampler.enhance(
                            np.array(x), outscale=args.sr_scale
                        )[0]
                else:
                    bg_img = None
                face_helper.get_inverse_affine(None)

                # paste each restored face to the input image
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img
                )

            # save faces
            for idx, (cropped_face, restored_face) in enumerate(
                zip(face_helper.cropped_faces, face_helper.restored_faces)
            ):
                # save cropped face
                if not args.has_aligned:
                    save_crop_path = os.path.join(
                        cropped_face_dir, f"{basename}_{idx:02d}.{img_save_ext}"
                    )
                    Image.fromarray(cropped_face).save(save_crop_path)
                    yield Path(save_crop_path)
                # save restored face
                if args.has_aligned:
                    save_face_name = f"{basename}.{img_save_ext}"
                    # remove padding
                    restored_face = restored_face[
                        : lq_resized.height, : lq_resized.width, :
                    ]
                else:
                    save_face_name = f"{basename}_{idx:02d}.{img_save_ext}"
                save_restore_path = os.path.join(restored_face_dir, save_face_name)
                Image.fromarray(restored_face).save(save_restore_path)
                yield Path(save_restore_path)

            # save restored whole image
            if not args.has_aligned:
                # remove padding
                restored_img = restored_img[: lq_resized.height, : lq_resized.width, :]
                # save restored image
                Image.fromarray(restored_img).resize(lq.size, Image.LANCZOS).convert(
                    "RGB"
                ).save(restored_img_path)
                yield Path(restored_img_path)
            print(f"Face image {basename} saved to {output_parent_dir}")

    def predict(
        self,
        input: Path = Input(
            description="Path to the input image you want to enhance.",
        ),
        upscaling_model_type: str = Input(
            description="Choose the type of model best suited for the primary content of the image: 'faces' for portraits and 'general_scenes' for everything else.",
            default="general_scenes",
            choices=[
                "faces",
                "general_scenes",
            ],
        ),
        restoration_model_type: str = Input(
            description="Select the restoration model that aligns with the content of your image. This model is responsible for image restoration which removes degradations.",
            default="general_scenes",
            choices=[
                "faces",
                "general_scenes",
            ],
        ),
        reload_restoration_model: bool = Input(
            description="Reload the image restoration model (SwinIR) if set to True. This can be useful if you've updated or changed the underlying SwinIR model.",
            default=False,
        ),
        steps: int = Input(
            description="The number of enhancement iterations to perform. More steps might result in a clearer image but can also introduce artifacts.",
            default=50,
            ge=1,
            le=100,
        ),
        super_resolution_factor: int = Input(
            description="Factor by which the input image resolution should be increased. For instance, a factor of 4 will make the resolution 4 times greater in both height and width.",
            default=4,
            ge=1,
            le=4,
        ),
        repeat_times: int = Input(
            description="Number of times the enhancement process is repeated by feeding the output back as input. This can refine the result but might also introduce over-enhancement issues.",
            default=1,
            ge=1,
            le=10,
        ),
        disable_preprocess_model: bool = Input(
            description="Disables the initial preprocessing step using SwinIR. Turn this off if your input image is already of high quality and doesn't require restoration.",
            default=False,
        ),
        tiled: bool = Input(
            description="Whether to use patch-based sampling. This can be useful for very large images to enhance them in smaller chunks rather than all at once.",
            default=False,
        ),
        tile_size: int = Input(
            description="Size of each tile (or patch) when 'tiled' option is enabled. Determines how the image is divided during patch-based enhancement.",
            default=512,
        ),
        tile_stride: int = Input(
            description="Distance between the start of each tile when the image is divided for patch-based enhancement. A smaller stride means more overlap between tiles.",
            default=256,
        ),
        use_guidance: bool = Input(
            description="Use latent image guidance for enhancement. This can help in achieving more accurate and contextually relevant enhancements.",
            default=False,
        ),
        guidance_scale: float = Input(
            description="For 'general_scenes': Scale factor for the guidance mechanism. Adjusts the influence of guidance on the enhancement process.",
            default=0.0,
        ),
        guidance_time_start: int = Input(
            description="For 'general_scenes': Specifies when (at which step) the guidance mechanism starts influencing the enhancement.",
            default=1001,
        ),
        guidance_time_stop: int = Input(
            description="For 'general_scenes': Specifies when (at which step) the guidance mechanism stops influencing the enhancement.",
            default=-1,
        ),
        guidance_space: str = Input(
            description="For 'general_scenes': Determines in which space (RGB or latent) the guidance operates. 'latent' can often provide more subtle and context-aware enhancements.",
            default="latent",
            choices=["rgb", "latent"],
        ),
        guidance_repeat: int = Input(
            description="For 'general_scenes': Number of times the guidance process is repeated during enhancement.",
            default=5,
        ),
        color_fix_type: str = Input(
            description="Method used for color correction post enhancement. 'wavelet' and 'adain' offer different styles of color correction, while 'none' skips this step.",
            default="wavelet",
            choices=["wavelet", "adain", "none"],
        ),
        seed: int = Input(
            description="Random seed to ensure reproducibility. Setting this ensures that multiple runs with the same input produce the same output.",
            default=231,
        ),
        has_aligned: bool = Input(
            description="For 'faces' mode: Indicates if the input images are already cropped and aligned to faces. If not, the model will attempt to do this.",
            default=False,
        ),
        only_center_face: bool = Input(
            description="For 'faces' mode: If multiple faces are detected, only enhance the center-most face in the image.",
            default=False,
        ),
        face_detection_model: str = Input(
            description="For 'faces' mode: Model used for detecting faces in the image. Choose based on accuracy and speed preferences.",
            default="retinaface_resnet50",
            choices=[
                "retinaface_resnet50",
                "retinaface_mobile0.25",
                "YOLOv5l",
                "YOLOv5n",
                "dlib",
            ],
        ),
        background_upsampler: str = Input(
            description="For 'faces' mode: Model used to upscale the background in images where the primary subject is a face.",
            default="RealESRGAN",
            choices=[
                "DiffBIR",
                "RealESRGAN",
            ],
        ),
        background_upsampler_tile: int = Input(
            description="For 'faces' mode: Size of each tile used by the background upsampler when dividing the image into patches.",
            default=400,
        ),
        background_upsampler_tile_stride: int = Input(
            description="For 'faces' mode: Distance between the start of each tile when the background is divided for upscaling. A smaller stride means more overlap between tiles.",
            default=400,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        model_checkpoint_map = {
            "faces": "face_full_v1",
            "general_scenes": "general_full_v1",
        }
        ckpt = f"weights/{model_checkpoint_map[upscaling_model_type]}.ckpt"

        swinir_checkpoint_map = {
            "faces": "face_swinir_v1",
            "general_scenes": "general_swinir_v1",
        }
        swinir_ckpt = f"weights/{swinir_checkpoint_map[restoration_model_type]}.ckpt"
        
        print(f"ckptckptckpt {ckpt}")
        if ckpt == "weights/general_full_v1.ckpt":
            args = Arguments(
                ckpt=ckpt,
                reload_swinir=reload_restoration_model,
                swinir_ckpt=swinir_ckpt,
                input=str(input),
                steps=steps,
                sr_scale=super_resolution_factor,
                repeat_times=repeat_times,
                disable_preprocess_model=disable_preprocess_model,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
                use_guidance=use_guidance,
                g_scale=guidance_scale,
                g_t_start=guidance_time_start,
                g_t_stop=guidance_time_stop,
                g_space=guidance_space,
                g_repeat=guidance_repeat,
                color_fix_type=color_fix_type,
                seed=seed,
            )
            self.switch_model(mode="FULL", args=args)
            yield from self.full_pipeline(args)
        if ckpt == "weights/face_full_v1.ckpt":
            args = Arguments(
                ckpt=ckpt,
                reload_swinir=reload_restoration_model,
                swinir_ckpt=swinir_ckpt,
                input=str(input),
                steps=steps,
                sr_scale=super_resolution_factor,
                repeat_times=repeat_times,
                disable_preprocess_model=disable_preprocess_model,
                has_aligned=has_aligned,
                only_center_face=only_center_face,
                detection_model=face_detection_model,
                bg_upsampler=background_upsampler,
                bg_tile=background_upsampler_tile,
                bg_tile_stride=background_upsampler_tile_stride,
                color_fix_type=color_fix_type,
                seed=seed,
            )
            self.switch_model(mode="FACE", args=args)
            yield from self.face_pipeline(args)
