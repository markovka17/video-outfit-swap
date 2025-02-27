import sys
sys.path.append("IDM-VTON")
sys.path.append("IDM-VTON/gradio_demo")

# Import third-party package IDM-VTON
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel

import apply_net
from utils_mask import get_mask_location
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

from typing import Optional, Sequence, List
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import typer
from loguru import logger

from transformers import AutoTokenizer
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import DDPMScheduler, AutoencoderKL

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import numpy as np

app = typer.Typer()

logger.add(
    sys.stdout,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level}</level> | "
        "<cyan>{file}</cyan>:<cyan>{line}</cyan> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
        "<level>{message}</level>"
    ),
    level="INFO",
)


class Segformer:
    """
    A class for performing semantic segmentation using the Segformer model.
    It supports processing both PIL images and PyTorch tensors as input.
    """

    def __init__(self, device: torch.device, labels: Sequence[int]):
        """
        Initialize the Segformer model.

        Args:
            device (torch.device): The device (e.g., 'cuda' or 'cpu') to run the model on.
            labels (Sequence[int]): A list of label IDs to extract from the segmentation output.
        """
        self.device = device
        self.labels = labels
        assert len(labels) > 0, "At least one label must be provided."

        # Load the pre-trained processor and model
        self.processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self.model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self.model = self.model.to(self.device).eval()  # Move model to the specified device and set to evaluation mode

    @torch.inference_mode()
    def __call__(self, image) -> torch.Tensor:
        """
        Perform semantic segmentation on the input image.

        Args:
            image: Input image, which can be a PIL.Image or a torch.Tensor of shape (H, W, 3) with dtype uint8.

        Returns:
            torch.Tensor: A binary mask tensor of shape (H, W) where True indicates the presence of any of the specified labels.
        """
        # Validate and convert input to PIL image
        if isinstance(image, torch.Tensor):
            assert image.size(2) == 3 and image.dtype == torch.uint8, "Input tensor must be of shape (H, W, 3) and dtype uint8."
            image = Image.fromarray(image.cpu().numpy())  # Convert tensor to PIL image
        elif isinstance(image, Image.Image):
            pass  # Input is already a PIL image
        else:
            raise TypeError("Unsupported input type. Expected PIL.Image or torch.Tensor.")

        # Preprocess the image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].to(self.device)  # Move input to the specified device

        # Perform inference
        outputs = self.model(**inputs)
        logits = outputs.logits.cpu()  # Move logits to CPU for further processing

        # Upsample logits to match the original image size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0]  # Get the predicted segmentation mask

        # Create binary masks for each label and combine them using logical OR
        masks = [pred_seg == label for label in self.labels]
        result = masks[0]
        for mask in masks[1:]:
            result = torch.logical_or(result, mask)

        return result



class Inference:
    """
    A class for performing inference using multiple models, including Segformer, Parsing, OpenPose, and IDM-VTON.
    It handles mask generation, video creation, and smoothing.
    """

    def __init__(self, device: torch.device):
        """
        Initialize the Inference class with all required models and components.

        Args:
            device (torch.device): The device (e.g., 'cuda' or 'cpu') to run the models on.
        """
        self.device = device

        # Initialize Segformer model
        logger.info("Initializing Segformer Model")
        self.segformer = Segformer(device, [14, 15])  # Example labels for segmentation

        # Initialize Parsing and OpenPose models
        logger.info("Initializing Parsing & OpenPose models")
        self.parsing_model = Parsing(self.device.index)
        self.openpose_model = OpenPose(self.device.index)
        self.openpose_model.preprocessor.body_estimation.model.to(device)

        # Base path and data type for IDM-VTON model
        base_path = "yisol/IDM-VTON"
        dtype = torch.float16

        # Initialize IDM-VTON model components
        logger.info("Initializing IDM-VTON model")
        unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet", torch_dtype=dtype)
        unet.requires_grad_(False)

        tokenizer_kwargs = {"revision": None, "use_fast": False}
        tokenizer_one = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer", **tokenizer_kwargs)
        tokenizer_two = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer_2", **tokenizer_kwargs)
        noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

        text_encoder_one = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder", torch_dtype=dtype)
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            base_path, subfolder="text_encoder_2", torch_dtype=dtype
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_path, subfolder="image_encoder", torch_dtype=dtype
        )
        vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae", torch_dtype=torch.float16)

        # Load UNet encoder
        UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(base_path, subfolder="unet_encoder", torch_dtype=dtype)

        # Initialize TryonPipeline
        logger.info("Initializing TryonPipeline pipeline")
        self.pipe = TryonPipeline.from_pretrained(
            base_path,
            unet=unet,
            vae=vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            scheduler=noise_scheduler,
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
        )
        self.pipe.unet_encoder = UNet_Encoder
        self.pipe.to(device)

        # Define tensor transformations
        self.tensor_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    @torch.inference_mode()
    def infer_masks(self, frames, use_smoothing=False, return_mask_gray=False):
        """
        Generate masks for a sequence of frames.

        Args:
            frames (List[torch.Tensor]): List of frames (as tensors) to process.
            use_smoothing (bool): Whether to apply smoothing to the masks.
            return_mask_gray (bool): Whether to return grayscale masks.

        Returns:
            List[PIL.Image]: List of masks.
            List[PIL.Image]: List of grayscale masks (if return_mask_gray is True).
        """
        masks = []
        for frame in tqdm(frames, desc="Processing frames"):
            human_img = Image.fromarray(frame.numpy())

            # Get keypoints and parsing model output
            keypoints = self.openpose_model(human_img.resize((384, 512)))
            model_parse, _ = self.parsing_model(human_img.resize((384, 512)))

            # Generate mask and grayscale mask
            mask, mask_gray = get_mask_location("hd", "upper_body", model_parse, keypoints)
            whole_mask = mask.resize((768, 1024))
            hand_mask = self.segformer(human_img).numpy()
            mask = Image.fromarray(np.logical_xor(np.array(whole_mask), hand_mask))
            masks.append(mask)

        # Apply smoothing if required
        if use_smoothing:
            stacked_masks = self.create_video(masks, return_tensor=True)
            stacked_masks = stacked_masks[..., None].mul(255)
            masks = self.smooth_video_mask(
                stacked_masks, space_kernal_size=5, time_kernal_size=3, device=self.device
            ).permute(0, 3, 1, 2)

            masks = [to_pil_image(mask) for mask in masks]

        if not return_mask_gray:
            return masks

        # Generate grayscale masks
        masks_gray = []
        for mask, frame in zip(masks, frames):
            human_img = Image.fromarray(frame.numpy())
            mask_gray = (1 - transforms.ToTensor()(mask)) * self.tensor_transform(human_img)
            mask_gray = to_pil_image((mask_gray + 1.0) / 2)
            masks_gray.append(mask_gray)

        return masks, masks_gray


    @staticmethod
    def create_video(pil_frames, filename=None, fps=25, return_tensor=False):
        """
        Create a video from a list of PIL frames.

        Args:
            pil_frames (List[PIL.Image]): List of frames as PIL images.
            filename (str): Output filename for the video.
            fps (int): Frames per second for the video.
            return_tensor (bool): Whether to return the frames as a tensor.

        Returns:
            torch.Tensor: Stacked frames as a tensor (if return_tensor is True).
        """
        frames = [torch.from_numpy(np.array(pil_frame)) for pil_frame in pil_frames]
        frames = torch.stack(frames).to(torch.uint8)

        if return_tensor:
            return frames
        assert filename is not None, "Filename must be provided if return_tensor is False."
        torchvision.io.write_video(filename, frames, fps=fps)

    @staticmethod
    def smooth_video_mask(video: torch.Tensor, space_kernal_size: int = 15, time_kernal_size: int = 7, device: str = "cuda"):
        """
        Smooth a video mask using spatial and temporal averaging.

        Args:
            video (torch.Tensor): Input video tensor of shape (T, H, W, C).
            space_kernal_size (int): Kernel size for spatial smoothing.
            time_kernal_size (int): Kernel size for temporal smoothing.
            device (str): Device to perform the smoothing on.

        Returns:
            torch.Tensor: Smoothed video tensor of shape (T, H, W, C).
        """
        # (T, H, W, C) -> (C, T, H, W)
        video = video.permute(3, 0, 1, 2)

        space_padding = space_kernal_size // 2
        time_padding = time_kernal_size // 2
        video = video.to(device)
        video = video[0].float() / 255  # Normalize to [0, 1]

        # Apply spatial smoothing
        video = torch.nn.functional.avg_pool2d(
            video,
            kernel_size=(space_kernal_size, space_kernal_size),
            stride=(1, 1),
            padding=(space_padding, space_padding),
        )

        # Apply temporal smoothing
        video = video.permute(1, 2, 0)  # (H, W, T)
        video = (
            torch.nn.functional.avg_pool1d(video, kernel_size=time_kernal_size, stride=1, padding=time_padding)
            .unsqueeze(0)
            .repeat(3, 1, 1, 1)
        )  # (C, H, W, T)

        # Binarize the mask
        video[video > 0.5] = 1
        video[video <= 0.5] = 0
        video = (video * 255).byte()

        # (C, H, W, T) -> (T, H, W, C)
        video = video.to("cpu").permute(3, 1, 2, 0)
        return video

    @torch.inference_mode()
    def vton_on_video(
        self,
        frames: List[torch.Tensor],
        masks: List[Image.Image],
        garment: Image.Image,
        denoise_steps: int = 15,
        repaint: bool = True,
    ) -> List[Image.Image]:
        """
        Perform virtual try-on on a sequence of video frames.

        Args:
            frames (List[torch.Tensor]): List of frames (as tensors) to process.
            masks (List[Image.Image]): List of masks corresponding to the frames.
            garment (Image.Image): The garment image to overlay on the frames.
            denoise_steps (int): Number of denoising steps for the diffusion process.
            repaint (bool): Whether to repaint the result using the original frame and mask.

        Returns:
            List[Image.Image]: List of try-on results as PIL images.
        """
        try_on_results = []

        for frame, mask in zip(frames, masks):
            # Convert frame to PIL image
            human_img = Image.fromarray(frame.numpy())

            # Preprocess human image for pose estimation
            human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
            human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

            # Generate pose image using DensePose
            args = apply_net.create_argument_parser().parse_args(
                (
                    "show",
                    "IDM-VTON/configs/densepose_rcnn_R_50_FPN_s1x.yaml",
                    "IDM-VTON/ckpt/densepose/model_final_162be9.pkl",
                    "dp_segm",
                    "-v",
                    "--opts",
                    "MODEL.DEVICE",
                    "cuda",
                )
            )
            pose_img = args.func(args, human_img_arg)
            pose_img = pose_img[:, :, ::-1]  # Convert BGR to RGB
            pose_img = Image.fromarray(pose_img).resize((768, 1024))

            # Perform virtual try-on using the TryonPipeline
            with torch.amp.autocast("cuda"):
                # Define prompts for the garment
                garment_des = "t-shirt"
                prompt = f"model is wearing {garment_des}"
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

                # Encode prompts
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
                    self.pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                )

                # Encode garment-specific prompt
                garment_prompt = f"a photo of {garment_des}"
                prompt_embeds_c, _, _, _ = self.pipe.encode_prompt(
                    garment_prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=negative_prompt,
                )

                # Generate try-on result
                images = self.pipe(
                    prompt_embeds=prompt_embeds.to(self.device, torch.float16),
                    negative_prompt_embeds=negative_prompt_embeds.to(self.device, torch.float16),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(self.device, torch.float16),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(self.device, torch.float16),
                    num_inference_steps=denoise_steps,
                    generator=torch.Generator(self.device).manual_seed(123),
                    strength=1,
                    pose_img=self.tensor_transform(pose_img).unsqueeze(0).to(self.device, torch.float16),
                    text_embeds_cloth=prompt_embeds_c.to(self.device, torch.float16),
                    cloth=self.tensor_transform(garment).unsqueeze(0).to(self.device, torch.float16),
                    mask_image=mask,
                    image=human_img,
                    height=1024,
                    width=768,
                    ip_adapter_image=garment.resize((768, 1024)),
                    guidance_scale=2.5,
                )[0]

            # Extract the result
            res = images[0]

            # Repaint the result using the original frame and mask
            if repaint:
                mask_tensor = torch.from_numpy(np.array(mask)).bool()
                res_tensor = torch.from_numpy(np.array(res))
                res = frame * ~mask_tensor + res_tensor * mask_tensor

            try_on_results.append(res)

        return try_on_results


@app.command()
def video_tryon(
    video_path: Path = typer.Option(..., help="Path to the input video file"),
    garment_path: Path = typer.Option(..., help="Path to the garment image file"),
    output_path: Path = typer.Option(..., help="Path to the output video file"),
    device: str = typer.Option("cuda:0", help="CUDA device, lile 'cuda' or 'cuda:2'"),
    end_frame: int = typer.Option(None, help="End frame of processing of video by `video_path`"),
    use_smoothing: bool = typer.Option(True, help="Use temporal and spatial mask smothing"),
    repaint: bool = typer.Option(True, help="Use repaint to original video"),
):
    """
    Process images and video frames using the provided paths.
    """
    device = torch.device(device)
    # Load video frames
    hg_frames, _, video_info = torchvision.io.read_video(str(video_path))
    logger.info(f"{hg_frames.shape=}")

    # Load and resize the garment image
    garm_img = Image.open(garment_path).convert("RGB").resize((768, 1024))

    inference = Inference(device)
    end_frame = end_frame or hg_frames.size(0)

    masks = inference.infer_masks(hg_frames[:end_frame], use_smoothing=use_smoothing, return_mask_gray=False)
    vton_result = inference.vton_on_video(hg_frames[:end_frame], masks[:end_frame], garm_img, repaint=repaint)

    torchvision.io.write_video(output_path.as_posix(), torch.stack(vton_result), video_info["video_fps"])


if __name__ == "__main__":
    app()
