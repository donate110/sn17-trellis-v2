from __future__ import annotations

import base64
import io
import time
from datetime import datetime
from typing import Optional

from PIL import Image
import pyspz
import torch
import gc

from config import Settings, settings
from logger_config import logger
from schemas import GenerateRequest, GenerateResponse, TrellisParams, TrellisRequest, TrellisResult
from modules.image_edit.qwen_edit_module import QwenEditModule
from modules.background_removal.rmbg_manager import BackgroundRemovalService
from modules.gs_generator.trellis_manager import TrellisService
from modules.utils import secure_randint, set_random_seed, decode_image, to_png_base64, save_files


class GenerationPipeline:
    def __init__(self, settings: Settings = settings):
        self.settings = settings

        # Initialize modules
        self.qwen_edit = QwenEditModule(settings)
        self.rmbg = BackgroundRemovalService(settings)
        self.trellis = TrellisService(settings)

    async def startup(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Starting pipeline")
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

        await self.qwen_edit.startup()
        await self.rmbg.startup()
        await self.trellis.startup()
        
        logger.info("Warming up generator...")
        await self.warmup_generator()
        self._clean_gpu_memory()
        
        logger.success("Warmup is complete. Pipeline ready to work.")

    async def shutdown(self) -> None:
        """Shutdown all pipeline components."""
        logger.info("Closing pipeline")

        # Shutdown all modules
        await self.qwen_edit.shutdown()
        await self.rmbg.shutdown()
        await self.trellis.shutdown()

        logger.info("Pipeline closed.")

    def _clean_gpu_memory(self) -> None:
        """
        Clean the GPU memory.
        """
        gc.collect()
        torch.cuda.empty_cache()

    async def warmup_generator(self) -> None:
        """Function for warming up the generator"""
        
        temp_image = Image.new("RGB",(64,64),color=(128,128,128))
        buffer = io.BytesIO()
        temp_image.save(buffer, format="PNG")
        temp_imge_bytes = buffer.getvalue()
        await self.generate_from_upload(temp_imge_bytes,seed=42)

    async def generate_from_upload(self, image_bytes: bytes, seed: int) -> bytes:
        """
        Generate 3D model from uploaded image file and return PLY as bytes.
        
        Args:
            image_bytes: Raw image bytes from uploaded file
            seed: Random seed for generation
            
        Returns:
            PLY file as bytes
        """
        # Validate input image
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()  # Verify it's a valid image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Reopen after verify
        except Exception as e:
            logger.error(f"Invalid image format: {e}")
            raise ValueError(f"Invalid image format: {e}")
        
        # Check minimum image size
        min_size = 256
        if image.width < min_size or image.height < min_size:
            logger.warning(f"Image size ({image.width}x{image.height}) is below recommended minimum ({min_size}x{min_size})")
        
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # Create request
        request = GenerateRequest(
            prompt_image=image_base64,
            prompt_type="image",
            seed=seed
        )
        
        # Generate
        response = await self.generate_gs(request)
        
        # Return binary PLY - ensure it's bytes
        if not response.ply_file_base64:
            raise ValueError("PLY generation failed")
        
        # Handle both bytes and base64 string cases
        if isinstance(response.ply_file_base64, bytes):
            return response.ply_file_base64
        elif isinstance(response.ply_file_base64, str):
            # If it's a base64 string, decode it
            return base64.b64decode(response.ply_file_base64)
        else:
            raise ValueError(f"Unexpected PLY file type: {type(response.ply_file_base64)}")

    async def generate_gs(self, request: GenerateRequest) -> GenerateResponse:
        """
        Execute full generation pipeline.
        
        Args:
            request: Generation request with prompt and settings
            
        Returns:
            GenerateResponse with generated assets
        """
        t1 = time.time()
        logger.info(f"New generation request")

        # Set seed
        if request.seed < 0:
            request.seed = secure_randint(0, 10000)
            set_random_seed(request.seed)
        else:
            set_random_seed(request.seed)

        # Decode input image
        image = decode_image(request.prompt_image)
        
        # Validate input image quality
        if image.width < 64 or image.height < 64:
            raise ValueError(f"Image too small: {image.width}x{image.height}. Minimum size is 64x64")
        if image.width > 4096 or image.height > 4096:
            logger.warning(f"Image very large: {image.width}x{image.height}. This may cause memory issues.")

        # 1. Edit the image using Qwen Edit
        image_edited = self.qwen_edit.edit_image(prompt_image=image, seed=request.seed)
        
        # Validate edited image
        if not image_edited or image_edited.size[0] == 0 or image_edited.size[1] == 0:
            raise ValueError("Image editing failed: invalid output image")

        # 2. Remove background
        image_without_background = self.rmbg.remove_background(image_edited)
        
        # Validate background-removed image
        if not image_without_background or image_without_background.size[0] == 0 or image_without_background.size[1] == 0:
            logger.warning("Background removal produced invalid image, using edited image instead")
            image_without_background = image_edited

        trellis_result: Optional[TrellisResult] = None
        
        # Resolve Trellis parameters from request
        trellis_params: TrellisParams = request.trellis_params
       
        # 3. Generate the 3D model
        # Ensure image is in RGB format and has valid dimensions
        if image_without_background.mode != "RGB":
            image_without_background = image_without_background.convert("RGB")
        
        # Validate image before 3D generation
        min_3d_size = 256
        if image_without_background.width < min_3d_size or image_without_background.height < min_3d_size:
            logger.warning(f"Image size ({image_without_background.width}x{image_without_background.height}) is below recommended minimum for 3D generation ({min_3d_size}x{min_3d_size})")
        
        trellis_result = self.trellis.generate(
            TrellisRequest(
                image=image_without_background,
                seed=request.seed,
                params=trellis_params
            )
        )
        
        # Validate 3D generation result
        if not trellis_result or not trellis_result.ply_file:
            raise ValueError("3D model generation failed: no PLY file produced")

        # Save generated files
        if self.settings.save_generated_files:
            save_files(trellis_result, image_edited, image_without_background)
        
        # Convert to PNG base64 for response (only if needed)
        image_edited_base64 = None
        image_without_background_base64 = None
        if self.settings.send_generated_files:
            image_edited_base64 = to_png_base64(image_edited)
            image_without_background_base64 = to_png_base64(image_without_background)

        t2 = time.time()
        generation_time = t2 - t1

        # Clean the GPU memory
        self._clean_gpu_memory()

        response = GenerateResponse(
            generation_time=generation_time,
            ply_file_base64=trellis_result.ply_file if trellis_result else None,
            image_edited_file_base64=image_edited_base64 if self.settings.send_generated_files else None,
            image_without_background_file_base64=image_without_background_base64 if self.settings.send_generated_files else None,
        )
        return response

