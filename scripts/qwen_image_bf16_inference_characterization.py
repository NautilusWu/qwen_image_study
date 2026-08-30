"""
Qwen-Image BF16 Inference Characterization
==========================================

Purpose
-------
This script performs a reproducible characterization of the BF16
text-to-image inference pipeline of:

    Qwen/Qwen-Image

The experiment is designed to answer four primary questions:

1. What is the end-to-end inference latency of Qwen-Image under a fixed
   BF16 text-to-image configuration?

2. What are the input/output tensor shapes at the major stages of the
   prompt-to-image pipeline?

3. How is inference latency distributed across the major pipeline stages?

4. Which major pipeline component dominates the total inference time and
   therefore represents the primary pipeline-level optimization target?


Pipeline Under Test
-------------------
The text-to-image execution path examined by this script is:

    Prompt
      -> Tokenizer
      -> Text Encoder
      -> Random Gaussian Latent Initialization
      -> Latent Packing
      -> QwenImageTransformer2DModel
      -> FlowMatchEulerDiscreteScheduler
         [Transformer + Scheduler repeated for N inference steps]
      -> Latent Unpacking
      -> VAE Decode
      -> Image Postprocessing
      -> Final RGB Image

Important:
For pure text-to-image generation, Qwen-Image does NOT perform VAE Encode.

There is no input image that needs to be encoded. Instead, generation
starts directly from random Gaussian latent noise. VAE Encode is relevant
to image-conditioned workflows such as image-to-image generation or image
editing.


Experimental Methodology
------------------------
The model is loaded exactly once inside one Python process.

The experiment then performs the following stages:

1. Environment Characterization
   Records:
       - operating system
       - Python version
       - PyTorch version
       - CUDA runtime
       - cuDNN version
       - GPU model
       - CUDA compute capability
       - CUDA-visible memory
       - BF16 support
       - Diffusers / Transformers / Accelerate versions

2. Pipeline Initialization
   Loads Qwen/Qwen-Image in BF16 and transfers the pipeline to CUDA.

   Model loading time is measured separately and is NOT included in the
   reported clean prompt-to-image inference latency.

3. GPU Warm-up
   Executes a short generation before formal timing.

   The purpose is to reduce first-run effects associated with:
       - CUDA context initialization
       - kernel initialization
       - memory allocator initialization
       - runtime caching

4. Clean End-to-End Baseline Benchmark
   Executes the same text-to-image inference multiple times using identical:
       - prompt
       - random seed
       - resolution
       - number of inference steps
       - precision
       - scheduler
       - CFG configuration

   No profiling hooks are active during these runs.

   The script reports:
       - individual run latency
       - mean latency
       - standard deviation
       - PyTorch peak allocated CUDA memory
       - PyTorch peak reserved CUDA memory

5. Instrumented Pipeline Run
   Executes one additional generation with runtime instrumentation.

   This run records:

       Tensor shapes:
           - tokenizer output
           - prompt embeddings
           - latent before packing
           - latent after packing
           - Transformer input
           - Transformer output
           - packed final latent
           - unpacked VAE latent
           - VAE decoder input/output
           - final image size

       Pipeline latency:
           - prompt encoding
           - latent preparation
           - Transformer forward passes
           - scheduler updates
           - latent unpacking
           - VAE decoding
           - image postprocessing

       Prompt sub-profile:
           - tokenizer
           - text encoder
           - total prompt encoding

6. Result Serialization
   Saves experimental configuration, raw measurements, tensor shapes,
   profiling results, and summary statistics.


Why Clean Benchmarking and Profiling Are Separate
-------------------------------------------------
CUDA execution is asynchronous.

Accurate timing of individual GPU stages requires explicit:

    torch.cuda.synchronize()

However, frequent synchronization changes the normal asynchronous execution
behavior of the pipeline and can slightly increase total runtime.

Therefore the experiment intentionally separates:

    Clean benchmark runs
        -> representative end-to-end inference latency

    Instrumented profiling run
        -> stage-level latency distribution and tensor inspection

The instrumented total runtime should therefore not be interpreted as an
identical replacement for clean benchmark latency.


Default Experimental Configuration
----------------------------------
Model:
    Qwen/Qwen-Image

Precision:
    BF16

Resolution:
    1024 x 1024

Batch size:
    1

Images per prompt:
    1

Inference steps:
    50

Random seed:
    42

Classifier-Free Guidance:
    Disabled

true_cfg_scale:
    1.0

Negative prompt:
    None

Scheduler:
    The scheduler provided by the loaded Qwen-Image pipeline
    (expected: FlowMatchEulerDiscreteScheduler)


Scope
-----
This experiment performs PIPELINE-LEVEL characterization.

It can identify whether the major bottleneck is:

    - prompt encoding
    - latent preparation
    - Transformer denoising
    - scheduler updates
    - VAE decoding
    - image postprocessing

It does NOT determine which internal Transformer operation is responsible
for Transformer latency.

For example, this script does not separately measure:

    - individual Transformer blocks
    - Attention
    - Q/K/V projections
    - MLP layers
    - GEMM kernels
    - Tensor Core utilization
    - memory bandwidth
    - kernel launch overhead

Those require lower-level tools such as:

    - PyTorch Profiler
    - NVIDIA Nsight Systems
    - NVIDIA Nsight Compute


Output Files
------------
By default, results are written to:

    /outputs/qwen_image_bf16_inference_characterization/

Files:

    environment.json
        Hardware and software environment.

    experiment_metadata.json
        Experimental configuration, model components, scheduler configuration,
        and model loading time.

    baseline_latency.csv
        Raw latency and PyTorch CUDA memory measurements for each clean run.

    baseline.png
        Representative image from the first clean baseline run.

    instrumented.png
        Image produced by the instrumented profiling run.

    tensor_shapes.json
        Runtime tensor shapes observed at major pipeline stages.

    pipeline_latency_profile.csv
        Pipeline-level stage timing and percentages.

    prompt_encoding_profile.json
        Tokenizer and text-encoder timing details.

    experiment_summary.json
        Main benchmark and profiling results.


Reproducibility
---------------
Unless explicitly overridden through command-line arguments, all measured
generation runs use the same experimental configuration.

The random generator is recreated with the same seed before every run so
that every generation starts from the same initial random latent.

Example
-------
Run with default settings:

    python scripts/qwen_image_bf16_inference_characterization.py

Run five clean baseline measurements:

    python scripts/qwen_image_bf16_inference_characterization.py \
        --baseline-runs 5

Change inference steps:

    python scripts/qwen_image_bf16_inference_characterization.py \
        --steps 30
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from functools import wraps
from typing import Any

import accelerate
import diffusers
import huggingface_hub
import torch
import transformers
from diffusers import QwenImagePipeline


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================


@dataclass
class ExperimentConfig:
    model_id: str = "Qwen/Qwen-Image"

    prompt: str = (
        "A red sports car parked on a quiet city street at sunset, "
        "with realistic lighting, detailed buildings, and reflections "
        "on the wet pavement."
    )

    width: int = 1024
    height: int = 1024

    num_inference_steps: int = 50
    seed: int = 42

    true_cfg_scale: float = 1.0
    negative_prompt: str | None = None

    baseline_runs: int = 3
    warmup_steps: int = 5

    output_dir: str = (
        "/outputs/qwen_image_bf16_inference_characterization"
    )


# ============================================================
# GENERAL UTILITIES
# ============================================================


def cuda_sync() -> None:
    """Wait until all previously submitted CUDA work has completed."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tensor_info(value: Any) -> dict[str, Any] | None:
    """Return a JSON-serializable description of a tensor."""
    if not isinstance(value, torch.Tensor):
        return None

    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def find_first_tensor(obj: Any) -> torch.Tensor | None:
    """Recursively search an object for the first PyTorch tensor."""

    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, dict):
        for value in obj.values():
            tensor = find_first_tensor(value)
            if tensor is not None:
                return tensor

    if isinstance(obj, (list, tuple)):
        for value in obj:
            tensor = find_first_tensor(value)
            if tensor is not None:
                return tensor

    if hasattr(obj, "sample"):
        sample = getattr(obj, "sample")

        if isinstance(sample, torch.Tensor):
            return sample

    return None


def json_safe(obj: Any) -> Any:
    """Convert objects into JSON-compatible representations."""

    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {
            str(key): json_safe(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [json_safe(value) for value in obj]

    return str(obj)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            json_safe(data),
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# ENVIRONMENT CHARACTERIZATION
# ============================================================


def get_nvidia_smi_info() -> str:
    """Return GPU name and NVIDIA driver version using nvidia-smi."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except Exception as error:
        return f"Unavailable: {error}"


def collect_environment() -> dict[str, Any]:
    """Collect software and CUDA-visible hardware information."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "This experiment requires a CUDA-capable NVIDIA GPU."
        )

    device_index = torch.cuda.current_device()

    properties = torch.cuda.get_device_properties(
        device_index
    )

    return {
        "system": {
            "hostname": platform.node(),
            "operating_system": platform.platform(),
            "architecture": platform.machine(),
        },

        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },

        "pytorch_cuda": {
            "pytorch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
            "tf32_matmul_allowed":
                torch.backends.cuda.matmul.allow_tf32,
        },

        "gpu": {
            "device_index": device_index,
            "name": torch.cuda.get_device_name(
                device_index
            ),
            "compute_capability": list(
                torch.cuda.get_device_capability(
                    device_index
                )
            ),
            "cuda_visible_memory_gib":
                properties.total_memory / (1024 ** 3),
            "bf16_supported":
                torch.cuda.is_bf16_supported(),
            "nvidia_smi":
                get_nvidia_smi_info(),
        },

        "python_libraries": {
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "huggingface_hub":
                huggingface_hub.__version__,
        },
    }


# ============================================================
# QWEN-IMAGE PIPELINE INITIALIZATION
# ============================================================


def load_pipeline(
    config: ExperimentConfig,
) -> tuple[QwenImagePipeline, float]:
    """
    Load Qwen-Image once in BF16 and move it to CUDA.

    Loading time is measured separately from inference latency.
    """

    print("\n" + "=" * 80)
    print("QWEN-IMAGE PIPELINE INITIALIZATION")
    print("=" * 80)

    start = time.perf_counter()

    pipe = QwenImagePipeline.from_pretrained(
        config.model_id,
        dtype=torch.bfloat16,
    )

    pipe.to("cuda")

    pipe.set_progress_bar_config(
        disable=True
    )

    cuda_sync()

    load_time = time.perf_counter() - start

    print(f"Model:      {config.model_id}")
    print(f"Precision:  BF16")
    print(f"GPU:        {torch.cuda.get_device_name(0)}")
    print(f"Load time:  {load_time:.4f} s")

    return pipe, load_time


# ============================================================
# INFERENCE GENERATION
# ============================================================


def generate(
    pipe: QwenImagePipeline,
    config: ExperimentConfig,
    steps: int | None = None,
):
    """
    Execute one deterministic Qwen-Image generation.

    The CUDA random generator is recreated for every run using the same
    seed so each run starts from the same initial latent noise.
    """

    if steps is None:
        steps = config.num_inference_steps

    generator = torch.Generator(
        device="cuda"
    ).manual_seed(
        config.seed
    )

    with torch.inference_mode():

        result = pipe(
            prompt=config.prompt,
            width=config.width,
            height=config.height,
            num_inference_steps=steps,
            true_cfg_scale=config.true_cfg_scale,
            negative_prompt=config.negative_prompt,
            generator=generator,
        )

    return result


# ============================================================
# GPU WARM-UP
# ============================================================


def run_warmup(
    pipe: QwenImagePipeline,
    config: ExperimentConfig,
) -> float:
    """Run a short generation before formal benchmark measurements."""

    print("\n" + "=" * 80)
    print("GPU WARM-UP")
    print("=" * 80)

    cuda_sync()

    start = time.perf_counter()

    _ = generate(
        pipe,
        config,
        steps=config.warmup_steps,
    )

    cuda_sync()

    elapsed = time.perf_counter() - start

    print(
        f"Warm-up inference: "
        f"{config.warmup_steps} steps"
    )

    print(
        f"Warm-up time:      "
        f"{elapsed:.4f} s"
    )

    return elapsed


# ============================================================
# CLEAN END-TO-END LATENCY BENCHMARK
# ============================================================


def run_clean_baseline(
    pipe: QwenImagePipeline,
    config: ExperimentConfig,
):
    """
    Measure normal prompt-to-image latency without profiling hooks.

    Image saving is performed after timing and is therefore not included
    in inference latency.
    """

    print("\n" + "=" * 80)
    print("CLEAN END-TO-END LATENCY BENCHMARK")
    print("=" * 80)

    rows = []

    baseline_image_path = os.path.join(
        config.output_dir,
        "baseline.png",
    )

    for run_index in range(
        1,
        config.baseline_runs + 1,
    ):

        torch.cuda.reset_peak_memory_stats()

        cuda_sync()

        start = time.perf_counter()

        result = generate(
            pipe,
            config,
        )

        cuda_sync()

        latency = (
            time.perf_counter()
            - start
        )

        peak_allocated_gib = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        peak_reserved_gib = (
            torch.cuda.max_memory_reserved()
            / (1024 ** 3)
        )

        row = {
            "run": run_index,
            "latency_s": latency,

            # These are PyTorch CUDA allocator statistics,
            # not total physical/unified system memory usage.
            "pytorch_peak_allocated_gib":
                peak_allocated_gib,

            "pytorch_peak_reserved_gib":
                peak_reserved_gib,
        }

        rows.append(row)

        print(
            f"Run {run_index:02d}: "
            f"{latency:.4f} s"
        )

        if run_index == 1:
            result.images[0].save(
                baseline_image_path
            )

    latencies = [
        row["latency_s"]
        for row in rows
    ]

    mean_latency = statistics.mean(
        latencies
    )

    std_latency = (
        statistics.stdev(latencies)
        if len(latencies) > 1
        else 0.0
    )

    print("\nClean baseline statistics:")
    print(
        f"Mean latency: "
        f"{mean_latency:.4f} s"
    )
    print(
        f"Std latency:  "
        f"{std_latency:.4f} s"
    )

    return (
        rows,
        mean_latency,
        std_latency,
    )


# ============================================================
# PIPELINE INSTRUMENTATION
# ============================================================


class PipelineProfiler:
    """
    Runtime instrumentation for Qwen-Image.

    This profiler records:
        - major pipeline stage latency
        - call counts
        - major tensor shapes

    It temporarily wraps pipeline methods and restores the original
    methods after the instrumented generation completes.
    """

    def __init__(
        self,
        pipe: QwenImagePipeline,
    ):
        self.pipe = pipe

        self.times: dict[str, list[float]] = {}
        self.counts: dict[str, int] = {}
        self.shapes: dict[str, Any] = {}

        self.originals: dict[str, Any] = {}


    def add_time(
        self,
        name: str,
        elapsed: float,
    ) -> None:

        self.times.setdefault(
            name,
            [],
        ).append(
            elapsed
        )

        self.counts[name] = (
            self.counts.get(name, 0)
            + 1
        )


    def total_time(
        self,
        name: str,
    ) -> float:

        return sum(
            self.times.get(
                name,
                [],
            )
        )


    def timed_wrapper(
        self,
        name: str,
        function,
    ):
        """Wrap a CUDA stage with synchronized timing."""

        @wraps(function)
        def wrapped(*args, **kwargs):

            cuda_sync()

            start = time.perf_counter()

            output = function(
                *args,
                **kwargs,
            )

            cuda_sync()

            elapsed = (
                time.perf_counter()
                - start
            )

            self.add_time(
                name,
                elapsed,
            )

            return output

        return wrapped


    # ========================================================
    # INSTALL INSTRUMENTATION
    # ========================================================

    def install(self) -> None:

        pipe = self.pipe
        profiler = self

        # ----------------------------------------------------
        # TOKENIZER
        # ----------------------------------------------------
        #
        # __call__ is a Python special method. Python resolves
        # special methods on the class rather than only on the
        # instance, so the tokenizer class is patched temporarily.
        #

        tokenizer = pipe.tokenizer
        tokenizer_class = type(tokenizer)

        original_tokenizer_call = (
            tokenizer_class.__call__
        )

        self.originals[
            "tokenizer_class"
        ] = tokenizer_class

        self.originals[
            "tokenizer_call"
        ] = original_tokenizer_call


        def tokenizer_call_wrapper(
            tokenizer_self,
            *args,
            **kwargs,
        ):

            # Do not interfere with unrelated tokenizer instances.
            if tokenizer_self is not tokenizer:

                return original_tokenizer_call(
                    tokenizer_self,
                    *args,
                    **kwargs,
                )

            start = time.perf_counter()

            output = original_tokenizer_call(
                tokenizer_self,
                *args,
                **kwargs,
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            profiler.add_time(
                "tokenizer",
                elapsed,
            )

            if (
                "tokenizer_output"
                not in profiler.shapes
            ):

                shapes = {}

                if hasattr(
                    output,
                    "input_ids",
                ):
                    shapes[
                        "input_ids"
                    ] = tensor_info(
                        output.input_ids
                    )

                if hasattr(
                    output,
                    "attention_mask",
                ):
                    shapes[
                        "attention_mask"
                    ] = tensor_info(
                        output.attention_mask
                    )

                profiler.shapes[
                    "tokenizer_output"
                ] = shapes

            return output


        tokenizer_class.__call__ = (
            tokenizer_call_wrapper
        )

        # ----------------------------------------------------
        # TEXT ENCODER
        # ----------------------------------------------------

        original_text_encoder_forward = (
            pipe.text_encoder.forward
        )

        self.originals[
            "text_encoder_forward"
        ] = original_text_encoder_forward


        def text_encoder_forward_wrapper(
            *args,
            **kwargs,
        ):

            if (
                "text_encoder_input"
                not in self.shapes
            ):

                input_ids = kwargs.get(
                    "input_ids"
                )

                if (
                    input_ids is None
                    and len(args) > 0
                    and isinstance(
                        args[0],
                        torch.Tensor,
                    )
                ):
                    input_ids = args[0]

                self.shapes[
                    "text_encoder_input"
                ] = {
                    "input_ids":
                        tensor_info(
                            input_ids
                        )
                }

            cuda_sync()

            start = time.perf_counter()

            output = (
                original_text_encoder_forward(
                    *args,
                    **kwargs,
                )
            )

            cuda_sync()

            self.add_time(
                "text_encoder",
                time.perf_counter()
                - start,
            )

            return output


        pipe.text_encoder.forward = (
            text_encoder_forward_wrapper
        )

        # ----------------------------------------------------
        # COMPLETE PROMPT ENCODING
        # ----------------------------------------------------

        original_encode_prompt = (
            pipe.encode_prompt
        )

        self.originals[
            "encode_prompt"
        ] = original_encode_prompt


        def encode_prompt_wrapper(
            *args,
            **kwargs,
        ):

            cuda_sync()

            start = time.perf_counter()

            output = original_encode_prompt(
                *args,
                **kwargs,
            )

            cuda_sync()

            elapsed = (
                time.perf_counter()
                - start
            )

            self.add_time(
                "prompt_encoding",
                elapsed,
            )

            if (
                "prompt_embeddings"
                not in self.shapes
            ):

                tensor = find_first_tensor(
                    output
                )

                self.shapes[
                    "prompt_embeddings"
                ] = tensor_info(
                    tensor
                )

            return output


        pipe.encode_prompt = (
            encode_prompt_wrapper
        )

        # ----------------------------------------------------
        # LATENT PREPARATION
        # ----------------------------------------------------

        original_prepare_latents = (
            pipe.prepare_latents
        )

        self.originals[
            "prepare_latents"
        ] = original_prepare_latents

        pipe.prepare_latents = (
            self.timed_wrapper(
                "latent_preparation",
                original_prepare_latents,
            )
        )

        # ----------------------------------------------------
        # LATENT PACKING
        # ----------------------------------------------------
        #
        # Packing is inspected for shapes.
        # Its runtime is already included inside latent preparation
        # and is therefore NOT separately added to the top-level
        # latency breakdown to avoid double-counting.
        #

        original_pack_latents = (
            pipe._pack_latents
        )

        self.originals[
            "_pack_latents"
        ] = original_pack_latents


        def pack_latents_wrapper(
            *args,
            **kwargs,
        ):

            if (
                "latent_before_packing"
                not in self.shapes
            ):

                tensor = find_first_tensor(
                    args
                )

                self.shapes[
                    "latent_before_packing"
                ] = tensor_info(
                    tensor
                )

            output = original_pack_latents(
                *args,
                **kwargs,
            )

            if (
                "latent_after_packing"
                not in self.shapes
            ):

                tensor = find_first_tensor(
                    output
                )

                self.shapes[
                    "latent_after_packing"
                ] = tensor_info(
                    tensor
                )

            return output


        pipe._pack_latents = (
            pack_latents_wrapper
        )

        # ----------------------------------------------------
        # TRANSFORMER
        # ----------------------------------------------------

        original_transformer_forward = (
            pipe.transformer.forward
        )

        self.originals[
            "transformer_forward"
        ] = original_transformer_forward


        def transformer_forward_wrapper(
            *args,
            **kwargs,
        ):

            if (
                "transformer_input"
                not in self.shapes
            ):

                self.shapes[
                    "transformer_input"
                ] = {
                    "hidden_states":
                        tensor_info(
                            kwargs.get(
                                "hidden_states"
                            )
                        ),

                    "encoder_hidden_states":
                        tensor_info(
                            kwargs.get(
                                "encoder_hidden_states"
                            )
                        ),

                    "timestep":
                        tensor_info(
                            kwargs.get(
                                "timestep"
                            )
                        ),
                }

            cuda_sync()

            start = time.perf_counter()

            output = (
                original_transformer_forward(
                    *args,
                    **kwargs,
                )
            )

            cuda_sync()

            elapsed = (
                time.perf_counter()
                - start
            )

            self.add_time(
                "transformer",
                elapsed,
            )

            if (
                "transformer_output"
                not in self.shapes
            ):

                tensor = find_first_tensor(
                    output
                )

                self.shapes[
                    "transformer_output"
                ] = tensor_info(
                    tensor
                )

            return output


        pipe.transformer.forward = (
            transformer_forward_wrapper
        )

        # ----------------------------------------------------
        # SCHEDULER
        # ----------------------------------------------------

        original_scheduler_step = (
            pipe.scheduler.step
        )

        self.originals[
            "scheduler_step"
        ] = original_scheduler_step

        pipe.scheduler.step = (
            self.timed_wrapper(
                "scheduler",
                original_scheduler_step,
            )
        )

        # ----------------------------------------------------
        # LATENT UNPACKING
        # ----------------------------------------------------

        original_unpack_latents = (
            pipe._unpack_latents
        )

        self.originals[
            "_unpack_latents"
        ] = original_unpack_latents


        def unpack_latents_wrapper(
            *args,
            **kwargs,
        ):

            if (
                "packed_final_latent"
                not in self.shapes
            ):

                tensor = find_first_tensor(
                    args
                )

                self.shapes[
                    "packed_final_latent"
                ] = tensor_info(
                    tensor
                )

            cuda_sync()

            start = time.perf_counter()

            output = (
                original_unpack_latents(
                    *args,
                    **kwargs,
                )
            )

            cuda_sync()

            elapsed = (
                time.perf_counter()
                - start
            )

            self.add_time(
                "latent_unpacking",
                elapsed,
            )

            if (
                "unpacked_vae_latent"
                not in self.shapes
            ):

                tensor = find_first_tensor(
                    output
                )

                self.shapes[
                    "unpacked_vae_latent"
                ] = tensor_info(
                    tensor
                )

            return output


        pipe._unpack_latents = (
            unpack_latents_wrapper
        )

        # ----------------------------------------------------
        # VAE DECODE
        # ----------------------------------------------------

        original_vae_decode = (
            pipe.vae.decode
        )

        self.originals[
            "vae_decode"
        ] = original_vae_decode


        def vae_decode_wrapper(
            *args,
            **kwargs,
        ):

            if (
                "vae_decode_input"
                not in self.shapes
            ):

                tensor = find_first_tensor(
                    args
                )

                self.shapes[
                    "vae_decode_input"
                ] = tensor_info(
                    tensor
                )

            cuda_sync()

            start = time.perf_counter()

            output = original_vae_decode(
                *args,
                **kwargs,
            )

            cuda_sync()

            elapsed = (
                time.perf_counter()
                - start
            )

            self.add_time(
                "vae_decode",
                elapsed,
            )

            if (
                "vae_decode_output"
                not in self.shapes
            ):

                tensor = find_first_tensor(
                    output
                )

                self.shapes[
                    "vae_decode_output"
                ] = tensor_info(
                    tensor
                )

            return output


        pipe.vae.decode = (
            vae_decode_wrapper
        )

        # ----------------------------------------------------
        # IMAGE POSTPROCESSING
        # ----------------------------------------------------

        original_postprocess = (
            pipe.image_processor.postprocess
        )

        self.originals[
            "image_postprocessing"
        ] = original_postprocess

        pipe.image_processor.postprocess = (
            self.timed_wrapper(
                "image_postprocessing",
                original_postprocess,
            )
        )


    # ========================================================
    # RESTORE ORIGINAL PIPELINE
    # ========================================================

    def restore(self) -> None:
        """Remove instrumentation and restore the original pipeline."""

        pipe = self.pipe

        self.originals[
            "tokenizer_class"
        ].__call__ = self.originals[
            "tokenizer_call"
        ]

        pipe.text_encoder.forward = (
            self.originals[
                "text_encoder_forward"
            ]
        )

        pipe.encode_prompt = (
            self.originals[
                "encode_prompt"
            ]
        )

        pipe.prepare_latents = (
            self.originals[
                "prepare_latents"
            ]
        )

        pipe._pack_latents = (
            self.originals[
                "_pack_latents"
            ]
        )

        pipe.transformer.forward = (
            self.originals[
                "transformer_forward"
            ]
        )

        pipe.scheduler.step = (
            self.originals[
                "scheduler_step"
            ]
        )

        pipe._unpack_latents = (
            self.originals[
                "_unpack_latents"
            ]
        )

        pipe.vae.decode = (
            self.originals[
                "vae_decode"
            ]
        )

        pipe.image_processor.postprocess = (
            self.originals[
                "image_postprocessing"
            ]
        )


# ============================================================
# INSTRUMENTED SHAPE AND LATENCY PROFILING
# ============================================================


def run_instrumented_profile(
    pipe: QwenImagePipeline,
    config: ExperimentConfig,
):
    """
    Run one generation with tensor-shape inspection and stage timing.
    """

    print("\n" + "=" * 80)
    print("INSTRUMENTED PIPELINE PROFILING")
    print("=" * 80)

    profiler = PipelineProfiler(
        pipe
    )

    profiler.install()

    try:

        cuda_sync()

        start = time.perf_counter()

        result = generate(
            pipe,
            config,
        )

        cuda_sync()

        total_latency = (
            time.perf_counter()
            - start
        )

    finally:

        profiler.restore()

    final_image = result.images[0]

    profiler.shapes[
        "final_image"
    ] = {
        "type":
            type(final_image).__name__,
        "size":
            list(final_image.size),
    }

    instrumented_image_path = os.path.join(
        config.output_dir,
        "instrumented.png",
    )

    final_image.save(
        instrumented_image_path
    )

    return (
        profiler,
        total_latency,
    )


# ============================================================
# RESULT SERIALIZATION
# ============================================================


def write_baseline_latency_csv(
    path: str,
    rows: list[dict[str, Any]],
) -> None:

    fieldnames = [
        "run",
        "latency_s",
        "pytorch_peak_allocated_gib",
        "pytorch_peak_reserved_gib",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def build_pipeline_profile(
    profiler: PipelineProfiler,
    total_latency: float,
) -> list[dict[str, Any]]:
    """
    Construct the non-overlapping top-level pipeline latency breakdown.

    Tokenizer and text encoder are NOT listed separately here because their
    timing is already included in total prompt encoding time.

    Their detailed timing is saved separately in
    prompt_encoding_profile.json.
    """

    stages = [
        (
            "Prompt Encoding",
            "prompt_encoding",
        ),
        (
            "Latent Preparation",
            "latent_preparation",
        ),
        (
            "Transformer Forward",
            "transformer",
        ),
        (
            "Scheduler Update",
            "scheduler",
        ),
        (
            "Latent Unpacking",
            "latent_unpacking",
        ),
        (
            "VAE Decode",
            "vae_decode",
        ),
        (
            "Image Postprocessing",
            "image_postprocessing",
        ),
    ]

    rows = []

    measured_total = 0.0

    for display_name, internal_name in stages:

        elapsed = profiler.total_time(
            internal_name
        )

        calls = profiler.counts.get(
            internal_name,
            0,
        )

        measured_total += elapsed

        rows.append({
            "stage": display_name,
            "calls": calls,
            "time_s": elapsed,
            "percentage":
                100.0
                * elapsed
                / total_latency,
        })

    other_latency = max(
        0.0,
        total_latency
        - measured_total,
    )

    rows.append({
        "stage":
            "Other Pipeline Overhead",
        "calls": "",
        "time_s":
            other_latency,
        "percentage":
            100.0
            * other_latency
            / total_latency,
    })

    return rows


def write_pipeline_profile_csv(
    path: str,
    rows: list[dict[str, Any]],
) -> None:

    fieldnames = [
        "stage",
        "calls",
        "time_s",
        "percentage",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def print_pipeline_profile(
    rows: list[dict[str, Any]],
    total_latency: float,
    profiler: PipelineProfiler,
) -> None:

    print("\nPipeline latency profile")
    print("-" * 90)

    for row in rows:

        calls = row["calls"]

        print(
            f"{row['stage']:30s}"
            f"{row['time_s']:12.4f} s"
            f"{row['percentage']:10.2f} %"
            f"    calls={calls}"
        )

    print("-" * 90)

    print(
        f"{'INSTRUMENTED TOTAL':30s}"
        f"{total_latency:12.4f} s"
        f"{100.0:10.2f} %"
    )

    transformer_calls = (
        profiler.counts.get(
            "transformer",
            0,
        )
    )

    if transformer_calls > 0:

        average_transformer = (
            profiler.total_time(
                "transformer"
            )
            / transformer_calls
        )

        print(
            "\nAverage Transformer forward: "
            f"{average_transformer:.6f} s/call"
        )

    scheduler_calls = (
        profiler.counts.get(
            "scheduler",
            0,
        )
    )

    if scheduler_calls > 0:

        average_scheduler = (
            profiler.total_time(
                "scheduler"
            )
            / scheduler_calls
        )

        print(
            "Average Scheduler update:    "
            f"{average_scheduler:.6f} s/call"
        )


# ============================================================
# COMMAND-LINE CONFIGURATION
# ============================================================


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Characterize Qwen-Image BF16 "
            "text-to-image inference latency, "
            "tensor shapes, and pipeline bottlenecks."
        )
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen-Image",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "/outputs/"
            "qwen_image_bf16_inference_characterization"
        ),
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--baseline-runs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "A red sports car parked on a quiet city street at sunset, "
            "with realistic lighting, detailed buildings, and reflections "
            "on the wet pavement."
        ),
    )

    return parser.parse_args()


# ============================================================
# EXPERIMENT ENTRY POINT
# ============================================================


def main() -> None:

    args = parse_arguments()

    config = ExperimentConfig(
        model_id=args.model_id,
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        seed=args.seed,
        baseline_runs=args.baseline_runs,
        warmup_steps=args.warmup_steps,
        output_dir=args.output_dir,
    )

    os.makedirs(
        config.output_dir,
        exist_ok=True,
    )

    print("\n" + "=" * 80)
    print("QWEN-IMAGE BF16 INFERENCE CHARACTERIZATION")
    print("=" * 80)

    print(f"Model:       {config.model_id}")
    print(
        f"Resolution:  "
        f"{config.width} x {config.height}"
    )
    print(
        f"Steps:       "
        f"{config.num_inference_steps}"
    )
    print(f"Seed:        {config.seed}")
    print(
        f"Baseline runs: "
        f"{config.baseline_runs}"
    )
    print(
        f"Output:      "
        f"{config.output_dir}"
    )

    # --------------------------------------------------------
    # 1. Environment characterization
    # --------------------------------------------------------

    environment = collect_environment()

    save_json(
        os.path.join(
            config.output_dir,
            "environment.json",
        ),
        environment,
    )

    # --------------------------------------------------------
    # 2. Load model ONCE
    # --------------------------------------------------------

    pipe, model_load_time = (
        load_pipeline(
            config
        )
    )

    experiment_metadata = {
        "experiment_configuration":
            asdict(config),

        "precision":
            "torch.bfloat16",

        "model_load_time_s":
            model_load_time,

        "pipeline_components": {
            "pipeline":
                pipe.__class__.__name__,

            "tokenizer":
                pipe.tokenizer.__class__.__name__,

            "text_encoder":
                pipe.text_encoder.__class__.__name__,

            "transformer":
                pipe.transformer.__class__.__name__,

            "vae":
                pipe.vae.__class__.__name__,

            "scheduler":
                pipe.scheduler.__class__.__name__,
        },

        "scheduler_configuration":
            dict(
                pipe.scheduler.config
            ),
    }

    save_json(
        os.path.join(
            config.output_dir,
            "experiment_metadata.json",
        ),
        experiment_metadata,
    )

    # --------------------------------------------------------
    # 3. Warm-up
    # --------------------------------------------------------

    warmup_time = run_warmup(
        pipe,
        config,
    )

    # --------------------------------------------------------
    # 4. Clean baseline benchmark
    # --------------------------------------------------------

    (
        baseline_rows,
        mean_latency,
        std_latency,
    ) = run_clean_baseline(
        pipe,
        config,
    )

    write_baseline_latency_csv(
        os.path.join(
            config.output_dir,
            "baseline_latency.csv",
        ),
        baseline_rows,
    )

    # --------------------------------------------------------
    # 5. Instrumented shape + latency profiling
    # --------------------------------------------------------

    (
        profiler,
        instrumented_total_latency,
    ) = run_instrumented_profile(
        pipe,
        config,
    )

    save_json(
        os.path.join(
            config.output_dir,
            "tensor_shapes.json",
        ),
        profiler.shapes,
    )

    pipeline_profile_rows = (
        build_pipeline_profile(
            profiler,
            instrumented_total_latency,
        )
    )

    write_pipeline_profile_csv(
        os.path.join(
            config.output_dir,
            "pipeline_latency_profile.csv",
        ),
        pipeline_profile_rows,
    )

    # --------------------------------------------------------
    # Prompt encoding sub-profile
    # --------------------------------------------------------

    prompt_encoding_total = (
        profiler.total_time(
            "prompt_encoding"
        )
    )

    tokenizer_time = (
        profiler.total_time(
            "tokenizer"
        )
    )

    text_encoder_time = (
        profiler.total_time(
            "text_encoder"
        )
    )

    prompt_other_time = max(
        0.0,
        prompt_encoding_total
        - tokenizer_time
        - text_encoder_time,
    )

    prompt_profile = {
        "prompt_encoding_total_s":
            prompt_encoding_total,

        "tokenizer": {
            "calls":
                profiler.counts.get(
                    "tokenizer",
                    0,
                ),
            "time_s":
                tokenizer_time,
        },

        "text_encoder": {
            "calls":
                profiler.counts.get(
                    "text_encoder",
                    0,
                ),
            "time_s":
                text_encoder_time,
        },

        "other_prompt_encoding_s":
            prompt_other_time,
    }

    save_json(
        os.path.join(
            config.output_dir,
            "prompt_encoding_profile.json",
        ),
        prompt_profile,
    )

    # --------------------------------------------------------
    # 6. Final experiment summary
    # --------------------------------------------------------

    transformer_calls = (
        profiler.counts.get(
            "transformer",
            0,
        )
    )

    transformer_total = (
        profiler.total_time(
            "transformer"
        )
    )

    transformer_average = (
        transformer_total
        / transformer_calls
        if transformer_calls > 0
        else 0.0
    )

    scheduler_calls = (
        profiler.counts.get(
            "scheduler",
            0,
        )
    )

    scheduler_total = (
        profiler.total_time(
            "scheduler"
        )
    )

    scheduler_average = (
        scheduler_total
        / scheduler_calls
        if scheduler_calls > 0
        else 0.0
    )

    experiment_summary = {
        "clean_baseline": {
            "number_of_runs":
                config.baseline_runs,

            "mean_latency_s":
                mean_latency,

            "std_latency_s":
                std_latency,
        },

        "instrumented_run": {
            "total_latency_s":
                instrumented_total_latency,

            "transformer": {
                "calls":
                    transformer_calls,

                "total_time_s":
                    transformer_total,

                "average_time_per_call_s":
                    transformer_average,

                "percentage_of_instrumented_runtime":
                    (
                        100.0
                        * transformer_total
                        / instrumented_total_latency
                    ),
            },

            "scheduler": {
                "calls":
                    scheduler_calls,

                "total_time_s":
                    scheduler_total,

                "average_time_per_call_s":
                    scheduler_average,
            },

            "prompt_encoding_time_s":
                prompt_encoding_total,

            "vae_decode_time_s":
                profiler.total_time(
                    "vae_decode"
                ),

            "image_postprocessing_time_s":
                profiler.total_time(
                    "image_postprocessing"
                ),
        },

        "model_initialization": {
            "model_load_time_s":
                model_load_time,
        },

        "warmup": {
            "steps":
                config.warmup_steps,

            "time_s":
                warmup_time,
        },
    }

    save_json(
        os.path.join(
            config.output_dir,
            "experiment_summary.json",
        ),
        experiment_summary,
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print_pipeline_profile(
        pipeline_profile_rows,
        instrumented_total_latency,
        profiler,
    )

    print("\n" + "=" * 80)
    print(
        "QWEN-IMAGE BF16 INFERENCE "
        "CHARACTERIZATION COMPLETE"
    )
    print("=" * 80)

    print(
        "\nClean baseline latency:"
    )

    print(
        f"    {mean_latency:.4f} "
        f"± {std_latency:.4f} s"
    )

    print(
        "\nInstrumented total latency:"
    )

    print(
        f"    "
        f"{instrumented_total_latency:.4f} s"
    )

    print(
        "\nTransformer share:"
    )

    print(
        f"    "
        f"{100.0 * transformer_total / instrumented_total_latency:.2f} %"
    )

    print(
        "\nResults saved to:"
    )

    print(
        f"    {config.output_dir}"
    )


if __name__ == "__main__":
    main()