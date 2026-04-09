"""
TRELLIS.2 generator for Modly v0.3.0+
Reference: https://github.com/microsoft/TRELLIS.2
"""

import io
import os
import sys
import time
import uuid
import zipfile
import threading
import tempfile
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from services.generators.base import BaseGenerator, smooth_progress

_HF_REPO_ID   = "microsoft/TRELLIS.2-4B"
_GITHUB_ZIP   = "https://github.com/microsoft/TRELLIS.2/archive/refs/heads/main.zip"

# ─────────────────────────────────────────────────────────────────── #

class Trellis2Generator(BaseGenerator):

    MODEL_ID     = "trellis-2-4b"
    DISPLAY_NAME = "TRELLIS.2-4B"
    VRAM_GB      = 24

    # ── Lifecycle ─────────────────────────────────────────────────── #

    def is_downloaded(self) -> bool:
        return (self.model_dir / "pipeline.json").exists()

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.is_downloaded():
            raise RuntimeError(
                "Model weights not found. "
                "Please download the model from the Extensions page first."
            )

        self._ensure_trellis2()

        import torch
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        print(f"[Trellis2Generator] Loading pipeline from {self.model_dir} ...")
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(self.model_dir))
        pipeline.cuda()
        self._model = pipeline
        print("[Trellis2Generator] Pipeline loaded on CUDA.")

    def unload(self) -> None:
        super().unload()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ── Inference ─────────────────────────────────────────────────── #

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
    ) -> Path:
        import torch
        import o_voxel

        resolution       = str(params.get("resolution", "1024"))
        decimation_target = int(params.get("decimation_target", 500000))
        texture_size     = int(params.get("texture_size", 2048))
        seed             = int(params.get("seed", -1))

        ss_steps         = int(params.get("ss_sampling_steps", 12))
        shape_steps      = int(params.get("shape_slat_sampling_steps", 12))
        tex_steps        = int(params.get("tex_slat_sampling_steps", 12))

        # pipeline_type mapping
        pipeline_type = {
            "512":  "512",
            "1024": "1024_cascade",
            "1536": "1536_cascade",
        }.get(resolution, "1024_cascade")

        # ── 1. Preprocess ─────────────────────────────────────────── #
        self._report(progress_cb, 5, "Preprocessing image...")
        image = self._preprocess(image_bytes)

        # ── 2. Generate ───────────────────────────────────────────── #
        self._report(progress_cb, 10, "Running TRELLIS.2 pipeline...")

        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 10, 80, "Generating 3D mesh...", stop_evt),
                daemon=True,
            )
            t.start()

        try:
            generator = None
            if seed >= 0:
                import torch as _torch
                generator = _torch.Generator().manual_seed(seed)

            outputs, latents = self._model.run(
                image,
                seed=seed if seed >= 0 else None,
                preprocess_image=False,
                sparse_structure_sampler_params={
                    "steps": ss_steps,
                    "guidance_strength": 7.5,
                    "guidance_rescale": 0.7,
                    "rescale_t": 5.0,
                },
                shape_slat_sampler_params={
                    "steps": shape_steps,
                    "guidance_strength": 7.5,
                    "guidance_rescale": 0.5,
                    "rescale_t": 3.0,
                },
                tex_slat_sampler_params={
                    "steps": tex_steps,
                    "guidance_strength": 1.0,
                    "guidance_rescale": 0.0,
                    "rescale_t": 3.0,
                },
                pipeline_type=pipeline_type,
                return_latent=True,
            )
        finally:
            stop_evt.set()

        mesh = outputs[0]
        mesh.simplify(16777216)  # nvdiffrast vertex limit

        # ── 3. Decode to GLB ──────────────────────────────────────── #
        self._report(progress_cb, 82, "Exporting GLB...")

        shape_slat, tex_slat, res = latents
        decoded = self._model.decode_latent(shape_slat, tex_slat, res)[0]

        glb = o_voxel.postprocess.to_glb(
            vertices         = decoded.vertices,
            faces            = decoded.faces,
            attr_volume      = decoded.attrs,
            coords           = decoded.coords,
            attr_layout      = self._model.pbr_attr_layout,
            grid_size        = res,
            aabb             = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target = decimation_target,
            texture_size     = texture_size,
            remesh           = True,
            remesh_band      = 1,
            remesh_project   = 0,
            use_tqdm         = False,
        )

        self._report(progress_cb, 96, "Saving file...")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        path = self.outputs_dir / name
        glb.export(str(path), extension_webp=True)

        torch.cuda.empty_cache()
        self._report(progress_cb, 100, "Done")
        return path

    # ── Helpers ───────────────────────────────────────────────────── #

    def _preprocess(self, image_bytes: bytes) -> Image.Image:
        import rembg
        img = Image.open(io.BytesIO(image_bytes))
        # すでにアルファ付きならそのまま使う
        if img.mode == "RGBA":
            return img
        return rembg.remove(img).convert("RGBA")

    def _ensure_trellis2(self) -> None:
        """trellis2 パッケージが import できなければ GitHub からソースを取得する"""
        try:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa
            return
        except ImportError:
            pass

        src_dir = self.model_dir / "_trellis2_src"
        if not (src_dir / "trellis2").exists():
            self._download_trellis2_src(src_dir)

        for extra in [str(src_dir), str(src_dir / "o-voxel")]:
            if extra not in sys.path:
                sys.path.insert(0, extra)

        # o_voxel は C 拡張なので build が必要な場合の案内
        try:
            import o_voxel  # noqa
        except ImportError:
            raise RuntimeError(
                "o_voxel (C++ extension) が見つかりません。\n"
                "以下のコマンドでビルドしてください:\n\n"
                f"  cd \"{src_dir / 'o-voxel'}\"\n"
                "  pip install . --no-build-isolation\n"
            )

        try:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa
        except ImportError as exc:
            raise RuntimeError(
                f"trellis2 が import できません: {exc}\n"
                f"ソース: {src_dir}"
            ) from exc

    def _download_trellis2_src(self, dest: Path) -> None:
        import urllib.request

        dest.mkdir(parents=True, exist_ok=True)
        print("[Trellis2Generator] Downloading TRELLIS.2 source from GitHub...")

        with urllib.request.urlopen(_GITHUB_ZIP, timeout=300) as resp:
            data = resp.read()

        print("[Trellis2Generator] Extracting source...")
        prefix = "TRELLIS.2-main/"
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                if not member.startswith(prefix):
                    continue
                rel = member[len(prefix):]
                if not rel:
                    continue
                target = dest / rel
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))

        print(f"[Trellis2Generator] Source extracted to {dest}.")

    # ── Schema ────────────────────────────────────────────────────── #

    @classmethod
    def params_schema(cls) -> list:
        return [
            {
                "id": "resolution",
                "label": "Resolution",
                "type": "select",
                "default": "1024",
                "options": [
                    {"value": "512",  "label": "512 (Fast, ~8GB VRAM)"},
                    {"value": "1024", "label": "1024 (Balanced, ~16GB VRAM)"},
                    {"value": "1536", "label": "1536 (High Quality, ~24GB VRAM)"},
                ],
            },
            {
                "id": "decimation_target",
                "label": "Decimation Target (faces)",
                "type": "int",
                "default": 500000,
                "min": 100000,
                "max": 1000000,
            },
            {
                "id": "texture_size",
                "label": "Texture Size",
                "type": "select",
                "default": 2048,
                "options": [
                    {"value": 1024, "label": "1024"},
                    {"value": 2048, "label": "2048 (Default)"},
                    {"value": 4096, "label": "4096 (High)"},
                ],
            },
            {
                "id": "seed",
                "label": "Seed (-1 = random)",
                "type": "int",
                "default": -1,
                "min": -1,
                "max": 2147483647,
            },
            {
                "id": "ss_sampling_steps",
                "label": "Sparse Structure Steps",
                "type": "int",
                "default": 12,
                "min": 1,
                "max": 50,
            },
            {
                "id": "shape_slat_sampling_steps",
                "label": "Shape SLAT Steps",
                "type": "int",
                "default": 12,
                "min": 1,
                "max": 50,
            },
            {
                "id": "tex_slat_sampling_steps",
                "label": "Texture SLAT Steps",
                "type": "int",
                "default": 12,
                "min": 1,
                "max": 50,
            },
        ]
