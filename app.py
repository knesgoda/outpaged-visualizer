import base64
import difflib
import io
import json
import math
import os
import re
import time
import zipfile

import numpy as np
import requests
import streamlit as st
from docx import Document
from PIL import Image, ImageFilter, ImageOps

BLOCKADE_BASE = "https://backend.blockadelabs.com/api/v1"
STABILITY_CORE_URL = "https://api.stability.ai/v2beta/stable-image/generate/core"
STABILITY_ULTRA_URL = "https://api.stability.ai/v2beta/stable-image/generate/ultra"
STABILITY_MAX_UPLOAD_BYTES = 9 * 1024 * 1024
POLE_DETAIL_STRIP_RATIO = 0.10
POLE_DETAIL_WEIGHT_DEFAULT = 1.0
POLE_DETAIL_EPS = 1e-6

st.set_page_config(page_title="OutPaged Visualizer", layout="wide")
st.title("OutPaged Visualizer")
st.caption("Bulk-generate backgrounds, character training sets, and skyboxes from your DOCX prompts.")

PADDING_STYLE_OPTIONS = {
    "Blurred edges": "blur",
    "Mirrored edges": "mirror",
    "Solid fill": "solid",
}

FRAMING_MODE_OPTIONS = {
    "Crop to 2:1 (no bars)": "crop",
    "Pad to 2:1 (no stretch)": "pad",
}

SKYBOX_ASPECT_OPTIONS = {
    "Crop to 2:1 (recommended)": "crop",
    "Leave as-is": "leave",
}

ANTI_LETTERBOX_TOKENS = [
    "letterbox",
    "letterboxing",
    "black bars",
    "cinematic bars",
    "frame",
    "border",
    "vignette",
]

SKYBOX_PROMPT_WRAPPER_TOKENS = [
    "full 360° equirectangular panorama (2:1)",
    "detailed zenith (ceiling) and nadir (floor)",
    "no smeared/blurred poles",
    "no vignetting",
    "no extreme wide-angle distortion, no fisheye",
]
SKYBOX_INIT_DETAIL_TOKENS = [
    "detailed ceiling beams/plaster texture at zenith",
    "detailed floorboards/rug texture at nadir",
]
SKYBOX_INIT_WRAPPER_TOKENS = [
    "skybox init plate",
    "evenly distributed detail",
    "no close foreground occluders",
]

STYLE_PACKS = {
    "default": "3D watercolor, warm natural light, storybook realism",
    "wizard_of_oz": "storybook watercolor and ink, slightly magical realism",
    "gullivers_travels": "vivid chalk pastel illustration, deep 3D chalk texture",
    "jekyll_hyde": "gritty photorealistic Victorian, moody fog, gaslight",
}

EMPTY_ENVIRONMENT_GUARDRAILS = (
    "empty environment, empty room, environment only, no characters present, "
    "no creatures, no toys, no plush, no figurines, no stuffed animals, "
    "no teddy bear, no dolls, no storybook characters"
)

NEGATIVE_INTERIOR = (
    "people, person, faces, bodies, hands, crowds, animals, pets, anthropomorphic animals, "
    "plush toys, teddy bears, dolls, cartoon characters, mascots, character costumes, "
    "text, letters, signage, watermark, logo, UI, modern electronics, cameras, camera rigs, "
    "microphones, tripods, screens, TVs, laptops, phones, wires, outlets, modern vehicles, "
    "anachronistic items, sci-fi, plastic, neon, LED, blurry, low-res, artifacts"
)
NEGATIVE_EXTERIOR = (
    "people, person, faces, bodies, hands, crowds, animals, pets, anthropomorphic animals, "
    "plush toys, teddy bears, dolls, cartoon characters, mascots, character costumes, "
    "text, letters, signage, watermark, logo, UI, modern electronics, cameras, camera rigs, "
    "microphones, tripods, screens, TVs, laptops, phones, wires, outlets, modern vehicles, "
    "anachronistic items, sci-fi, plastic, neon, LED, blurry, low-res, artifacts"
)

SKYBOX_HARD_NEGATIVE_DEFAULT = ""

class PromptPolicy:
    def __init__(
        self,
        style_pack: str = "default",
        era_locale_tags: str | None = None,
    ) -> None:
        self.style_pack = style_pack if style_pack in STYLE_PACKS else "default"
        self.era_locale_tags = era_locale_tags.strip() if era_locale_tags else ""

    def base_prefix(self, environment_type: str, output_type: str) -> str:
        env = environment_type.lower()
        output = output_type.lower()
        if output == "skybox":
            if env == "exterior":
                return (
                    "full 360° equirectangular panorama (2:1), ground view, eye level, "
                    "wide panoramic environment plate, horizon stable, evenly distributed detail, "
                    "no close foreground occluders"
                )
            return (
                "full 360° equirectangular panorama (2:1), ground view, eye level, "
                "wide-angle interior environment plate, camera centered in the room, "
                "continuous walls and ceiling, natural perspective, soft depth, "
                "no close foreground occluders"
            )
        if output == "background":
            return (
                "cinematic wide establishing shot, strong depth, "
                "clear foreground/midground/background separation, environment only"
            )
        return ""

    def style_suffix(self) -> str:
        return STYLE_PACKS.get(self.style_pack, STYLE_PACKS["default"])

    def negative_prompt(self, environment_type: str) -> str:
        env = environment_type.lower()
        return NEGATIVE_EXTERIOR if env == "exterior" else NEGATIVE_INTERIOR

    def build_prompt(
        self,
        scene_text: str,
        environment_type: str,
        output_type: str,
    ) -> str:
        parts = [
            self.base_prefix(environment_type, output_type),
            scene_text,
            self.era_locale_tags,
            EMPTY_ENVIRONMENT_GUARDRAILS,
            self.style_suffix(),
        ]
        cleaned = ", ".join(part for part in parts if part)
        cleaned = re.sub(r"\s*,\s*", ", ", cleaned).strip(" ,")
        return cleaned

    def build_skybox_init_prompt(
        self,
        scene_text: str,
        environment_type: str,
    ) -> str:
        base_prompt = self.build_prompt(scene_text, environment_type, "skybox")
        tokens = SKYBOX_INIT_WRAPPER_TOKENS + SKYBOX_INIT_DETAIL_TOKENS
        return _append_prompt_tokens(base_prompt, tokens)

with st.sidebar:
    st.header("Output options")
    panoramic_enabled = st.checkbox(
        "Enable 2:1 post-process",
        value=False,
        help="Convert outputs to a 2:1 framing after generation.",
        key="panoramic_enabled",
    )
    panoramic_mode_label = st.selectbox(
        "2:1 framing mode",
        list(FRAMING_MODE_OPTIONS.keys()),
        index=0,
        disabled=not panoramic_enabled,
        key="panoramic_mode",
    )
    panoramic_mode = FRAMING_MODE_OPTIONS[panoramic_mode_label]
    panoramic_style_label = st.selectbox(
        "Panoramic padding style",
        list(PADDING_STYLE_OPTIONS.keys()),
        index=0,
        disabled=not panoramic_enabled or panoramic_mode != "pad",
        key="panoramic_style",
    )
    panoramic_style = PADDING_STYLE_OPTIONS[panoramic_style_label]
    st.header("Prompt policy")
    style_pack_label = st.selectbox(
        "Style pack",
        list(STYLE_PACKS.keys()),
        index=0,
        key="prompt_style_pack",
    )
    era_locale_tags = st.text_input(
        "Era/locale tags (optional)",
        value="",
        key="prompt_era_locale_tags",
    )
    environment_type_label = st.selectbox(
        "Environment type",
        ["Auto", "Interior", "Exterior"],
        index=0,
        key="prompt_environment_type",
    )

    st.header("Reuse options")
    cache_mode_label = st.selectbox(
        "Cache mode",
        ["OFF", "APPROVAL ONLY", "ALWAYS"],
        index=1,
        key="cache_mode",
    )
    reuse_location_cache = st.checkbox(
        "Reuse location cache",
        value=True,
        help="Disable to avoid reusing cached init images or cached seeds.",
        key="reuse_location_cache",
    )
    location_reuse_enabled = st.checkbox(
        "Enable location-based reuse",
        value=False,
        help="Only reuse cached locations when a Location: line is provided.",
        key="location_reuse_enabled",
    )
    reuse_cached_seed = st.checkbox(
        "Reuse cached seed",
        value=False,
        help="Allow cached seeds to override manual seed offsets.",
        key="reuse_cached_seed",
    )
    if st.button("Clear location cache", key="clear_location_cache"):
        st.session_state["bg_cache"] = {}
        st.session_state["char_cache"] = {}
        st.session_state["skybox_cache"] = {}
        st.session_state["location_cache"] = {}
        st.session_state["pending_cache"] = {"background": {}, "skybox": {}}
        st.success("Cleared location cache.")
    st.header("Skybox init scoring")
    avoid_blurred_poles = st.checkbox(
        "Avoid blurred skybox poles (prefer ceiling/floor detail)",
        value=True,
        key="skybox_avoid_blurred_poles",
    )
    pole_detail_weight = st.slider(
        "Pole detail weight",
        min_value=0.0,
        max_value=3.0,
        value=POLE_DETAIL_WEIGHT_DEFAULT,
        step=0.1,
        disabled=not avoid_blurred_poles,
        key="skybox_pole_detail_weight",
    )
    st.header("Skybox safeguards")
    neutralize_possessives = st.checkbox(
        "Neutralize character-name possessives",
        value=True,
        key="skybox_neutralize_possessives",
    )
    skybox_exclude_animals = st.checkbox(
        "Skyboxes: exclude animals",
        value=True,
        key="skybox_exclude_animals",
    )
    skybox_hard_negative_text = st.text_area(
        "Skybox hard negative list",
        value=SKYBOX_HARD_NEGATIVE_DEFAULT,
        height=140,
        key="skybox_hard_negative_text",
    )

if "bg_cache" not in st.session_state:
    st.session_state["bg_cache"] = {}
if "char_cache" not in st.session_state:
    st.session_state["char_cache"] = {}
if "skybox_cache" not in st.session_state:
    st.session_state["skybox_cache"] = {}
if "location_cache" not in st.session_state:
    st.session_state["location_cache"] = {}
if "pending_cache" not in st.session_state:
    st.session_state["pending_cache"] = {"background": {}, "skybox": {}}

def _b64_of_uploaded_file(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    return base64.b64encode(data).decode("utf-8")

def get_secret(key: str) -> str:
    try:
        value = st.secrets.get(key, "")
    except FileNotFoundError:
        value = ""
    if value:
        return value
    return os.getenv(key, "")

def blockade_api_key() -> str:
    return get_secret("BLOCKADE_API_KEY")

def blockade_headers(api_key: str) -> dict:
    return {
        "x-api-key": api_key,
        "accept": "application/json",
        "content-type": "application/json",
    }

@st.cache_data(ttl=3600)
def blockade_get_styles(model_version: int = 3, api_key: str = ""):
    # Docs: Get Skybox Styles. :contentReference[oaicite:10]{index=10}
    url = f"{BLOCKADE_BASE}/skybox/styles"
    resp = requests.get(
        url,
        headers=blockade_headers(api_key),
        params={"model_version": model_version},
        timeout=60,
    )
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        detail = resp.text
        try:
            detail = resp.json()
        except ValueError:
            pass
        raise RuntimeError(
            f"Blockade styles request failed ({resp.status_code}): {detail}"
        ) from exc
    return resp.json()

@st.cache_data(ttl=3600)
def blockade_get_export_types(api_key: str = ""):
    endpoints = (
        "skybox/export-types",
        "skybox/exports/types",
        "skybox/export/types",
    )
    last_error: RuntimeError | None = None
    for endpoint in endpoints:
        url = f"{BLOCKADE_BASE}/{endpoint}"
        resp = requests.get(url, headers=blockade_headers(api_key), timeout=60)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            detail = resp.text
            try:
                detail = resp.json()
            except ValueError:
                pass
            last_error = RuntimeError(
                f"Blockade export types request failed ({resp.status_code}): {detail}"
            )
            if resp.status_code == 404:
                continue
            raise last_error from exc
        return resp.json()
    if last_error:
        raise last_error
    raise RuntimeError("Blockade export types request failed: no endpoints available.")

def _build_label_index(items: list[dict] | None) -> dict[str, int]:
    label_index: dict[str, int] = {}
    for item in items or []:
        label = item.get("label") or item.get("name") or item.get("title")
        if label is None:
            continue
        item_id = item.get("id")
        if item_id is None:
            continue
        label_index[str(label)] = int(item_id)
    return label_index

def _extract_export_types(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("types", "export_types", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []

def _extract_export_resolutions(payload, export_types: list[dict]) -> list[dict]:
    if isinstance(payload, dict):
        for key in ("resolutions", "export_resolutions"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    for export_type in export_types or []:
        resolutions = export_type.get("resolutions")
        if isinstance(resolutions, list):
            return resolutions
    return []

def blockade_generate_skybox(
    prompt: str,
    style_id: int,
    negative_text: str = "",
    seed: int | None = 0,
    enhance_prompt: bool = False,
    init_image_b64: str | None = None,
    init_strength: float = 0.5,
    control_image_b64: str | None = None,
    control_model: str = "remix",
    api_key: str = "",
):
    # Docs: POST /skybox parameters include prompt, negative_text, seed,
    # enhance_prompt, control_image/control_model, init_image/init_strength. :contentReference[oaicite:11]{index=11}
    payload = {
        "skybox_style_id": int(style_id),
        "prompt": prompt,
        "enhance_prompt": bool(enhance_prompt),
    }
    if negative_text.strip():
        payload["negative_text"] = negative_text.strip()
    if isinstance(seed, int) and seed > 0:
        payload["seed"] = int(seed)

    # Prefer init_image if provided; control_image is more "structure only".
    if init_image_b64:
        payload["init_image"] = init_image_b64
        payload["init_strength"] = float(init_strength)

    if control_image_b64:
        payload["control_image"] = control_image_b64
        payload["control_model"] = control_model  # required for remix from control image :contentReference[oaicite:12]{index=12}

    resp = requests.post(
        f"{BLOCKADE_BASE}/skybox",
        headers=blockade_headers(api_key),
        json=payload,
        timeout=120,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Blockade error {resp.status_code}: {resp.text}")
    return resp.json()

def blockade_poll_generation(obfuscated_id: str, sleep_s: float = 2.0, max_wait_s: int = 300):
    # Docs mention tracking generation progress; you can poll by obfuscated id. :contentReference[oaicite:13]{index=13}
    # Endpoint shown in docs nav: "Get Skybox by Obfuscated id"
    url = f"{BLOCKADE_BASE}/skybox/{obfuscated_id}"
    headers = blockade_headers(blockade_api_key())
    waited = 0
    while waited < max_wait_s:
        r = requests.get(url, headers=headers, timeout=60)
        if r.status_code == 404:
            time.sleep(sleep_s)
            waited += sleep_s
            continue
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            detail = r.text
            try:
                detail = r.json()
            except ValueError:
                pass
            raise RuntimeError(
                f"Skybox status request failed ({r.status_code}): {detail}"
            ) from exc
        data = r.json()
        if data.get("status") == "complete" and data.get("file_url"):
            return data
        if data.get("status") in ("error", "abort"):
            raise RuntimeError(f"Skybox failed: {data.get('error_message')}")
        time.sleep(sleep_s)
        waited += sleep_s
    raise TimeoutError("Skybox generation timed out while polling.")

def blockade_request_export(skybox_obfuscated_id: str, type_id: int, resolution_id: int | None = None):
    # Docs: POST /skybox/export with skybox_id (obfuscated_id), type_id, optional resolution_id. :contentReference[oaicite:14]{index=14}
    payload = {
        "skybox_id": skybox_obfuscated_id,
        "type_id": int(type_id),
    }
    if resolution_id is not None:
        payload["resolution_id"] = int(resolution_id)

    resp = requests.post(
        f"{BLOCKADE_BASE}/skybox/export",
        headers=blockade_headers(blockade_api_key()),
        json=payload,
        timeout=120,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Export error {resp.status_code}: {resp.text}")
    return resp.json()

def blockade_poll_export(export_id: str, sleep_s: float = 2.0, max_wait_s: int = 300):
    # Docs: GET /skybox/export/{export.id} returns file_url + status. :contentReference[oaicite:15]{index=15}
    url = f"{BLOCKADE_BASE}/skybox/export/{export_id}"
    headers = blockade_headers(blockade_api_key())
    waited = 0
    while waited < max_wait_s:
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "complete" and data.get("file_url"):
            return data
        if data.get("status") in ("error", "abort"):
            raise RuntimeError(f"Export failed: {data.get('error_message')}")
        time.sleep(sleep_s)
        waited += sleep_s
    raise TimeoutError("Export timed out while polling.")

def download_url_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    return r.content

def stability_headers(api_key: str) -> dict:
    return {
        "authorization": f"Bearer {api_key}",
        "accept": "image/*",
    }

def stability_api_key() -> str:
    api_key = get_secret("STABILITY_API_KEY") or get_secret("STABILITY_KEY")
    if not api_key:
        st.error(
            "Missing STABILITY_API_KEY (or STABILITY_KEY) in Streamlit Secrets. The key "
            "needs access to Stability.ai Stable Image Core (v2beta) generation."
        )
        st.stop()
    return api_key

def stability_generate_images(
    prompt: str,
    negative_prompt: str,
    seed: int | None,
    aspect_ratio: str,
    image_count: int,
    api_key: str,
    init_image_bytes: bytes | None = None,
    init_strength: float | None = None,
):
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "aspect_ratio": aspect_ratio,
        "output_format": "png",
        "samples": image_count,
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if init_image_bytes is not None and init_strength is not None:
        payload["strength"] = str(init_strength)

    files = {key: (None, str(value)) for key, value in payload.items()}
    if init_image_bytes is not None:
        init_image_bytes, init_filename, init_mime, resized = _prepare_stability_init_image(
            init_image_bytes
        )
        if resized:
            st.info(
                "Reference image was resized/compressed to stay within Stability.ai upload limits."
            )
        files["image"] = (init_filename, init_image_bytes, init_mime)

    resp = requests.post(
        STABILITY_CORE_URL,
        headers=stability_headers(api_key),
        files=files,
        timeout=180,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Stability.ai error {resp.status_code}: {resp.text}")

    content_type = resp.headers.get("content-type", "")
    if content_type.startswith("image/"):
        return [resp.content]

    if "application/zip" in content_type:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zip_file:
            return [zip_file.read(name) for name in zip_file.namelist()]

    data = resp.json()
    if "image" in data:
        return [base64.b64decode(data["image"])]
    if "images" in data:
        return [base64.b64decode(item) for item in data["images"]]
    if "artifacts" in data:
        return [
            base64.b64decode(item.get("base64") or item.get("image"))
            for item in data["artifacts"]
            if item.get("base64") or item.get("image")
        ]
    raise RuntimeError("Stability.ai response did not include images.")

def stability_generate_ultra(
    prompt: str,
    negative_prompt: str,
    seed: int | None,
    aspect_ratio: str,
    api_key: str,
) -> bytes:
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "aspect_ratio": aspect_ratio,
        "output_format": "png",
    }
    if seed is not None:
        payload["seed"] = int(seed)

    files = {key: (None, str(value)) for key, value in payload.items()}
    resp = requests.post(
        STABILITY_ULTRA_URL,
        headers=stability_headers(api_key),
        files=files,
        timeout=180,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Stability Ultra error {resp.status_code}: {resp.text}")
    return resp.content

def make_skybox_init_from_stability(
    scene_prompt: str,
    negative_prompt: str,
    base_seed: int | None,
    api_key: str,
    candidates: int = 3,
    make_tileable: bool = True,
    avoid_blurred_poles: bool = True,
    pole_detail_weight: float = POLE_DETAIL_WEIGHT_DEFAULT,
    neutralize_possessives: bool = False,
) -> tuple[bytes, int | None]:
    if neutralize_possessives:
        scene_prompt = neutralize_character_possessives(scene_prompt)
    prompt = _build_skybox_init_prompt(scene_prompt)

    source_aspect = "16:9"

    best_bytes = None
    best_seed = None
    best_score = None

    seed0 = None if (base_seed is None or base_seed <= 0) else int(base_seed)

    candidate_count = max(candidates, 5) if avoid_blurred_poles else candidates
    for i in range(candidate_count):
        seed_i = None if seed0 is None else seed0 + i
        raw = stability_generate_ultra(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed_i,
            aspect_ratio=source_aspect,
            api_key=api_key,
        )
        cropped = prepare_skybox_ready_bytes(
            raw,
            target_size=(2048, 1024),
            make_tileable=False,
        )
        seam_score = edge_seam_score(cropped)
        pole_score = pole_detail_score(cropped) if avoid_blurred_poles else 0.0
        combined_score = seam_score
        if avoid_blurred_poles:
            combined_score = seam_score + (pole_detail_weight / (pole_score + POLE_DETAIL_EPS))
        fixed = cropped
        if make_tileable:
            fixed = prepare_skybox_ready_bytes(
                raw,
                target_size=(2048, 1024),
                make_tileable=True,
            )

        if best_score is None or combined_score < best_score:
            best_score = combined_score
            best_bytes = fixed
            best_seed = seed_i

    if best_bytes is None:
        raise RuntimeError("Stability Ultra did not return any images for skybox init.")
    return best_bytes, best_seed

def _encode_image_for_upload(image: Image.Image, use_png: bool) -> tuple[bytes, str, str]:
    buffer = io.BytesIO()
    if use_png:
        image.save(buffer, format="PNG", optimize=True, compress_level=9)
        return buffer.getvalue(), "init.png", "image/png"
    image.save(
        buffer,
        format="JPEG",
        quality=85,
        optimize=True,
        progressive=True,
    )
    return buffer.getvalue(), "init.jpg", "image/jpeg"

def _prepare_stability_init_image(
    init_image_bytes: bytes,
    max_bytes: int = STABILITY_MAX_UPLOAD_BYTES,
    min_dimension: int = 256,
) -> tuple[bytes, str, str, bool]:
    if len(init_image_bytes) <= max_bytes:
        return init_image_bytes, "init.png", "image/png", False

    try:
        with Image.open(io.BytesIO(init_image_bytes)) as image:
            image_copy = image.copy()
    except OSError:
        return init_image_bytes, "init.png", "image/png", False

    has_alpha = image_copy.mode in ("RGBA", "LA") or (
        image_copy.mode == "P" and "transparency" in image_copy.info
    )
    use_png = bool(has_alpha)
    encoded, filename, mime_type = _encode_image_for_upload(image_copy, use_png)

    width, height = image_copy.size
    resized = image_copy
    while len(encoded) > max_bytes and min(width, height) > min_dimension:
        scale = (max_bytes / len(encoded)) ** 0.5
        scale = max(0.5, min(0.9, scale))
        width = max(min_dimension, int(width * scale))
        height = max(min_dimension, int(height * scale))
        resized = resized.resize((width, height), Image.LANCZOS)
        encoded, filename, mime_type = _encode_image_for_upload(resized, use_png)

    return encoded, filename, mime_type, True

def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, compress_level=9)
    return buffer.getvalue()

def _center_crop_to_ratio(img: Image.Image, target_ratio: float) -> Image.Image:
    width, height = img.size
    current = width / height
    if abs(current - target_ratio) < 1e-6:
        return img

    if current > target_ratio:
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        return img.crop((left, 0, left + new_width, height))

    new_height = int(width / target_ratio)
    top = (height - new_height) // 2
    return img.crop((0, top, width, top + new_height))

def _make_tileable_horizontal(img: Image.Image, blend_px: int = 64) -> Image.Image:
    """
    Makes left/right edges more compatible by doing a wrap shift and blending the seam.
    This is a lightweight way to reduce visible seams when Skybox wraps the pano.
    """
    img = img.convert("RGB")
    width, height = img.size

    shift = width // 2
    shifted = Image.new("RGB", (width, height))
    shifted.paste(img.crop((shift, 0, width, height)), (0, 0))
    shifted.paste(img.crop((0, 0, shift, height)), (width - shift, 0))

    seam_x = width // 2
    band_left = max(0, seam_x - blend_px)
    band_right = min(width, seam_x + blend_px)

    left_band = shifted.crop((band_left, 0, seam_x, height))
    right_band = shifted.crop((seam_x, 0, band_right, height))

    left_arr = np.array(left_band).astype("float32")
    right_arr = np.array(right_band).astype("float32")

    band_width = right_arr.shape[1]
    alpha = np.linspace(0.0, 1.0, band_width, dtype="float32")[None, :, None]
    alpha = np.repeat(alpha, height, axis=0)

    blended = (left_arr[:, :band_width, :] * (1.0 - alpha)) + (right_arr[:, :band_width, :] * alpha)
    blended_img = Image.fromarray(np.clip(blended, 0, 255).astype("uint8"), mode="RGB")

    out = shifted.copy()
    out.paste(blended_img, (seam_x, 0))

    unshifted = Image.new("RGB", (width, height))
    unshifted.paste(out.crop((shift, 0, width, height)), (0, 0))
    unshifted.paste(out.crop((0, 0, shift, height)), (width - shift, 0))
    return unshifted

def prepare_skybox_ready_bytes(
    image_bytes: bytes,
    target_size: tuple[int, int] = (2048, 1024),
    make_tileable: bool = True,
    output_format: str = "PNG",
) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image = _center_crop_to_ratio(image, 2.0)
        image = image.resize(target_size, Image.LANCZOS)
        if make_tileable:
            image = _make_tileable_horizontal(image, blend_px=64)

        buffer = io.BytesIO()
        if output_format.upper() in ("JPG", "JPEG"):
            image.save(buffer, format="JPEG", quality=92, optimize=True, progressive=True)
        else:
            image.save(buffer, format="PNG", optimize=True, compress_level=9)
        return buffer.getvalue()

def edge_seam_score(image_bytes: bytes) -> float:
    """
    Lower is better. Measures how similar the left and right edges are.
    Cheap heuristic to pick the best candidate before spending Skybox credits.
    """
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        arr = np.array(image).astype("float32")
        strip = 16
        left = arr[:, :strip, :]
        right = arr[:, -strip:, :]
        return float(((left - right) ** 2).mean())

def pole_detail_score(image_bytes: bytes) -> float:
    """
    Higher is better. Measures texture/detail in the top and bottom bands.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
    except OSError:
        return 0.0
    arr = np.array(image).astype("float32")
    if arr.ndim != 3 or arr.shape[0] == 0:
        return 0.0
    height = arr.shape[0]
    strip = max(1, int(round(height * POLE_DETAIL_STRIP_RATIO)))
    if strip < 2 or arr.shape[1] < 2:
        return 0.0
    lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    top = lum[:strip, :]
    bottom = lum[-strip:, :]
    top_detail = np.abs(np.diff(top, axis=0)).mean() + np.abs(np.diff(top, axis=1)).mean()
    bottom_detail = np.abs(np.diff(bottom, axis=0)).mean() + np.abs(np.diff(bottom, axis=1)).mean()
    return float((top_detail + bottom_detail) / 2.0)

def _build_blurred_background(
    image: Image.Image,
    canvas_width: int,
    canvas_height: int,
) -> Image.Image:
    cover_scale = max(canvas_width / image.width, canvas_height / image.height)
    cover_size = (max(1, int(image.width * cover_scale)), max(1, int(image.height * cover_scale)))
    cover = image.resize(cover_size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=24))
    left = max(0, (cover.width - canvas_width) // 2)
    upper = max(0, (cover.height - canvas_height) // 2)
    return cover.crop((left, upper, left + canvas_width, upper + canvas_height))

def _build_mirrored_background(
    image: Image.Image,
    canvas_width: int,
    canvas_height: int,
) -> Image.Image:
    tile = Image.new("RGBA", (image.width * 2, image.height * 2))
    tile.paste(image, (0, 0))
    tile.paste(ImageOps.mirror(image), (image.width, 0))
    tile.paste(ImageOps.flip(image), (0, image.height))
    tile.paste(ImageOps.mirror(ImageOps.flip(image)), (image.width, image.height))

    background = Image.new("RGBA", (canvas_width, canvas_height))
    for y in range(0, canvas_height, tile.height):
        for x in range(0, canvas_width, tile.width):
            background.paste(tile, (x, y))
    return background

def remove_letterbox_bars(
    image_bytes: bytes,
    min_bar_px: int = 10,
    max_bar_ratio: float = 0.25,
) -> tuple[bytes, int, int, bool]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb = image.convert("RGB")
    except OSError:
        return image_bytes, 0, 0, False

    arr = np.array(rgb).astype("float32")
    height, width, _ = arr.shape
    if height < min_bar_px * 2:
        return image_bytes, 0, 0, False

    lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    row_mean = lum.mean(axis=1)
    row_var = lum.var(axis=1)

    interior_start = int(height * 0.25)
    interior_end = int(height * 0.75)
    interior_mean = float(np.median(row_mean[interior_start:interior_end]))
    interior_var = float(np.median(row_var[interior_start:interior_end])) + 1e-6

    max_bar_px = int(height * max_bar_ratio)

    def _scan_bar(row_indices: range) -> int:
        bar_height = 0
        for idx in row_indices:
            dark_enough = row_mean[idx] < max(10.0, interior_mean * 0.6)
            low_variance = row_var[idx] < interior_var * 0.15
            if dark_enough and low_variance:
                bar_height += 1
                if bar_height > max_bar_px:
                    break
            else:
                break
        return bar_height

    top_bar = _scan_bar(range(0, height))
    bottom_bar = _scan_bar(range(height - 1, -1, -1))

    if not (min_bar_px <= top_bar <= max_bar_px):
        top_bar = 0
    if not (min_bar_px <= bottom_bar <= max_bar_px):
        bottom_bar = 0

    if top_bar == 0 and bottom_bar == 0:
        return image_bytes, 0, 0, False

    crop_top = top_bar
    crop_bottom = height - bottom_bar
    if crop_top >= crop_bottom:
        return image_bytes, 0, 0, False

    cropped = rgb.crop((0, crop_top, width, crop_bottom))
    return _image_to_png_bytes(cropped), top_bar, bottom_bar, True

def crop_to_ratio(image: Image.Image, target_ratio: float) -> Image.Image:
    return _center_crop_to_ratio(image, target_ratio)

def pad_to_ratio(
    image: Image.Image,
    target_ratio: float,
    padding_style: str,
) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        return image
    current_aspect = width / height
    if current_aspect > target_ratio:
        canvas_width = width
        canvas_height = max(1, int(round(width / target_ratio)))
    else:
        canvas_height = height
        canvas_width = max(1, int(round(height * target_ratio)))

    scale = min(canvas_width / width, canvas_height / height)
    resized_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    resized = image.resize(resized_size, Image.LANCZOS)

    if padding_style == "blur":
        # Padding uses a blurred stretch of the image to avoid harsh solid bars.
        background = _build_blurred_background(resized, canvas_width, canvas_height)
    elif padding_style == "mirror":
        # Padding mirrors the image edges outward so the extra area feels continuous.
        background = _build_mirrored_background(resized, canvas_width, canvas_height)
    else:
        # Solid padding keeps the focus on the center image without visual noise.
        background = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 255))

    paste_x = (canvas_width - resized.width) // 2
    paste_y = (canvas_height - resized.height) // 2
    background.paste(resized, (paste_x, paste_y), resized)
    return background

def postprocess_to_2to1(
    image_bytes: bytes,
    enabled: bool,
    mode: str,
    padding_style: str,
) -> bytes:
    if not enabled:
        return image_bytes
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGBA")
            cleaned_bytes, _, _, _ = remove_letterbox_bars(_image_to_png_bytes(image))
            with Image.open(io.BytesIO(cleaned_bytes)) as cleaned:
                image = cleaned.convert("RGBA")
            if mode == "crop":
                processed = crop_to_ratio(image, 2.0)
            else:
                processed = pad_to_ratio(image, 2.0, padding_style)
            return _image_to_png_bytes(processed)
    except OSError:
        return image_bytes

def _apply_panoramic_conversion(
    image_bytes: bytes,
    enabled: bool,
    mode: str,
    padding_style: str,
) -> bytes:
    if not enabled:
        return image_bytes
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
    except OSError:
        return image_bytes
    if height == 0 or width == 0:
        return image_bytes
    current_ratio = width / height
    if abs(current_ratio - 2.0) <= 0.01:
        return image_bytes
    return postprocess_to_2to1(image_bytes, enabled, mode, padding_style)

def convert_to_panoramic(
    image_bytes: bytes,
    target_ratio: tuple[int, int],
    padding_style: str,
) -> bytes:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGBA")
            ratio = target_ratio[0] / target_ratio[1]
            width, height = image.size
            if height == 0 or width == 0:
                return image_bytes
            current_ratio = width / height
            if abs(current_ratio - ratio) <= 0.01:
                return image_bytes
            if abs(ratio - 2.0) <= 0.01:
                cropped = _center_crop_to_ratio(image, ratio)
                return _image_to_png_bytes(cropped)
            padded = pad_to_ratio(image, ratio, padding_style)
            return _image_to_png_bytes(padded)
    except OSError:
        return image_bytes

def postprocess_image_bytes(
    image_bytes: bytes,
    intended_ratio_tuple: tuple[int, int],
    padding_style: str,
    remove_bars: bool = True,
) -> bytes:
    processed_bytes = image_bytes
    if remove_bars:
        processed_bytes, top_px, bottom_px, did_remove = remove_letterbox_bars(processed_bytes)
        if did_remove:
            st.write(f"Removed letterbox bars: top={top_px}px bottom={bottom_px}px")
    ratio = intended_ratio_tuple[0] / intended_ratio_tuple[1]
    try:
        with Image.open(io.BytesIO(processed_bytes)) as image:
            width, height = image.size
    except OSError:
        return processed_bytes
    if height == 0 or width == 0:
        return processed_bytes
    current_ratio = width / height
    if abs(current_ratio - ratio) > 1e-6:
        processed_bytes = convert_to_panoramic(processed_bytes, intended_ratio_tuple, padding_style)
    return processed_bytes

def read_docx_text(uploaded_file) -> str:
    document = Document(uploaded_file)
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)

def _sanitize_prompt_text(text: str) -> str:
    cleaned = re.sub(
        r"\bstreetview,\s*viewer standing at the\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bground view(?:\s+center\s+eye\s+level)?\b",
        "camera at human eye level (about 1.6m), centered in the room",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\baspect\s+ration\b",
        "aspect ratio",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b\d{1,2}\s*:\s*\d{1,2}\s*(aspect\s*ratio)?\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\baspect\s*ratio\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bcinematic\s*(16:9|21:9|2:1)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(equirectangular|panoramic|panorama|skybox|environment\s*map)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\b360(?:°)?\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*,\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" ,;-")

def sanitize_skybox_prompt(text: str) -> str:
    cleaned = _sanitize_prompt_text(text)
    cleaned = re.sub(
        r"\b(equirectangular|panoramic|panorama|skybox|environment\s*map|env\s*map)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\b360(?:°)?\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*,\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" ,;-")

def infer_environment_type(text: str, default: str = "interior") -> str:
    if not text:
        return default
    lower = text.lower()
    interior_keywords = {
        "interior",
        "room",
        "hall",
        "nursery",
        "bedroom",
        "parlor",
        "study",
        "lab",
        "laboratory",
        "staircase",
        "cabin",
        "corridor",
        "attic",
        "kitchen",
        "loft",
        "foyer",
        "library",
        "workshop",
    }
    exterior_keywords = {
        "exterior",
        "outdoor",
        "garden",
        "field",
        "forest",
        "meadow",
        "river",
        "shore",
        "street",
        "village",
        "countryside",
        "mountain",
        "valley",
        "coast",
        "farm",
        "yard",
        "roof",
    }
    if any(word in lower for word in interior_keywords):
        return "interior"
    if any(word in lower for word in exterior_keywords):
        return "exterior"
    return default

def resolve_environment_type(prompt_text: str, environment_type_label: str) -> str:
    if environment_type_label.lower() == "interior":
        return "interior"
    if environment_type_label.lower() == "exterior":
        return "exterior"
    return infer_environment_type(prompt_text, default="interior")

def extract_character_names(char_items: list[dict]) -> list[str]:
    names: list[str] = []
    alias_pattern = re.compile(r"\b(?:aka|a\.k\.a\.|also known as|alias)\b", re.IGNORECASE)
    for item in char_items:
        filename = item.get("filename", "")
        prompt = item.get("prompt", "")
        for candidate in re.split(r"[/,]+", filename):
            candidate = candidate.strip()
            if candidate:
                names.append(candidate)
        if alias_pattern.search(prompt):
            parts = alias_pattern.split(prompt)
            for alias_part in parts[1:]:
                alias_chunk = re.split(r"[.;]", alias_part, maxsplit=1)[0]
                for candidate in re.split(r"[/,]+", alias_chunk):
                    candidate = candidate.strip()
                    if candidate:
                        names.append(candidate)
        match = re.search(r"\b(?:named|called)\s+([A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*){0,3})", prompt)
        if match:
            names.append(match.group(1).strip())
    deduped = []
    seen = set()
    for name in names:
        key = re.sub(r"\s+", " ", name.strip().lower())
        if key and key not in seen:
            seen.add(key)
            deduped.append(name.strip())
    return deduped

def _build_name_pattern(name: str) -> str:
    tokens = re.split(r"[\s\-]+", name.strip())
    tokens = [re.escape(token) for token in tokens if token]
    if not tokens:
        return ""
    return r"\b" + r"(?:[\s\-]+)".join(tokens) + r"\b"

def sanitize_environment_prompt(prompt_text: str, character_name_list: list[str]) -> str:
    if not prompt_text or not character_name_list:
        return prompt_text
    sanitized = prompt_text
    child_words = {"nursery", "bedroom", "playroom", "room", "staircase"}
    workplace_words = {"lab", "laboratory", "study", "office", "workshop"}
    patterns = []
    for name in sorted(character_name_list, key=len, reverse=True):
        pattern = _build_name_pattern(name)
        if pattern:
            patterns.append(pattern)
    for pattern in patterns:
        possessive_regex = re.compile(rf"(?i){pattern}\s*['’]s\b")

        def possessive_replacer(match: re.Match) -> str:
            remainder = match.string[match.end():]
            next_word_match = re.search(r"\s+([A-Za-z][\w-]*)", remainder)
            next_word = next_word_match.group(1).lower() if next_word_match else ""
            if next_word in child_words:
                return "child's"
            if next_word in workplace_words:
                return "private"
            return "a"

        sanitized = possessive_regex.sub(possessive_replacer, sanitized)
        sanitized = re.sub(rf"(?i){pattern}", "", sanitized)
    sanitized = re.sub(r"\s{2,}", " ", sanitized)
    sanitized = re.sub(r"\s+,", ",", sanitized)
    sanitized = re.sub(r",\s*,", ", ", sanitized)
    return sanitized.strip(" ,;-")

def neutralize_character_possessives(text: str) -> str:
    pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s*['’]s\b")
    workplace_words = {"lab", "laboratory", "study", "office", "workshop"}
    child_words = {"nursery", "bedroom", "playroom"}

    def replacer(match: re.Match) -> str:
        remainder = match.string[match.end():]
        next_word_match = re.search(r"\s+([A-Za-z][\w-]*)", remainder)
        next_word = next_word_match.group(1).lower() if next_word_match else ""
        if next_word in workplace_words:
            return "a private"
        if next_word in child_words:
            return "a child's"
        return "a"

    return pattern.sub(replacer, text)

def _split_negative_tokens(text: str) -> list[str]:
    if not text:
        return []
    tokens = []
    for part in re.split(r"[,\n]+", text):
        token = part.strip()
        if token:
            tokens.append(token)
    return tokens

def _build_skybox_init_prompt(scene_prompt: str) -> str:
    if not scene_prompt:
        return ", ".join(SKYBOX_INIT_WRAPPER_TOKENS + SKYBOX_INIT_DETAIL_TOKENS)
    tokens = SKYBOX_INIT_WRAPPER_TOKENS + SKYBOX_INIT_DETAIL_TOKENS
    return _append_prompt_tokens(scene_prompt, tokens)

def _merge_negative_prompts(
    user_negative: str,
    item_negative: str,
    extra_tokens: list[str],
) -> str:
    parts = []
    for entry in (user_negative, item_negative):
        if entry and entry.strip():
            parts.append(entry.strip())
    combined = ", ".join(parts)
    lower_combined = combined.lower()
    extras = [token for token in extra_tokens if token.lower() not in lower_combined]
    if combined and extras:
        combined = f"{combined}, {', '.join(extras)}"
    elif extras:
        combined = ", ".join(extras)
    return combined.strip(" ,")

def _parse_ratio_tuple(ratio_text: str, fallback: tuple[int, int]) -> tuple[int, int]:
    match = re.match(r"\s*(\d+)\s*:\s*(\d+)\s*", ratio_text or "")
    if not match:
        return fallback
    return int(match.group(1)), int(match.group(2))

def normalize_location(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()

def normalize_filename(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()

def _extract_location_line(lines: list[str]) -> tuple[str | None, list[str], bool]:
    location = None
    remaining_lines: list[str] = []
    found = False
    for line in lines:
        match = re.match(r"location\s*:\s*(.+)", line, flags=re.IGNORECASE)
        if match and location is None:
            location = match.group(1).strip()
            found = True
            continue
        remaining_lines.append(line)
    return location, remaining_lines, found

def _resolve_location(location: str | None) -> str | None:
    return location.strip() if location else None

def find_similar_location(
    location_key: str,
    cache: dict[str, dict],
    threshold: float = 0.82,
) -> tuple[str | None, dict | None, float]:
    best_key = None
    best_entry = None
    best_score = 0.0
    for cached_key, entry in cache.items():
        score = difflib.SequenceMatcher(None, location_key, cached_key).ratio()
        if score > best_score:
            best_score = score
            best_key = cached_key
            best_entry = entry
    if best_key and best_score >= threshold:
        return best_key, best_entry, best_score
    return None, None, best_score

def update_location_cache(
    cache: dict[str, dict],
    location_key: str,
    location_label: str,
    image_bytes: bytes | None,
    seed: int | None,
    prompt: str,
) -> None:
    if not location_key:
        return
    cache[location_key] = {
        "location_label": location_label,
        "image_bytes": image_bytes,
        "seed": seed,
        "prompt": prompt,
    }

def cache_mode_allows_reuse(cache_mode: str) -> bool:
    return cache_mode.upper() != "OFF"

def cache_mode_allows_auto_save(cache_mode: str) -> bool:
    return cache_mode.upper() == "ALWAYS"

def cache_mode_requires_approval(cache_mode: str) -> bool:
    return cache_mode.upper() == "APPROVAL ONLY"

def _split_prompt_and_negative(text: str) -> tuple[str, str]:
    match = re.search(r"\bnegative prompt\b\s*:?", text, flags=re.IGNORECASE)
    if not match:
        return _sanitize_prompt_text(text), ""
    prompt_part = text[:match.start()].strip()
    negative_part = text[match.end():].strip()
    return _sanitize_prompt_text(prompt_part), negative_part.strip(" ,;-")

def parse_prompt_blocks(text: str, label_patterns: list[str]) -> list[dict]:
    items: list[dict] = []
    current_name: str | None = None
    current_lines: list[str] = []
    regexes = [re.compile(pattern, re.IGNORECASE) for pattern in label_patterns]
    for line in [line.strip() for line in text.splitlines()]:
        if not line:
            if current_name and current_lines:
                current_lines.append("")
            continue
        match = None
        for regex in regexes:
            match = regex.search(line)
            if match:
                break
        if match:
            if current_name:
                location_line, remaining_lines, location_present = _extract_location_line(current_lines)
                raw_text = " ".join(l for l in remaining_lines if l.strip()).strip()
                prompt_text, negative_text = _split_prompt_and_negative(raw_text)
                location_text = _resolve_location(location_line)
                items.append(
                    {
                        "filename": current_name,
                        "prompt": prompt_text,
                        "negative_prompt": negative_text,
                        "location": location_text,
                        "location_line_present": location_present,
                    }
                )
            current_name = match.group("filename").strip()
            current_lines = []
        elif current_name:
            current_lines.append(line)
    if current_name:
        location_line, remaining_lines, location_present = _extract_location_line(current_lines)
        raw_text = " ".join(l for l in remaining_lines if l.strip()).strip()
        prompt_text, negative_text = _split_prompt_and_negative(raw_text)
        location_text = _resolve_location(location_line)
        items.append(
            {
                "filename": current_name,
                "prompt": prompt_text,
                "negative_prompt": negative_text,
                "location": location_text,
                "location_line_present": location_present,
            }
        )
    return [item for item in items if item["filename"] and item["prompt"]]

def build_seed(base_seed: int, offset: int) -> int | None:
    if base_seed <= 0:
        return None
    return base_seed + offset

def _format_aspect_ratio(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return ""
    gcd = math.gcd(width, height)
    return f"{width // gcd}:{height // gcd}"

def _aspect_ratio_from_bytes(image_bytes: bytes) -> str:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
        return _format_aspect_ratio(width, height)
    except OSError:
        return ""

def _append_prompt_tokens(prompt: str, tokens: list[str]) -> str:
    if not prompt:
        return prompt
    lower_prompt = prompt.lower()
    additions = [token for token in tokens if token.lower() not in lower_prompt]
    if not additions:
        return prompt
    return f"{prompt}, {', '.join(additions)}"

def _apply_skybox_wrapper(prompt: str, include_detail: bool = False) -> str:
    tokens = SKYBOX_PROMPT_WRAPPER_TOKENS[:]
    if include_detail:
        tokens.extend(SKYBOX_INIT_DETAIL_TOKENS)
    return _append_prompt_tokens(prompt, tokens)

def _remove_negative_tokens(negative_text: str, tokens_to_remove: list[str]) -> str:
    if not negative_text:
        return ""
    tokens = [token.strip() for token in negative_text.split(",") if token.strip()]
    remove_set = {token.lower() for token in tokens_to_remove}
    filtered = [token for token in tokens if token.lower() not in remove_set]
    return ", ".join(filtered)

def _adjust_skybox_prompts(
    prompt_text: str,
    negative_text: str,
    minimal_furnishings: bool,
) -> tuple[str, str]:
    adjusted_prompt = prompt_text
    adjusted_negative = negative_text
    if minimal_furnishings:
        adjusted_prompt = _append_prompt_tokens(
            adjusted_prompt,
            ["minimal furnishings", "uncluttered"],
        )
        adjusted_negative = _merge_negative_prompts(adjusted_negative, "", ["no clutter"])
    else:
        adjusted_negative = _remove_negative_tokens(adjusted_negative, ["furniture"])
    return adjusted_prompt, adjusted_negative

def _prepare_skybox_output_bytes(
    image_bytes: bytes,
    apply_crop: bool,
    make_tileable: bool,
    target_size: tuple[int, int] = (2048, 1024),
) -> tuple[bytes, bool]:
    if not apply_crop:
        return image_bytes, False
    processed = prepare_skybox_ready_bytes(
        image_bytes,
        target_size=target_size,
        make_tileable=make_tileable,
    )
    return processed, True

def zip_outputs(
    file_entries: list[tuple[str, bytes]],
    folder: str,
    manifest_entries: list[dict] | None = None,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filename, data in file_entries:
            zip_file.writestr(f"{folder}/{filename}", data)
        if manifest_entries is not None:
            manifest_payload = json.dumps(manifest_entries, indent=2, ensure_ascii=False)
            zip_file.writestr(f"{folder}/manifest.json", manifest_payload)
    buffer.seek(0)
    return buffer.getvalue()

tabs = st.tabs(["Backgrounds", "Characters", "Skyboxes"])

with tabs[0]:
    st.subheader("🖼️ Background Generator (Stability.ai)")
    st.caption("Upload your background prompts DOCX or paste the text directly.")

    bg_doc = st.file_uploader(
        "Background DOCX",
        type=["docx"],
        key="background_docx",
    )
    bg_text_default = read_docx_text(bg_doc) if bg_doc else ""
    bg_text = st.text_area(
        "Background prompt text",
        value=bg_text_default,
        height=200,
        key="background_text",
    )

    bg_items = parse_prompt_blocks(
        bg_text,
        [r"background file name\s*:\s*(?P<filename>.+)"],
    )
    st.write(f"Parsed backgrounds: {len(bg_items)}")
    if bg_items:
        show_bg_negative = any(item.get("negative_prompt") for item in bg_items)
        st.dataframe(
            [
                {
                    "filename": item["filename"],
                    "prompt": item["prompt"][:120],
                    **(
                        {"negative_prompt": item["negative_prompt"][:120]}
                        if show_bg_negative
                        else {}
                    ),
                }
                for item in bg_items
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No background entries parsed yet. Ensure your DOCX includes 'Background file name: <name>'.")

    bg_negative = st.text_input(
        "Negative prompt (optional overrides)",
        "",
        key="background_negative",
    )
    bg_aspect = "21:9"
    st.text_input(
        "Aspect ratio (locked)",
        value=bg_aspect,
        disabled=True,
        key="background_aspect_display",
    )
    bg_seed = st.number_input(
        "Seed (0 = random)",
        min_value=0,
        max_value=2147483647,
        value=0,
        key="background_seed",
    )
    bg_count = st.slider(
        "Images per prompt",
        min_value=1,
        max_value=4,
        value=1,
        key="background_count",
    )

    bg_col_a, bg_col_b = st.columns(2)
    with bg_col_a:
        bg_ref_image = st.file_uploader(
            "Optional reference image (style targeting)",
            type=["png", "jpg", "jpeg"],
            key="background_ref_image",
        )
    with bg_col_b:
        bg_strength = st.slider(
            "Reference strength",
            min_value=0.05,
            max_value=0.95,
            value=0.35,
            step=0.05,
            key="background_strength",
        )

    preview_bg = st.button("Preview first background", type="secondary", key="background_preview")
    generate_bg = st.button("Generate background batch", type="primary", key="background_generate")

    if preview_bg or generate_bg:
        if not bg_items:
            st.error("No background prompts parsed. Please check your DOCX or pasted text.")
            st.stop()
        prompt_policy = PromptPolicy(style_pack_label, era_locale_tags)
        character_names = extract_character_names(
            parse_prompt_blocks(
                st.session_state.get("character_text", ""),
                [
                    r"character name\s*/\s*file name\s*:\s*(?P<filename>.+)",
                    r"character file name\s*:\s*(?P<filename>.+)",
                    r"character filename\s*:\s*(?P<filename>.+)",
                    r"file name\s*:\s*(?P<filename>.+)",
                ],
            )
        )
        bg_cache = st.session_state["bg_cache"]
        cache_mode = cache_mode_label
        reuse_cache_allowed = reuse_location_cache and cache_mode_allows_reuse(cache_mode)
        items_to_run = bg_items[:1] if preview_bg else bg_items
        all_outputs: list[tuple[str, bytes]] = []
        manifest_entries: list[dict] = []
        init_bytes = bg_ref_image.getvalue() if bg_ref_image else None
        with st.status("Generating backgrounds…", expanded=True) as status:
            try:
                for idx, item in enumerate(items_to_run):
                    seed = build_seed(bg_seed, idx)
                    env_type = resolve_environment_type(item.get("prompt", ""), environment_type_label)
                    policy_negative_tokens = _split_negative_tokens(
                        prompt_policy.negative_prompt(env_type)
                    )
                    negative_prompt = _merge_negative_prompts(
                        bg_negative,
                        item.get("negative_prompt", ""),
                        ANTI_LETTERBOX_TOKENS + policy_negative_tokens,
                    )
                    location_key = ""
                    cached_entry = None
                    similarity = 0.0
                    location_line_present = bool(item.get("location_line_present"))
                    if reuse_cache_allowed and location_reuse_enabled and location_line_present:
                        location_key = normalize_location(item.get("location", ""))
                        _, cached_entry, similarity = find_similar_location(
                            location_key,
                            bg_cache,
                        )
                    cached_init_bytes = None
                    cached_seed = None
                    if cached_entry:
                        cached_init_bytes = cached_entry.get("image_bytes")
                        cached_seed = cached_entry.get("seed")
                        status.write(
                            "Similar location detected "
                            f"({similarity:.2f} match to '{cached_entry.get('location_label', 'unknown')}')."
                        )
                    status.write(f"Generating {item['filename']}…")
                    provider_label = "Stability.ai"
                    init_bytes_to_use = init_bytes
                    reused_init = False
                    if reuse_cache_allowed and cached_init_bytes and init_bytes_to_use is None:
                        init_bytes_to_use = cached_init_bytes
                        reused_init = True
                        status.write("Reusing cached reference image for this background.")
                    base_seed_selected = bg_seed > 0
                    seed_to_use = seed
                    reused_seed = False
                    if (
                        reuse_cache_allowed
                        and reuse_cached_seed
                        and base_seed_selected
                        and cached_seed is not None
                    ):
                        seed_to_use = cached_seed
                        reused_seed = True
                        status.write(f"Reusing cached seed {seed_to_use} for this background.")
                    stability_key = stability_api_key()
                    sanitized_prompt = sanitize_environment_prompt(
                        item["prompt"], character_names
                    )
                    prompt_text = _sanitize_prompt_text(sanitized_prompt)
                    prompt_text = prompt_policy.build_prompt(prompt_text, env_type, "background")
                    images = stability_generate_images(
                        prompt=prompt_text,
                        negative_prompt=negative_prompt,
                        seed=seed_to_use,
                        aspect_ratio=bg_aspect,
                        image_count=bg_count,
                        api_key=stability_key,
                        init_image_bytes=init_bytes_to_use,
                        init_strength=bg_strength if init_bytes_to_use else None,
                    )
                    status.write(f"Provider used: {provider_label}")
                    seed_used = seed_to_use if provider_label == "Stability.ai" else None
                    intended_ratio = (2, 1) if panoramic_enabled else (21, 9)
                    processed_images = [
                        postprocess_image_bytes(
                            image_bytes,
                            intended_ratio_tuple=intended_ratio,
                            padding_style=panoramic_style,
                            remove_bars=True,
                        )
                        for image_bytes in images
                    ]
                    if processed_images:
                        if reuse_cache_allowed and location_reuse_enabled and location_line_present:
                            if cache_mode_allows_auto_save(cache_mode):
                                update_location_cache(
                                    bg_cache,
                                    location_key,
                                    item.get("location", ""),
                                    processed_images[0],
                                    seed_used,
                                    prompt_text,
                                )
                            elif cache_mode_requires_approval(cache_mode):
                                st.session_state["pending_cache"]["background"][location_key] = {
                                    "location_label": item.get("location", ""),
                                    "image_bytes": processed_images[0],
                                    "seed": seed_used,
                                    "prompt": prompt_text,
                                }
                    for image_index, image_bytes in enumerate(processed_images, start=1):
                        if bg_count > 1:
                            filename = f"{item['filename']}_{image_index:02d}.png"
                        else:
                            filename = f"{item['filename']}.png"
                        all_outputs.append((filename, image_bytes))
                        manifest_entries.append(
                            {
                                "filename": filename,
                                "provider": provider_label,
                                "prompt": prompt_text,
                                "negative_prompt": negative_prompt,
                                "seed_used": seed_used,
                                "aspect_ratio": _aspect_ratio_from_bytes(image_bytes),
                                "init_strength": bg_strength if init_bytes_to_use else None,
                                "location_based_reuse_enabled": location_reuse_enabled,
                                "location_line_present": location_line_present,
                                "reused_cached_init": reused_init,
                                "reused_cached_seed": reused_seed,
                            }
                        )
                        st.image(
                            image_bytes,
                            caption=f"{filename} • {provider_label}",
                            use_container_width=True,
                        )
                        if (
                            cache_mode_requires_approval(cache_mode)
                            and reuse_cache_allowed
                            and location_reuse_enabled
                            and location_line_present
                            and image_index == 1
                        ):
                            approve_key = f"approve_bg_{location_key}_{item['filename']}"
                            if st.button("Approve as reference", key=approve_key):
                                pending = st.session_state["pending_cache"]["background"].pop(
                                    location_key, None
                                )
                                if pending:
                                    update_location_cache(
                                        bg_cache,
                                        location_key,
                                        pending["location_label"],
                                        pending["image_bytes"],
                                        pending["seed"],
                                        pending["prompt"],
                                    )
                                    st.success("Reference approved and cached.")

                if all_outputs:
                    zip_bytes = zip_outputs(all_outputs, "backgrounds", manifest_entries)
                    st.download_button(
                        "Download backgrounds ZIP",
                        data=zip_bytes,
                        file_name="backgrounds.zip",
                        mime="application/zip",
                    )
                status.update(label="Done", state="complete", expanded=False)
            except requests.exceptions.RequestException as exc:
                status.update(label="Failed", state="error", expanded=True)
                st.error(f"Request failed while generating backgrounds: {exc}")
                st.exception(exc)
            except RuntimeError as exc:
                status.update(label="Failed", state="error", expanded=True)
                st.error(f"Background generation failed: {exc}")
                st.exception(exc)

with tabs[1]:
    st.subheader("🧍 Character Training Images (Stability.ai)")
    st.caption("Upload your character prompts DOCX or paste the text directly.")

    char_doc = st.file_uploader(
        "Character DOCX",
        type=["docx"],
        key="character_docx",
    )
    char_text_default = read_docx_text(char_doc) if char_doc else ""
    char_text = st.text_area(
        "Character prompt text",
        value=char_text_default,
        height=200,
        key="character_text",
    )

    char_items = parse_prompt_blocks(
        char_text,
        [
            r"character name\s*/\s*file name\s*:\s*(?P<filename>.+)",
            r"character file name\s*:\s*(?P<filename>.+)",
            r"character filename\s*:\s*(?P<filename>.+)",
            r"file name\s*:\s*(?P<filename>.+)",
        ],
    )
    st.write(f"Parsed characters: {len(char_items)}")
    if char_items:
        show_char_negative = any(item.get("negative_prompt") for item in char_items)
        st.dataframe(
            [
                {
                    "filename": item["filename"],
                    "prompt": item["prompt"][:120],
                    **(
                        {"negative_prompt": item["negative_prompt"][:120]}
                        if show_char_negative
                        else {}
                    ),
                }
                for item in char_items
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No character entries parsed yet. Ensure your DOCX includes 'Character file name: <name>'.")

    st.markdown("**Variant prompts**")
    full_body_suffix = st.text_input(
        "Full body suffix",
        "neutral full body pose, studio background",
        key="char_full_body_suffix",
    )
    face_front_suffix = st.text_input(
        "Face front suffix",
        "face portrait, front view",
        key="char_face_front_suffix",
    )
    face_45_suffix = st.text_input(
        "Face 45° suffix",
        "face portrait, 45 degree view",
        key="char_face_45_suffix",
    )
    face_profile_suffix = st.text_input(
        "Face profile suffix",
        "face portrait, profile view",
        key="char_face_profile_suffix",
    )

    char_negative = st.text_input(
        "Negative prompt",
        "text, watermark, logo, blurry, low quality, cropped, extra limbs, letterbox, black bars, borders, frame, vignette frame",
        key="char_negative",
    )
    char_aspect = st.selectbox(
        "Aspect ratio",
        ["1:1", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21", "16:9", "21:9"],
        index=0,
        key="char_aspect",
    )
    char_seed = st.number_input(
        "Seed (0 = random)",
        min_value=0,
        max_value=2147483647,
        value=0,
        key="char_seed",
    )
    char_count = st.slider(
        "Images per variant",
        min_value=1,
        max_value=4,
        value=1,
        key="char_count",
    )

    char_col_a, char_col_b = st.columns(2)
    with char_col_a:
        char_ref_image = st.file_uploader(
            "Optional reference image (style targeting)",
            type=["png", "jpg", "jpeg"],
            key="char_ref_image",
        )
    with char_col_b:
        char_strength = st.slider(
            "Reference strength",
            min_value=0.05,
            max_value=0.95,
            value=0.35,
            step=0.05,
            key="char_strength",
        )

    preview_chars = st.button(
        "Preview first character",
        type="secondary",
        key="character_preview",
    )
    generate_chars = st.button(
        "Generate character batch",
        type="primary",
        key="character_generate",
    )

    if preview_chars or generate_chars:
        if not char_items:
            st.error("No character prompts parsed. Please check your DOCX or pasted text.")
            st.stop()
        char_cache = st.session_state["char_cache"]
        init_bytes = char_ref_image.getvalue() if char_ref_image else None
        variants = [
            ("full_body", full_body_suffix),
            ("face_front", face_front_suffix),
            ("face_45", face_45_suffix),
            ("face_profile", face_profile_suffix),
        ]
        items_to_run = char_items[:1] if preview_chars else char_items
        all_outputs: list[tuple[str, bytes]] = []
        manifest_entries: list[dict] = []
        with st.status("Generating character images…", expanded=True) as status:
            try:
                for item_index, item in enumerate(items_to_run):
                    cache_key = normalize_filename(item.get("filename", ""))
                    cached_entry = char_cache.get(cache_key) if reuse_location_cache else None
                    cached_init_bytes = None
                    cached_seed = None
                    if cached_entry:
                        cached_init_bytes = cached_entry.get("image_bytes")
                        cached_seed = cached_entry.get("seed")
                    init_bytes_to_use = init_bytes
                    reused_init = False
                    if reuse_location_cache and cached_init_bytes and init_bytes_to_use is None:
                        init_bytes_to_use = cached_init_bytes
                        reused_init = True
                        status.write("Reusing cached reference image for this character.")
                    for variant_index, (variant_key, variant_suffix) in enumerate(variants):
                        seed_offset = item_index * 100 + variant_index * 10
                        seed = build_seed(char_seed, seed_offset)
                        view_prompt = f"{item['prompt']}, {variant_suffix}"
                        negative_prompt = _merge_negative_prompts(
                            char_negative,
                            item.get("negative_prompt", ""),
                            ANTI_LETTERBOX_TOKENS,
                        )
                        status.write(f"Generating {item['filename']} {variant_key}…")
                        provider_label = "Stability.ai"
                        base_seed_selected = char_seed > 0
                        seed_to_use = seed
                        reused_seed = False
                        if (
                            reuse_location_cache
                            and reuse_cached_seed
                            and base_seed_selected
                            and cached_seed is not None
                        ):
                            seed_to_use = cached_seed
                            reused_seed = True
                            status.write(
                                f"Reusing cached seed {seed_to_use} for {item['filename']} {variant_key}."
                            )
                        stability_key = stability_api_key()
                        images = stability_generate_images(
                            prompt=view_prompt,
                            negative_prompt=negative_prompt,
                            seed=seed_to_use,
                            aspect_ratio=char_aspect,
                            image_count=char_count,
                            api_key=stability_key,
                            init_image_bytes=init_bytes_to_use,
                            init_strength=char_strength if init_bytes_to_use else None,
                        )
                        status.write(f"Provider used: {provider_label}")
                        seed_used = seed_to_use if provider_label == "Stability.ai" else None
                        intended_ratio = (
                            (2, 1)
                            if panoramic_enabled
                            else _parse_ratio_tuple(char_aspect, (1, 1))
                        )
                        processed_images = [
                            postprocess_image_bytes(
                                image_bytes,
                                intended_ratio_tuple=intended_ratio,
                                padding_style=panoramic_style,
                                remove_bars=True,
                            )
                            for image_bytes in images
                        ]
                        if reuse_location_cache and variant_index == 0 and processed_images:
                            update_location_cache(
                                char_cache,
                                cache_key,
                                item.get("filename", ""),
                                processed_images[0],
                                seed_used,
                                item["prompt"],
                            )
                        for image_index, image_bytes in enumerate(processed_images, start=1):
                            if char_count > 1:
                                filename = f"{item['filename']}_{variant_key}_{image_index:02d}.png"
                            else:
                                filename = f"{item['filename']}_{variant_key}.png"
                            all_outputs.append((filename, image_bytes))
                            manifest_entries.append(
                                {
                                    "filename": filename,
                                    "provider": provider_label,
                                    "prompt": view_prompt,
                                    "negative_prompt": negative_prompt,
                                    "seed_used": seed_used,
                                    "aspect_ratio": _aspect_ratio_from_bytes(image_bytes),
                                    "init_strength": char_strength if init_bytes_to_use else None,
                                    "reused_cached_init": reused_init,
                                    "reused_cached_seed": reused_seed,
                                }
                            )
                            st.image(
                                image_bytes,
                                caption=f"{filename} • {provider_label}",
                                use_container_width=True,
                            )
                        if (
                            cache_mode_requires_approval(cache_mode)
                            and reuse_cache_allowed
                            and location_reuse_enabled
                            and location_line_present
                            and image_index == 1
                        ):
                            approve_key = f"approve_bg_{location_key}_{item['filename']}"
                            if st.button("Approve as reference", key=approve_key):
                                pending = st.session_state["pending_cache"]["background"].pop(
                                    location_key, None
                                )
                                if pending:
                                    update_location_cache(
                                        bg_cache,
                                        location_key,
                                        pending["location_label"],
                                        pending["image_bytes"],
                                        pending["seed"],
                                        pending["prompt"],
                                    )
                                    st.success("Reference approved and cached.")

                if all_outputs:
                    zip_bytes = zip_outputs(all_outputs, "characters", manifest_entries)
                    st.download_button(
                        "Download character images ZIP",
                        data=zip_bytes,
                        file_name="characters.zip",
                        mime="application/zip",
                    )

                status.update(label="Done", state="complete", expanded=False)
            except requests.exceptions.RequestException as exc:
                status.update(label="Failed", state="error", expanded=True)
                st.error(f"Request failed while generating characters: {exc}")
                st.exception(exc)
            except RuntimeError as exc:
                status.update(label="Failed", state="error", expanded=True)
                st.error(f"Character generation failed: {exc}")
                st.exception(exc)

with tabs[2]:
    st.subheader("🌐 Skybox Generator (Blockade Labs)")

    blockade_key = blockade_api_key()
    if not blockade_key:
        st.warning("Missing BLOCKADE_API_KEY in Streamlit Secrets. Skybox generation is disabled.")
    else:
        try:
            styles = blockade_get_styles(model_version=3, api_key=blockade_key)
        except (requests.exceptions.RequestException, RuntimeError) as exc:
            st.error(f"Unable to load Blockade styles: {exc}")
            st.stop()
        style_options = {f"{s['name']} (id {s['id']})": s["id"] for s in styles}
        style_label = st.selectbox("Skybox Style (Model 3)", list(style_options.keys()), key="skybox_style")
        style_id = style_options[style_label]

        st.caption("Upload your skybox prompts DOCX or paste the text directly.")
        skybox_doc = st.file_uploader(
            "Skybox DOCX",
            type=["docx"],
            key="skybox_docx",
        )
        skybox_text_default = read_docx_text(skybox_doc) if skybox_doc else ""
        skybox_text = st.text_area(
            "Skybox prompt text",
            value=skybox_text_default,
            height=200,
            key="skybox_text",
        )

        skybox_items = parse_prompt_blocks(
            skybox_text,
            [
                r"skybox name\s*/\s*file name\s*:\s*(?P<filename>.+)",
                r"skybox file name\s*:\s*(?P<filename>.+)",
                r"skybox filename\s*:\s*(?P<filename>.+)",
                r"file name\s*:\s*(?P<filename>.+)",
            ],
        )
        st.write(f"Parsed skyboxes: {len(skybox_items)}")
        if skybox_items:
            show_skybox_negative = any(item.get("negative_prompt") for item in skybox_items)
            st.dataframe(
                [
                    {
                        "filename": item["filename"],
                        "prompt": item["prompt"][:120],
                        **(
                            {"negative_prompt": item["negative_prompt"][:120]}
                            if show_skybox_negative
                            else {}
                        ),
                    }
                    for item in skybox_items
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No skybox entries parsed yet. Ensure your DOCX includes "
                "'Skybox file name: <name>' or 'Skybox filename: <name>'."
            )

        hard_negative_tokens = _split_negative_tokens(skybox_hard_negative_text)
        if skybox_exclude_animals:
            hard_negative_tokens.extend(["animals", "birds", "insects"])

        st.markdown("**Single prompt**")
        prompt = st.text_area(
            "Skybox Prompt",
            "ground view, center eye level, indoors child’s nursery staircase landing, "
            "early 20th-century English cottage, morning light through window, warm cozy shadows, "
            "3D watercolor diorama",
            key="skybox_prompt",
        )
        minimal_furnishings = st.checkbox(
            "Minimal furnishings",
            value=True,
            help="Add minimal furnishings guidance and keep 'no clutter' in negatives.",
            key="skybox_minimal_furnishings",
        )
        apply_skybox_wrapper = st.checkbox(
            "Apply skybox prompt wrapper",
            value=True,
            help="Adds panoramic camera guidance (equirectangular 2:1, centered camera, crisp poles).",
            key="skybox_apply_wrapper",
        )
        negative = st.text_input(
            "Negative text (optional)",
            "",
            key="skybox_negative",
        )
        skybox_aspect_label = st.selectbox(
            "Skybox aspect handling",
            list(SKYBOX_ASPECT_OPTIONS.keys()),
            index=0,
            key="skybox_aspect_handling",
        )
        skybox_aspect_mode = SKYBOX_ASPECT_OPTIONS[skybox_aspect_label]
        skybox_tileable_blend = st.checkbox(
            "Blend skybox seam (tileable)",
            value=True,
            key="skybox_tileable_blend",
        )
        seed = st.number_input(
            "Seed (0 = random)",
            min_value=0,
            max_value=2147483647,
            value=0,
            key="skybox_seed",
        )
        enhance = st.checkbox("Enhance prompt (Blockade)", value=False, key="skybox_enhance")
    
        colA, colB = st.columns(2)
        with colA:
            init_img = st.file_uploader(
                "Optional INIT image (2:1 equirectangular)",
                type=["png", "jpg", "jpeg"],
                key="skybox_init",
            )
            init_strength = st.slider(
                "Init strength (lower = more influence)",
                min_value=0.11,
                max_value=0.90,
                value=0.50,
                step=0.01,
                key="skybox_init_strength",
            )
        with colB:
            control_img = st.file_uploader(
                "Optional CONTROL image (2:1 equirectangular)",
                type=["png", "jpg", "jpeg"],
                key="skybox_control",
            )
            st.caption("Control image preserves structure/perspective more than color. Requires control_model='remix'.")
    
        preview_skybox = st.button(
            "Preview first skybox",
            type="secondary",
            key="skybox_preview",
        )
        generate_skybox_batch = st.button(
            "Generate skybox batch",
            type="primary",
            key="skybox_generate_batch",
        )

        if preview_skybox or generate_skybox_batch:
            if not skybox_items:
                st.error("No skybox prompts parsed. Please check your DOCX or pasted text.")
                st.stop()
            prompt_policy = PromptPolicy(style_pack_label, era_locale_tags)
            character_names = extract_character_names(
                parse_prompt_blocks(
                    st.session_state.get("character_text", ""),
                    [
                        r"character name\s*/\s*file name\s*:\s*(?P<filename>.+)",
                        r"character file name\s*:\s*(?P<filename>.+)",
                        r"character filename\s*:\s*(?P<filename>.+)",
                        r"file name\s*:\s*(?P<filename>.+)",
                    ],
                )
            )
            init_b64 = None
            init_bytes_uploaded = None
            if init_img:
                init_bytes_uploaded = prepare_skybox_ready_bytes(
                    init_img.getvalue(),
                    target_size=(2048, 1024),
                    make_tileable=skybox_tileable_blend,
                )
                init_b64 = base64.b64encode(init_bytes_uploaded).decode("utf-8")
            control_b64 = _b64_of_uploaded_file(control_img) if control_img else None
            skybox_cache = st.session_state["skybox_cache"]
            cache_mode = cache_mode_label
            reuse_cache_allowed = reuse_location_cache and cache_mode_allows_reuse(cache_mode)
            items_to_run = skybox_items[:1] if preview_skybox else skybox_items
            all_outputs: list[tuple[str, bytes]] = []
            manifest_entries: list[dict] = []
            with st.status("Generating skyboxes…", expanded=True) as status:
                try:
                    for idx, item in enumerate(items_to_run):
                        skybox_seed = build_seed(int(seed), idx)
                        env_type = resolve_environment_type(item.get("prompt", ""), environment_type_label)
                        policy_negative_tokens = _split_negative_tokens(
                            prompt_policy.negative_prompt(env_type)
                        )
                        negative_text = _merge_negative_prompts(
                            negative,
                            item.get("negative_prompt", ""),
                            ANTI_LETTERBOX_TOKENS + hard_negative_tokens + policy_negative_tokens,
                        )
                        raw_prompt_text = sanitize_environment_prompt(
                            item["prompt"], character_names
                        )
                        if neutralize_possessives:
                            raw_prompt_text = neutralize_character_possessives(raw_prompt_text)
                        sanitized_prompt = sanitize_skybox_prompt(raw_prompt_text)
                        sanitized_prompt, negative_text = _adjust_skybox_prompts(
                            sanitized_prompt,
                            negative_text,
                            minimal_furnishings,
                        )
                        scene_prompt = prompt_policy.build_prompt(
                            sanitized_prompt,
                            env_type,
                            "skybox",
                        )
                        blockade_prompt = (
                            _apply_skybox_wrapper(scene_prompt) if apply_skybox_wrapper else scene_prompt
                        )
                        location_key = ""
                        cached_entry = None
                        similarity = 0.0
                        location_line_present = bool(item.get("location_line_present"))
                        if reuse_cache_allowed and location_reuse_enabled and location_line_present:
                            location_key = normalize_location(item.get("location", ""))
                            _, cached_entry, similarity = find_similar_location(
                                location_key,
                                skybox_cache,
                            )
                        cached_init_bytes = None
                        cached_seed = None
                        if cached_entry:
                            cached_init_bytes = cached_entry.get("image_bytes")
                            cached_seed = cached_entry.get("seed")
                            status.write(
                                "Similar location detected "
                                f"({similarity:.2f} match to '{cached_entry.get('location_label', 'unknown')}')."
                            )
                        status.write(f"Generating {item['filename']}…")
                        init_bytes_generated = None
                        init_b64_to_use = init_b64
                        reused_init = False
                        init_scene_prompt = scene_prompt
                        init_prompt_debug = prompt_policy.build_skybox_init_prompt(
                            sanitized_prompt,
                            env_type,
                        )
                        if reuse_cache_allowed and cached_init_bytes and init_b64_to_use is None:
                            init_b64_to_use = base64.b64encode(cached_init_bytes).decode("utf-8")
                            reused_init = True
                            status.write("Reusing cached INIT image for this skybox.")
                        if init_b64_to_use is None:
                            stability_key = stability_api_key()
                            init_bytes, init_seed_used = make_skybox_init_from_stability(
                                scene_prompt=init_scene_prompt,
                                negative_prompt=negative_text,
                                base_seed=skybox_seed,
                                api_key=stability_key,
                                candidates=3,
                                make_tileable=skybox_tileable_blend,
                                avoid_blurred_poles=avoid_blurred_poles,
                                pole_detail_weight=pole_detail_weight,
                                neutralize_possessives=False,
                            )
                            init_bytes_generated = init_bytes
                            init_b64_to_use = base64.b64encode(init_bytes).decode("utf-8")
                            status.write(f"Stability init prompt: {init_prompt_debug}")
                            status.write(f"Stability init negative: {negative_text}")
                            status.write(f"Stability init seed used: {init_seed_used}")
                            status.write(f"Stability init cache reused: {reused_init}")
                            status.write("Generated Stability.ai init plate for Skybox.")
                            st.image(
                                init_bytes_generated,
                                caption=f"{item['filename']} • init plate preview",
                                use_container_width=True,
                            )
                        elif reused_init:
                            status.write(f"Stability init prompt (cached): {init_prompt_debug}")
                            status.write(f"Stability init negative (cached): {negative_text}")
                            status.write(f"Stability init seed used (cached): {cached_seed}")
                            status.write("Stability init cache reused: True")
                        else:
                            status.write(f"Stability init prompt (provided): {init_prompt_debug}")
                            status.write(f"Stability init negative (provided): {negative_text}")
                            status.write("Stability init cache reused: False")
                        base_seed_selected = seed > 0
                        seed_to_use = skybox_seed
                        reused_seed = False
                        if (
                            reuse_cache_allowed
                            and reuse_cached_seed
                            and base_seed_selected
                            and cached_seed is not None
                        ):
                            seed_to_use = cached_seed
                            reused_seed = True
                            status.write(f"Reusing cached seed {seed_to_use} for this skybox.")
                        status.write(f"Blockade prompt: {blockade_prompt}")
                        status.write(f"Blockade negative: {negative_text}")
                        status.write(f"Blockade cache reused (init/seed): {reused_init}/{reused_seed}")
                        status.write(f"Blockade seed used: {seed_to_use}")
                        gen = blockade_generate_skybox(
                            prompt=blockade_prompt,
                            style_id=style_id,
                            negative_text=negative_text,
                            seed=seed_to_use,
                            enhance_prompt=enhance,
                            init_image_b64=init_b64_to_use,
                            init_strength=float(init_strength),
                            control_image_b64=control_b64,
                            control_model="remix",
                            api_key=blockade_key,
                        )
                        skybox_oid = gen["obfuscated_id"]
                        done = blockade_poll_generation(skybox_oid)
                        skybox_png = download_url_bytes(done["file_url"])
                        skybox_display, cropped_to_2to1 = _prepare_skybox_output_bytes(
                            skybox_png,
                            apply_crop=skybox_aspect_mode == "crop",
                            make_tileable=skybox_tileable_blend,
                        )
                        cache_bytes = (
                            init_bytes_uploaded
                            or cached_init_bytes
                            or init_bytes_generated
                            or skybox_display
                        )
                        if reuse_cache_allowed and location_reuse_enabled and location_line_present:
                            if cache_mode_allows_auto_save(cache_mode):
                                update_location_cache(
                                    skybox_cache,
                                    location_key,
                                    item.get("location", ""),
                                    cache_bytes,
                                    seed_to_use,
                                    blockade_prompt,
                                )
                            elif cache_mode_requires_approval(cache_mode):
                                st.session_state["pending_cache"]["skybox"][location_key] = {
                                    "location_label": item.get("location", ""),
                                    "image_bytes": cache_bytes,
                                    "seed": seed_to_use,
                                    "prompt": blockade_prompt,
                                }
                        filename = f"{item['filename']}.png"
                        all_outputs.append((filename, skybox_display))
                        manifest_entries.append(
                            {
                                "filename": filename,
                                "provider": "Blockade Labs",
                                "prompt": blockade_prompt,
                                "negative_prompt": negative_text,
                                "seed_used": seed_to_use,
                                "aspect_ratio": _aspect_ratio_from_bytes(skybox_display),
                                "init_strength": float(init_strength),
                                "location_based_reuse_enabled": location_reuse_enabled,
                                "location_line_present": location_line_present,
                                "reused_cached_init": reused_init,
                                "reused_cached_seed": reused_seed,
                                "cropped_to_2to1": cropped_to_2to1,
                                "tileable_blend": bool(cropped_to_2to1 and skybox_tileable_blend),
                            }
                        )
                        st.image(
                            skybox_display,
                            caption=f"{item['filename']} (equirectangular preview)",
                            use_container_width=True,
                        )
                        if (
                            cache_mode_requires_approval(cache_mode)
                            and reuse_cache_allowed
                            and location_reuse_enabled
                            and location_line_present
                        ):
                            approve_key = f"approve_skybox_{location_key}_{item['filename']}"
                            if st.button("Approve as reference", key=approve_key):
                                pending = st.session_state["pending_cache"]["skybox"].pop(
                                    location_key, None
                                )
                                if pending:
                                    update_location_cache(
                                        skybox_cache,
                                        location_key,
                                        pending["location_label"],
                                        pending["image_bytes"],
                                        pending["seed"],
                                        pending["prompt"],
                                    )
                                    st.success("Reference approved and cached.")

                    if all_outputs:
                        zip_bytes = zip_outputs(all_outputs, "skyboxes", manifest_entries)
                        st.download_button(
                            "Download skybox ZIP",
                            data=zip_bytes,
                            file_name="skyboxes.zip",
                            mime="application/zip",
                        )
                    status.update(label="Done", state="complete", expanded=False)
                except requests.exceptions.RequestException as exc:
                    status.update(label="Failed", state="error", expanded=True)
                    st.error(f"Request failed while generating skyboxes: {exc}")
                    st.exception(exc)
                except RuntimeError as exc:
                    status.update(label="Failed", state="error", expanded=True)
                    st.error(f"Skybox generation failed: {exc}")
                    st.exception(exc)
                except TimeoutError as exc:
                    status.update(label="Failed", state="error", expanded=True)
                    st.error(f"Skybox generation timed out: {exc}")
                    st.exception(exc)

        try:
            export_meta = blockade_get_export_types(api_key=blockade_key)
        except (requests.exceptions.RequestException, RuntimeError) as exc:
            export_meta = None
            st.warning(
                "Unable to load Blockade export types. You can still generate a skybox, "
                "or enter export type/resolution IDs manually."
            )
    
        exports_enabled_default = export_meta is not None
        exports_enabled = st.checkbox(
            "Request exports (equirectangular PNG + cubemap ZIP)",
            value=exports_enabled_default,
            help="Disable if the export type metadata is unavailable.",
            key="skybox_exports",
        )
    
        export_png_type_id = None
        export_cubemap_type_id = None
        resolution_id = None
        res_choice = "custom"
    
        if export_meta is not None:
            export_types = _extract_export_types(export_meta)
            export_type_ids = _build_label_index(export_types)
    
            expected_export_labels = {
                "equirectangular-png": "equirectangular-png",
                "cube-map-default-png": "cube-map-default-png",
            }
            missing_export_labels = [
                label for label in expected_export_labels.values() if label not in export_type_ids
            ]
            if missing_export_labels:
                st.error(
                    "Export types response missing expected labels: "
                    f"{', '.join(missing_export_labels)}. Please refresh or check the API response."
                )
                st.stop()
    
            export_png_type_id = export_type_ids[expected_export_labels["equirectangular-png"]]
            export_cubemap_type_id = export_type_ids[expected_export_labels["cube-map-default-png"]]
    
            export_resolutions = _extract_export_resolutions(export_meta, export_types)
            export_resolution_ids = _build_label_index(export_resolutions)
            resolution_labels = ["2K", "4K", "8K", "16K"]
            missing_resolution_labels = [
                label for label in resolution_labels if label not in export_resolution_ids
            ]
            if missing_resolution_labels:
                st.error(
                    "Export resolutions response missing expected labels: "
                    f"{', '.join(missing_resolution_labels)}. Please refresh or check the API response."
                )
                st.stop()
    
            res_choice = st.selectbox("Export resolution", resolution_labels, index=2, key="skybox_resolution")
            resolution_id = export_resolution_ids[res_choice]
        else:
            st.info(
                "Enter export type IDs from the Blockade API docs or dashboard if you want "
                "to request exports."
            )
            export_png_type_id = st.number_input(
                "Equirectangular PNG export type ID",
                min_value=1,
                value=1,
                step=1,
                key="skybox_export_png_id",
            )
            export_cubemap_type_id = st.number_input(
                "Cubemap ZIP export type ID",
                min_value=1,
                value=2,
                step=1,
                key="skybox_export_cube_id",
            )
            res_choice = st.text_input(
                "Export resolution label (for filenames)",
                value="custom",
                key="skybox_res_label",
            )
            resolution_id = st.number_input(
                "Export resolution ID",
                min_value=1,
                value=1,
                step=1,
                key="skybox_resolution_id",
            )
    
        gen_btn = st.button("Generate Skybox", type="primary", use_container_width=True, key="skybox_generate")

        if gen_btn:
            prompt_policy = PromptPolicy(style_pack_label, era_locale_tags)
            character_names = extract_character_names(
                parse_prompt_blocks(
                    st.session_state.get("character_text", ""),
                    [
                        r"character name\s*/\s*file name\s*:\s*(?P<filename>.+)",
                        r"character file name\s*:\s*(?P<filename>.+)",
                        r"character filename\s*:\s*(?P<filename>.+)",
                        r"file name\s*:\s*(?P<filename>.+)",
                    ],
                )
            )
            init_b64 = None
            init_bytes_uploaded = None
            if init_img:
                init_bytes_uploaded = prepare_skybox_ready_bytes(
                    init_img.getvalue(),
                    target_size=(2048, 1024),
                    make_tileable=skybox_tileable_blend,
                )
                init_b64 = base64.b64encode(init_bytes_uploaded).decode("utf-8")
            control_b64 = _b64_of_uploaded_file(control_img) if control_img else None
            skybox_seed = None if seed == 0 else int(seed)
            prompt_lines = [line.strip() for line in prompt.splitlines() if line.strip()]
            location_line, remaining_lines, location_present = _extract_location_line(prompt_lines)
            prompt_text, prompt_negative = _split_prompt_and_negative(" ".join(remaining_lines).strip())
            env_type = resolve_environment_type(prompt_text, environment_type_label)
            policy_negative_tokens = _split_negative_tokens(
                prompt_policy.negative_prompt(env_type)
            )
            negative_text = _merge_negative_prompts(
                negative,
                prompt_negative,
                ANTI_LETTERBOX_TOKENS + hard_negative_tokens + policy_negative_tokens,
            )
            raw_prompt_text = sanitize_environment_prompt(prompt_text, character_names)
            if neutralize_possessives:
                raw_prompt_text = neutralize_character_possessives(raw_prompt_text)
            sanitized_prompt = sanitize_skybox_prompt(raw_prompt_text)
            sanitized_prompt, negative_text = _adjust_skybox_prompts(
                sanitized_prompt,
                negative_text,
                minimal_furnishings,
            )
            scene_prompt = prompt_policy.build_prompt(
                sanitized_prompt,
                env_type,
                "skybox",
            )
            blockade_prompt = (
                _apply_skybox_wrapper(scene_prompt) if apply_skybox_wrapper else scene_prompt
            )
            location_text = _resolve_location(location_line)
            location_key = ""
            skybox_cache = st.session_state["skybox_cache"]
            cached_entry = None
            similarity = 0.0
            cache_mode = cache_mode_label
            reuse_cache_allowed = reuse_location_cache and cache_mode_allows_reuse(cache_mode)
            if reuse_cache_allowed and location_reuse_enabled and location_present and location_text:
                location_key = normalize_location(location_text)
                _, cached_entry, similarity = find_similar_location(
                    location_key,
                    skybox_cache,
                )
            cached_init_bytes = None
            cached_seed = None
            if cached_entry:
                cached_init_bytes = cached_entry.get("image_bytes")
                cached_seed = cached_entry.get("seed")

            with st.status("Generating skybox…", expanded=True) as status:
                try:
                    if cached_entry:
                        status.write(
                            "Similar location detected "
                            f"({similarity:.2f} match to '{cached_entry.get('location_label', 'unknown')}')."
                        )
                    init_bytes_generated = None
                    init_b64_to_use = init_b64
                    reused_init = False
                    init_scene_prompt = scene_prompt
                    init_prompt_debug = prompt_policy.build_skybox_init_prompt(
                        sanitized_prompt,
                        env_type,
                    )
                    if reuse_cache_allowed and cached_init_bytes and init_b64_to_use is None:
                        init_b64_to_use = base64.b64encode(cached_init_bytes).decode("utf-8")
                        reused_init = True
                        status.write("Reusing cached INIT image for this skybox.")
                    if init_b64_to_use is None:
                        stability_key = stability_api_key()
                        init_bytes, init_seed_used = make_skybox_init_from_stability(
                            scene_prompt=init_scene_prompt,
                            negative_prompt=negative_text,
                            base_seed=skybox_seed,
                            api_key=stability_key,
                            candidates=3,
                            make_tileable=skybox_tileable_blend,
                            avoid_blurred_poles=avoid_blurred_poles,
                            pole_detail_weight=pole_detail_weight,
                            neutralize_possessives=False,
                        )
                        init_bytes_generated = init_bytes
                        init_b64_to_use = base64.b64encode(init_bytes).decode("utf-8")
                        status.write(f"Stability init prompt: {init_prompt_debug}")
                        status.write(f"Stability init negative: {negative_text}")
                        status.write(f"Stability init seed used: {init_seed_used}")
                        status.write(f"Stability init cache reused: {reused_init}")
                        status.write("Generated Stability.ai init plate for Skybox.")
                        st.image(
                            init_bytes_generated,
                            caption="Skybox init plate preview",
                            use_container_width=True,
                        )
                    elif reused_init:
                        status.write(f"Stability init prompt (cached): {init_prompt_debug}")
                        status.write(f"Stability init negative (cached): {negative_text}")
                        status.write(f"Stability init seed used (cached): {cached_seed}")
                        status.write("Stability init cache reused: True")
                    else:
                        status.write(f"Stability init prompt (provided): {init_prompt_debug}")
                        status.write(f"Stability init negative (provided): {negative_text}")
                        status.write("Stability init cache reused: False")
                    base_seed_selected = seed > 0
                    seed_to_use = skybox_seed
                    reused_seed = False
                    if (
                        reuse_cache_allowed
                        and reuse_cached_seed
                        and base_seed_selected
                        and cached_seed is not None
                    ):
                        seed_to_use = cached_seed
                        reused_seed = True
                        status.write(f"Reusing cached seed {seed_to_use} for this skybox.")
                    status.write(f"Blockade prompt: {blockade_prompt}")
                    status.write(f"Blockade negative: {negative_text}")
                    status.write(f"Blockade cache reused (init/seed): {reused_init}/{reused_seed}")
                    status.write(f"Blockade seed used: {seed_to_use}")
                    gen = blockade_generate_skybox(
                        prompt=blockade_prompt,
                        style_id=style_id,
                        negative_text=negative_text,
                        seed=seed_to_use,
                        enhance_prompt=enhance,
                        init_image_b64=init_b64_to_use,
                        init_strength=float(init_strength),
                        control_image_b64=control_b64,
                        control_model="remix",
                        api_key=blockade_key,
                    )
                    skybox_oid = gen["obfuscated_id"]
                    status.write(f"Generation started. obfuscated_id: {skybox_oid}")
    
                    done = blockade_poll_generation(skybox_oid)
                    status.write("Skybox complete. Fetching base image…")
                    skybox_png = download_url_bytes(done["file_url"])
                    skybox_display, cropped_to_2to1 = _prepare_skybox_output_bytes(
                        skybox_png,
                        apply_crop=skybox_aspect_mode == "crop",
                        make_tileable=skybox_tileable_blend,
                    )

                    st.image(
                        skybox_display,
                        caption="Skybox (equirectangular preview)",
                        use_container_width=True,
                    )
                    cache_bytes = (
                        init_bytes_uploaded
                        or cached_init_bytes
                        or init_bytes_generated
                        or skybox_display
                    )
                    if reuse_cache_allowed and location_reuse_enabled and location_present and location_key:
                        if cache_mode_allows_auto_save(cache_mode):
                            update_location_cache(
                                skybox_cache,
                                location_key,
                                location_text,
                                cache_bytes,
                                seed_to_use,
                                blockade_prompt,
                            )
                        elif cache_mode_requires_approval(cache_mode):
                            st.session_state["pending_cache"]["skybox"][location_key] = {
                                "location_label": location_text,
                                "image_bytes": cache_bytes,
                                "seed": seed_to_use,
                                "prompt": blockade_prompt,
                            }
                    st.download_button(
                        "Download equirectangular (base)",
                        data=skybox_display,
                        file_name="skybox_equirectangular_base.png",
                        mime="image/png",
                    )
                    if (
                        cache_mode_requires_approval(cache_mode)
                        and reuse_cache_allowed
                        and location_reuse_enabled
                        and location_present
                        and location_key
                    ):
                        approve_key = f"approve_skybox_single_{location_key}"
                        if st.button("Approve as reference", key=approve_key):
                            pending = st.session_state["pending_cache"]["skybox"].pop(
                                location_key, None
                            )
                            if pending:
                                update_location_cache(
                                    skybox_cache,
                                    location_key,
                                    pending["location_label"],
                                    pending["image_bytes"],
                                    pending["seed"],
                                    pending["prompt"],
                                )
                                st.success("Reference approved and cached.")
    
                    if exports_enabled:
                        if export_png_type_id is None or export_cubemap_type_id is None or resolution_id is None:
                            raise RuntimeError(
                                "Export type metadata is unavailable. Disable exports or provide manual IDs."
                            )
                        status.write("Requesting exports…")
                        exp_png = blockade_request_export(
                            skybox_oid,
                            type_id=export_png_type_id,
                            resolution_id=resolution_id,
                        )
                        exp_cube = blockade_request_export(
                            skybox_oid,
                            type_id=export_cubemap_type_id,
                            resolution_id=resolution_id,
                        )
    
                        exp_png_done = blockade_poll_export(exp_png["id"])
                        exp_cube_done = blockade_poll_export(exp_cube["id"])

                        png_bytes = download_url_bytes(exp_png_done["file_url"])
                        cube_bytes = download_url_bytes(exp_cube_done["file_url"])
                        png_display, _ = _prepare_skybox_output_bytes(
                            png_bytes,
                            apply_crop=skybox_aspect_mode == "crop",
                            make_tileable=skybox_tileable_blend,
                        )

                        st.download_button(
                            "Download equirectangular PNG (export)",
                            data=png_display,
                            file_name=f"skybox_{res_choice}_equirectangular.png",
                            mime="image/png",
                        )
                        st.download_button(
                            "Download cubemap (export)",
                            data=cube_bytes,
                            file_name=f"skybox_{res_choice}_cubemap.zip",
                            mime="application/zip",
                        )
    
                    status.update(label="Done", state="complete", expanded=False)
                except requests.exceptions.RequestException as exc:
                    status.update(label="Failed", state="error", expanded=True)
                    st.error(f"Request failed while generating the skybox: {exc}")
                    st.exception(exc)
                except RuntimeError as exc:
                    status.update(label="Failed", state="error", expanded=True)
                    st.error(f"Skybox generation failed: {exc}")
                    st.exception(exc)
                except TimeoutError as exc:
                    status.update(label="Failed", state="error", expanded=True)
                    st.error(f"Skybox generation timed out: {exc}")
                    st.exception(exc)
