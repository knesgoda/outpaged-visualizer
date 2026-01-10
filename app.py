import base64
import difflib
import io
import os
import re
import time
import zipfile

import requests
import streamlit as st
from docx import Document
from PIL import Image, ImageFilter, ImageOps

BLOCKADE_BASE = "https://backend.blockadelabs.com/api/v1"
STABILITY_CORE_URL = "https://api.stability.ai/v2beta/stable-image/generate/core"
STABILITY_ULTRA_URL = "https://api.stability.ai/v2beta/stable-image/generate/ultra"
STABILITY_MAX_UPLOAD_BYTES = 9 * 1024 * 1024

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

if "bg_cache" not in st.session_state:
    st.session_state["bg_cache"] = {}
if "char_cache" not in st.session_state:
    st.session_state["char_cache"] = {}
if "skybox_cache" not in st.session_state:
    st.session_state["skybox_cache"] = {}

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
) -> tuple[bytes, int | None]:
    wrapper_prefix = (
        "wide environment plate, horizon centered, no close foreground, evenly distributed detail, "
        "panoramic environment map look, seamless left and right edges, calm camera, "
    )
    wrapper_suffix = " no people, no text"
    prompt = f"{wrapper_prefix}{scene_prompt},{wrapper_suffix}"

    source_aspect = "16:9"

    best_bytes = None
    best_seed = None
    best_score = None

    seed0 = None if (base_seed is None or base_seed <= 0) else int(base_seed)

    for i in range(candidates):
        seed_i = None if seed0 is None else seed0 + i
        raw = stability_generate_ultra(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed_i,
            aspect_ratio=source_aspect,
            api_key=api_key,
        )
        fixed = prepare_skybox_ready_bytes(raw, target_size=(2048, 1024), make_tileable=True)
        score = edge_seam_score(fixed)

        if best_score is None or score < best_score:
            best_score = score
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
    import numpy as np

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
    import numpy as np

    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        arr = np.array(image).astype("float32")
        strip = 16
        left = arr[:, :strip, :]
        right = arr[:, -strip:, :]
        return float(((left - right) ** 2).mean())

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

def strip_letterbox(
    image: Image.Image,
    threshold: int = 8,
    min_bar_px: int = 24,
) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    if height < min_bar_px * 2:
        return image

    pixels = rgb.load()

    def row_is_dark(row_y: int) -> bool:
        total = 0
        for x in range(width):
            r, g, b = pixels[x, row_y]
            total += r + g + b
        mean = total / (width * 3)
        return mean <= threshold

    top = 0
    while top < height and row_is_dark(top):
        top += 1
    bottom = height - 1
    while bottom >= 0 and row_is_dark(bottom):
        bottom -= 1

    crop_top = top if top >= min_bar_px else 0
    crop_bottom = (height - 1 - bottom) if (height - 1 - bottom) >= min_bar_px else 0

    if crop_top == 0 and crop_bottom == 0:
        return image

    new_bottom = height - crop_bottom
    if crop_top >= new_bottom:
        return image
    return image.crop((0, crop_top, width, new_bottom))

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
            image = strip_letterbox(image)
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
    return postprocess_to_2to1(image_bytes, enabled, mode, padding_style)

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
    cleaned = re.sub(r"\s*,\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" ,;-")

def normalize_location(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()

def normalize_filename(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()

def _extract_location_line(lines: list[str]) -> tuple[str | None, list[str]]:
    location = None
    remaining_lines: list[str] = []
    for line in lines:
        match = re.match(r"location\s*:\s*(.+)", line, flags=re.IGNORECASE)
        if match and location is None:
            location = match.group(1).strip()
            continue
        remaining_lines.append(line)
    return location, remaining_lines

def _fallback_location(prompt_text: str, max_len: int = 120) -> str:
    trimmed = prompt_text.strip()
    if not trimmed:
        return "unknown location"
    return trimmed[:max_len].strip()

def _resolve_location(location: str | None, prompt_text: str) -> str:
    return location or _fallback_location(prompt_text)

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
                location_line, remaining_lines = _extract_location_line(current_lines)
                raw_text = " ".join(l for l in remaining_lines if l.strip()).strip()
                prompt_text, negative_text = _split_prompt_and_negative(raw_text)
                location_text = _resolve_location(location_line, prompt_text)
                items.append(
                    {
                        "filename": current_name,
                        "prompt": prompt_text,
                        "negative_prompt": negative_text,
                        "location": location_text,
                    }
                )
            current_name = match.group("filename").strip()
            current_lines = []
        elif current_name:
            current_lines.append(line)
    if current_name:
        location_line, remaining_lines = _extract_location_line(current_lines)
        raw_text = " ".join(l for l in remaining_lines if l.strip()).strip()
        prompt_text, negative_text = _split_prompt_and_negative(raw_text)
        location_text = _resolve_location(location_line, prompt_text)
        items.append(
            {
                "filename": current_name,
                "prompt": prompt_text,
                "negative_prompt": negative_text,
                "location": location_text,
            }
        )
    return [item for item in items if item["filename"] and item["prompt"]]

def build_seed(base_seed: int, offset: int) -> int | None:
    if base_seed <= 0:
        return None
    return base_seed + offset

def zip_outputs(file_entries: list[tuple[str, bytes]], folder: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filename, data in file_entries:
            zip_file.writestr(f"{folder}/{filename}", data)
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
        "Negative prompt",
        "text, watermark, logo, low quality, blurry, letterbox, black bars, borders, frame, vignette frame",
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
        bg_cache = st.session_state["bg_cache"]
        items_to_run = bg_items[:1] if preview_bg else bg_items
        all_outputs: list[tuple[str, bytes]] = []
        init_bytes = bg_ref_image.getvalue() if bg_ref_image else None
        with st.status("Generating backgrounds…", expanded=True) as status:
            try:
                for idx, item in enumerate(items_to_run):
                    seed = build_seed(bg_seed, idx)
                    negative_prompt = item.get("negative_prompt") or bg_negative.strip()
                    location_key = normalize_location(item.get("location", ""))
                    cached_key, cached_entry, similarity = find_similar_location(
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
                    if cached_init_bytes and init_bytes_to_use is None:
                        init_bytes_to_use = cached_init_bytes
                        status.write("Reusing cached reference image for this background.")
                    base_seed_selected = bg_seed > 0
                    seed_to_use = seed
                    if base_seed_selected and cached_seed is not None:
                        seed_to_use = cached_seed
                        status.write(f"Reusing cached seed {seed_to_use} for this background.")
                    stability_key = stability_api_key()
                    prompt_text = _sanitize_prompt_text(item["prompt"])
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
                    if images:
                        update_location_cache(
                            bg_cache,
                            location_key,
                            item.get("location", ""),
                            images[0],
                            seed_used,
                            prompt_text,
                        )
                    for image_index, image_bytes in enumerate(images, start=1):
                        if bg_count > 1:
                            filename = f"{item['filename']}_{image_index:02d}.png"
                        else:
                            filename = f"{item['filename']}.png"
                        output_bytes = _apply_panoramic_conversion(
                            image_bytes,
                            panoramic_enabled,
                            panoramic_mode,
                            panoramic_style,
                        )
                        all_outputs.append((filename, output_bytes))
                        st.image(
                            output_bytes,
                            caption=f"{filename} • {provider_label}",
                            use_container_width=True,
                        )

                if all_outputs:
                    zip_bytes = zip_outputs(all_outputs, "backgrounds")
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
        with st.status("Generating character images…", expanded=True) as status:
            try:
                for item_index, item in enumerate(items_to_run):
                    cache_key = normalize_filename(item.get("filename", ""))
                    cached_entry = char_cache.get(cache_key)
                    cached_init_bytes = None
                    cached_seed = None
                    if cached_entry:
                        cached_init_bytes = cached_entry.get("image_bytes")
                        cached_seed = cached_entry.get("seed")
                    init_bytes_to_use = init_bytes
                    if cached_init_bytes and init_bytes_to_use is None:
                        init_bytes_to_use = cached_init_bytes
                        status.write("Reusing cached reference image for this character.")
                    for variant_index, (variant_key, variant_suffix) in enumerate(variants):
                        seed_offset = item_index * 100 + variant_index * 10
                        seed = build_seed(char_seed, seed_offset)
                        view_prompt = f"{item['prompt']}, {variant_suffix}"
                        negative_prompt = item.get("negative_prompt") or char_negative.strip()
                        status.write(f"Generating {item['filename']} {variant_key}…")
                        provider_label = "Stability.ai"
                        base_seed_selected = char_seed > 0
                        seed_to_use = seed
                        if base_seed_selected and cached_seed is not None:
                            seed_to_use = cached_seed
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
                        if variant_index == 0 and images:
                            update_location_cache(
                                char_cache,
                                cache_key,
                                item.get("filename", ""),
                                images[0],
                                seed_used,
                                item["prompt"],
                            )
                        for image_index, image_bytes in enumerate(images, start=1):
                            if char_count > 1:
                                filename = f"{item['filename']}_{variant_key}_{image_index:02d}.png"
                            else:
                                filename = f"{item['filename']}_{variant_key}.png"
                            output_bytes = _apply_panoramic_conversion(
                                image_bytes,
                                panoramic_enabled,
                                panoramic_mode,
                                panoramic_style,
                            )
                            all_outputs.append((filename, output_bytes))
                            st.image(
                                output_bytes,
                                caption=f"{filename} • {provider_label}",
                                use_container_width=True,
                            )

                if all_outputs:
                    zip_bytes = zip_outputs(all_outputs, "characters")
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

        st.markdown("**Single prompt**")
        prompt = st.text_area(
            "Skybox Prompt",
            "Laputan observatory walkway under cold sky light, chalk-marked geometric diagrams, star charts, brass instruments, vivid chalk pastel look",
            key="skybox_prompt",
        )
        negative = st.text_input(
            "Negative text (optional)",
            "people, text, watermark, letterbox, black bars, borders, frame, vignette frame",
            key="skybox_negative",
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
            init_b64 = None
            init_bytes_uploaded = None
            if init_img:
                init_bytes_uploaded = prepare_skybox_ready_bytes(
                    init_img.getvalue(),
                    target_size=(2048, 1024),
                    make_tileable=True,
                )
                init_b64 = base64.b64encode(init_bytes_uploaded).decode("utf-8")
            control_b64 = _b64_of_uploaded_file(control_img) if control_img else None
            skybox_cache = st.session_state["skybox_cache"]
            items_to_run = skybox_items[:1] if preview_skybox else skybox_items
            all_outputs: list[tuple[str, bytes]] = []
            with st.status("Generating skyboxes…", expanded=True) as status:
                try:
                    for idx, item in enumerate(items_to_run):
                        skybox_seed = build_seed(int(seed), idx)
                        negative_text = item.get("negative_prompt") or negative
                        location_key = normalize_location(item.get("location", ""))
                        cached_key, cached_entry, similarity = find_similar_location(
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
                        if cached_init_bytes and init_b64_to_use is None:
                            init_b64_to_use = base64.b64encode(cached_init_bytes).decode("utf-8")
                            status.write("Reusing cached INIT image for this skybox.")
                        if init_b64_to_use is None:
                            stability_key = stability_api_key()
                            init_bytes, init_seed_used = make_skybox_init_from_stability(
                                scene_prompt=item["prompt"],
                                negative_prompt=(item.get("negative_prompt") or negative),
                                base_seed=skybox_seed,
                                api_key=stability_key,
                                candidates=3,
                            )
                            init_bytes_generated = init_bytes
                            init_b64_to_use = base64.b64encode(init_bytes).decode("utf-8")
                            update_location_cache(
                                skybox_cache,
                                location_key,
                                item.get("location", ""),
                                init_bytes,
                                init_seed_used,
                                item["prompt"],
                            )
                            status.write("Generated Stability.ai init plate for Skybox.")
                        base_seed_selected = seed > 0
                        seed_to_use = skybox_seed
                        if base_seed_selected and cached_seed is not None:
                            seed_to_use = cached_seed
                            status.write(f"Reusing cached seed {seed_to_use} for this skybox.")
                        gen = blockade_generate_skybox(
                            prompt=item["prompt"],
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
                        cache_bytes = (
                            init_bytes_uploaded
                            or cached_init_bytes
                            or init_bytes_generated
                            or skybox_png
                        )
                        update_location_cache(
                            skybox_cache,
                            location_key,
                            item.get("location", ""),
                            cache_bytes,
                            seed_to_use,
                            item["prompt"],
                        )
                        filename = f"{item['filename']}.png"
                        output_bytes = _apply_panoramic_conversion(
                            skybox_png,
                            panoramic_enabled,
                            panoramic_mode,
                            panoramic_style,
                        )
                        all_outputs.append((filename, output_bytes))
                        st.image(
                            output_bytes,
                            caption=f"{item['filename']} (equirectangular preview)",
                            use_container_width=True,
                        )

                    if all_outputs:
                        zip_bytes = zip_outputs(all_outputs, "skyboxes")
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
            init_b64 = None
            init_bytes_uploaded = None
            if init_img:
                init_bytes_uploaded = prepare_skybox_ready_bytes(
                    init_img.getvalue(),
                    target_size=(2048, 1024),
                    make_tileable=True,
                )
                init_b64 = base64.b64encode(init_bytes_uploaded).decode("utf-8")
            control_b64 = _b64_of_uploaded_file(control_img) if control_img else None
            skybox_seed = None if seed == 0 else int(seed)
            prompt_lines = [line.strip() for line in prompt.splitlines() if line.strip()]
            location_line, remaining_lines = _extract_location_line(prompt_lines)
            prompt_text, prompt_negative = _split_prompt_and_negative(" ".join(remaining_lines).strip())
            negative_text = prompt_negative or negative
            location_text = _resolve_location(location_line, prompt_text)
            location_key = normalize_location(location_text)
            skybox_cache = st.session_state["skybox_cache"]
            cached_key, cached_entry, similarity = find_similar_location(
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
                    if cached_init_bytes and init_b64_to_use is None:
                        init_b64_to_use = base64.b64encode(cached_init_bytes).decode("utf-8")
                        status.write("Reusing cached INIT image for this skybox.")
                    if init_b64_to_use is None:
                        stability_key = stability_api_key()
                        init_bytes, init_seed_used = make_skybox_init_from_stability(
                            scene_prompt=prompt_text,
                            negative_prompt=negative_text,
                            base_seed=skybox_seed,
                            api_key=stability_key,
                            candidates=3,
                        )
                        init_bytes_generated = init_bytes
                        init_b64_to_use = base64.b64encode(init_bytes).decode("utf-8")
                        update_location_cache(
                            skybox_cache,
                            location_key,
                            location_text,
                            init_bytes,
                            init_seed_used,
                            prompt_text,
                        )
                        status.write("Generated Stability.ai init plate for Skybox.")
                    base_seed_selected = seed > 0
                    seed_to_use = skybox_seed
                    if base_seed_selected and cached_seed is not None:
                        seed_to_use = cached_seed
                        status.write(f"Reusing cached seed {seed_to_use} for this skybox.")
                    gen = blockade_generate_skybox(
                        prompt=prompt_text,
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
                    skybox_display = _apply_panoramic_conversion(
                        skybox_png,
                        panoramic_enabled,
                        panoramic_mode,
                        panoramic_style,
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
                        or skybox_png
                    )
                    update_location_cache(
                        skybox_cache,
                        location_key,
                        location_text,
                        cache_bytes,
                        seed_to_use,
                        prompt_text,
                    )
                    st.download_button(
                        "Download equirectangular (base)",
                        data=skybox_display,
                        file_name="skybox_equirectangular_base.png",
                        mime="image/png",
                    )
    
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
                        png_display = _apply_panoramic_conversion(
                            png_bytes,
                            panoramic_enabled,
                            panoramic_mode,
                            panoramic_style,
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
