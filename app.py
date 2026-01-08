import os
import io
import re
import json
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image

# Stability gRPC SDK
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

# DOCX parsing
from docx import Document

# Google Drive API (service account approach)
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload


# ============================================================
# Data models
# ============================================================
@dataclass
class BackgroundItem:
    file_name: str  # ch01bg01
    prompt: str

@dataclass
class CanonCharacter:
    name: str
    canon_file_name: str  # canon_lemuel_gulliver
    prompt: str


# ============================================================
# Parsing helpers for your exact format
# ============================================================
_BG_FILE_RE = re.compile(r"(?i)^\s*Background\s*file\s*name\s*:\s*([A-Za-z0-9_\-]+)\s*$")
_SCENE_RE = re.compile(r"(?i)^\s*Scene\s*\d+")
_CANON_RE = re.compile(r"(?i)^\s*Canonical\s+Master\s+Prompts\s*$")

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def read_docx_to_text(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    paras = [p.text for p in doc.paragraphs if p.text is not None]
    return "\n".join(paras)

def parse_outpaged_text(raw_text: str) -> Tuple[List[BackgroundItem], Dict[str, CanonCharacter]]:
    """
    Parses:
      - Background blocks keyed by "Background file name: ch##bg##"
      - Canonical Master Prompts section: Name line then prompt line(s)
    Returns:
      backgrounds: list
      canon_chars: dict keyed by display name
    """
    lines = [ln.rstrip() for ln in raw_text.splitlines()]
    n = len(lines)

    backgrounds: List[BackgroundItem] = []
    canon_chars: Dict[str, CanonCharacter] = {}

    i = 0

    # 1) Backgrounds
    while i < n:
        if _CANON_RE.match(lines[i]):
            break

        m = _BG_FILE_RE.match(lines[i])
        if m:
            file_name = m.group(1).strip()
            i += 1

            prompt_lines: List[str] = []
            while i < n:
                if _CANON_RE.match(lines[i]) or _BG_FILE_RE.match(lines[i]) or _SCENE_RE.match(lines[i]):
                    break
                if lines[i].strip() != "":
                    prompt_lines.append(lines[i].strip())
                i += 1

            prompt = " ".join(prompt_lines).strip()
            if prompt:
                backgrounds.append(BackgroundItem(file_name=file_name, prompt=prompt))
            continue

        i += 1

    # 2) Canonical master prompts
    while i < n and not _CANON_RE.match(lines[i]):
        i += 1

    if i < n and _CANON_RE.match(lines[i]):
        i += 1

        # Skip helper sentence lines (optional)
        while i < n and lines[i].strip():
            if lines[i].strip().lower().startswith("use these master prompts"):
                i += 1
            else:
                break

        # Parse blocks: name line then prompt line(s)
        while i < n:
            name = lines[i].strip()
            if not name:
                i += 1
                continue

            if _SCENE_RE.match(name) or _BG_FILE_RE.match(name) or _CANON_RE.match(name):
                break

            i += 1
            prompt_lines: List[str] = []

            while i < n:
                ln = lines[i].strip()

                if not ln:
                    i += 1
                    continue

                # Heuristic: new character name line tends to be short and often contains no commas.
                if "," not in ln and len(ln) <= 70:
                    break

                prompt_lines.append(ln)
                i += 1

            prompt = " ".join(prompt_lines).strip()
            if prompt:
                canon_file = f"canon_{slugify(name)}"
                canon_chars[name] = CanonCharacter(
                    name=name,
                    canon_file_name=canon_file,
                    prompt=prompt
                )

    return backgrounds, canon_chars


# ============================================================
# Character training pack expansion
# ============================================================
TRAIN_VARIANTS = [
    ("neutral_full_body", "Neutral full-body pose, standing straight, arms relaxed, neutral expression."),
    ("face_front", "Tight head-and-shoulders portrait, straight-on front view, neutral expression."),
    ("face_45", "Tight head-and-shoulders portrait, 45-degree angle view, neutral expression."),
    ("face_profile", "Tight head-and-shoulders portrait, full profile view, neutral expression."),
]

FACE_VARIANTS = {"face_front", "face_45", "face_profile"}

def build_character_variant_prompt(canon_prompt: str, variant_suffix: str) -> str:
    addon_map = {k: v for k, v in TRAIN_VARIANTS}
    addon = addon_map.get(variant_suffix, "")
    canon_prompt = canon_prompt.strip()
    if addon:
        return f"{canon_prompt} {addon}"
    return canon_prompt


# ============================================================
# Character Pack Template framing (crop + resize back)
# ============================================================
@dataclass
class FramingTemplate:
    enabled: bool
    apply_to_face_variants_only: bool
    x0: float  # 0.0 to 1.0
    y0: float
    x1: float
    y1: float
    resize_back_to_target: bool

def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))

def apply_framing(
    img: Image.Image,
    variant_suffix: str,
    tpl: FramingTemplate,
    target_w: int,
    target_h: int,
) -> Image.Image:
    """
    For training consistency:
    - Crops a normalized region (x0,y0,x1,y1)
    - Optionally resizes back to (target_w, target_h)
    Default use is for face variants only.
    """
    if not tpl.enabled:
        return img

    if tpl.apply_to_face_variants_only and variant_suffix not in FACE_VARIANTS:
        return img

    # Validate box
    x0 = clamp01(tpl.x0)
    y0 = clamp01(tpl.y0)
    x1 = clamp01(tpl.x1)
    y1 = clamp01(tpl.y1)

    # Ensure ordering and minimum size
    if x1 <= x0:
        x1 = min(1.0, x0 + 0.10)
    if y1 <= y0:
        y1 = min(1.0, y0 + 0.10)

    w, h = img.size
    left = int(round(x0 * w))
    top = int(round(y0 * h))
    right = int(round(x1 * w))
    bottom = int(round(y1 * h))

    # Boundaries
    left = max(0, min(left, w - 2))
    top = max(0, min(top, h - 2))
    right = max(left + 2, min(right, w))
    bottom = max(top + 2, min(bottom, h))

    cropped = img.crop((left, top, right, bottom))

    if tpl.resize_back_to_target:
        cropped = cropped.resize((target_w, target_h), resample=Image.LANCZOS)

    return cropped


# ============================================================
# Stability generation
# ============================================================
def get_stability_client(api_key: str, engine: str, host: str):
    os.environ["STABILITY_HOST"] = host
    os.environ["STABILITY_KEY"] = api_key
    return client.StabilityInference(
        key=api_key,
        verbose=False,
        engine=engine,
    )

def generate_images(
    stability_api,
    prompt: str,
    negative_prompt: str,
    samples: int,
    steps: int,
    cfg_scale: float,
    width: int,
    height: int,
    style_preset: Optional[str],
) -> List[Image.Image]:
    answers = stability_api.generate(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt.strip() else None,
        samples=samples,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        style_preset=style_preset if style_preset else None,
    )

    imgs: List[Image.Image] = []
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                raise RuntimeError("Safety filter triggered for this prompt.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary)).convert("RGBA")
                imgs.append(img)
    return imgs

def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================
# Google Drive helpers (service account)
# ============================================================
def drive_service_from_sa_json(sa_info: dict):
    creds = service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return build("drive", "v3", credentials=creds)

def drive_list_files(service, folder_id: str) -> List[dict]:
    q = f"'{folder_id}' in parents and trashed=false"
    res = service.files().list(
        q=q,
        fields="files(id,name,mimeType,modifiedTime)",
        pageSize=200,
    ).execute()
    return res.get("files", [])

def drive_download_file(service, file_id: str, mime_type: str) -> bytes:
    if mime_type == "application/vnd.google-apps.document":
        request = service.files().export_media(
            fileId=file_id,
            mimeType="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    else:
        request = service.files().get_media(fileId=file_id)

    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue()

def drive_find_or_create_folder(service, parent_folder_id: str, folder_name: str) -> str:
    q = (
        f"'{parent_folder_id}' in parents and trashed=false and "
        f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}'"
    )
    res = service.files().list(q=q, fields="files(id,name)", pageSize=10).execute()
    files = res.get("files", [])
    if files:
        return files[0]["id"]

    meta = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }
    created = service.files().create(body=meta, fields="id").execute()
    return created["id"]

def drive_upload_png(service, folder_id: str, file_name: str, png_bytes: bytes) -> dict:
    media = MediaIoBaseUpload(io.BytesIO(png_bytes), mimetype="image/png", resumable=False)
    meta = {"name": file_name, "parents": [folder_id]}
    return service.files().create(body=meta, media_body=media, fields="id,name").execute()


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="OutPaged Visualizer", layout="wide")
st.title("📚 OutPaged Concept Generator")
st.caption("Preview style, then batch-generate backgrounds and character training packs with consistent framing.")

# Sidebar settings
with st.sidebar:
    st.header("API Settings")
    api_key = st.text_input("Stability API Key", type="password", value=st.secrets.get("STABILITY_KEY", ""))
    host = st.text_input("STABILITY_HOST", value=st.secrets.get("STABILITY_HOST", "grpc.stability.ai:443"))
    engine = st.text_input("Engine", value=st.secrets.get("STABILITY_ENGINE", "stable-diffusion-xl-1024-v1-0"))

    st.divider()
    st.header("Generation Settings")
    style_preset = st.selectbox("Style Preset", ["", "cinematic", "fantasy-art", "photographic", "comic-book"])
    negative_prompt = st.text_area(
        "Negative Prompt (optional)",
        "text, watermark, logo, lowres, blurry, deformed, extra limbs",
        height=90
    )
    steps = st.slider("Steps", 10, 60, 30)
    cfg_scale = st.slider("CFG Scale", 1.0, 12.0, 7.0)

    st.divider()
    st.header("Size Presets")
    size_mode = st.radio("Preset", ["Background 16:9", "Character Square", "Custom"], index=0)
    if size_mode == "Background 16:9":
        width, height = 1792, 1024
    elif size_mode == "Character Square":
        width, height = 1024, 1024
    else:
        width = st.number_input("Width", min_value=256, max_value=2048, value=1024, step=64)
        height = st.number_input("Height", min_value=256, max_value=2048, value=1024, step=64)

    st.divider()
    st.header("Batch")
    samples_per_prompt = st.slider("Images per prompt", 1, 10, 1)
    preview_only = st.checkbox("Preview only (force 1 image)", value=True)

    st.divider()
    st.header("Character Pack Template")
    framing_enabled = st.checkbox("Enable consistent framing for character training", value=True)
    apply_face_only = st.checkbox("Apply framing to face variants only", value=True)
    resize_back = st.checkbox("Resize cropped region back to target size", value=True)

    st.caption("Crop box percentages (normalized 0.0 to 1.0).")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        crop_x0 = st.number_input("x0", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
        crop_y0 = st.number_input("y0", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    with col_f2:
        crop_x1 = st.number_input("x1", min_value=0.0, max_value=1.0, value=0.80, step=0.01)
        crop_y1 = st.number_input("y1", min_value=0.0, max_value=1.0, value=0.65, step=0.01)

    framing_tpl = FramingTemplate(
        enabled=framing_enabled,
        apply_to_face_variants_only=apply_face_only,
        x0=crop_x0,
        y0=crop_y0,
        x1=crop_x1,
        y1=crop_y1,
        resize_back_to_target=resize_back,
    )

    st.divider()
    st.header("Drive (optional)")
    sa_json_text = st.text_area(
        "Service Account JSON (paste) or use st.secrets['GDRIVE_SA_JSON']",
        value=st.secrets.get("GDRIVE_SA_JSON", ""),
        height=120
    )
    drive_input_folder = st.text_input("Drive Input Folder ID", value=st.secrets.get("GDRIVE_INPUT_FOLDER_ID", ""))
    drive_output_folder = st.text_input("Drive Output Folder ID", value=st.secrets.get("GDRIVE_OUTPUT_FOLDER_ID", ""))


# Session state init
if "raw_text" not in st.session_state:
    st.session_state["raw_text"] = ""
if "backgrounds" not in st.session_state:
    st.session_state["backgrounds"] = []
if "canon_chars" not in st.session_state:
    st.session_state["canon_chars"] = {}
if "parsed_ok" not in st.session_state:
    st.session_state["parsed_ok"] = False


tabs = st.tabs(["1) Import + Parse", "2) Preview", "3) Batch + Export", "4) Drive Import + Upload"])


# ============================================================
# Tab 1: Import + Parse
# ============================================================
with tabs[0]:
    st.subheader("Import your prompt doc content")

    source_mode = st.radio("Input type", ["Paste text", "Upload DOCX"], horizontal=True)

    raw_text = ""
    if source_mode == "Paste text":
        raw_text = st.text_area(
            "Paste prompts here (background scenes + Canonical Master Prompts)",
            value=st.session_state["raw_text"],
            height=360
        )
    else:
        up = st.file_uploader("Upload a .docx", type=["docx"])
        if up is not None:
            raw_text = read_docx_to_text(up.read())
            st.success("DOCX loaded and converted to text.")
        else:
            raw_text = st.session_state["raw_text"]

    col1, col2 = st.columns([1, 3])
    with col1:
        parse_btn = st.button("Parse Document", type="primary")
    with col2:
        st.info("Parser is tuned to your format: 'Background file name:' and 'Canonical Master Prompts' blocks.")

    if parse_btn:
        st.session_state["raw_text"] = raw_text
        bgs, chars = parse_outpaged_text(raw_text)
        st.session_state["backgrounds"] = bgs
        st.session_state["canon_chars"] = chars
        st.session_state["parsed_ok"] = True

    if st.session_state["parsed_ok"]:
        bgs: List[BackgroundItem] = st.session_state["backgrounds"]
        chars: Dict[str, CanonCharacter] = st.session_state["canon_chars"]

        st.write(f"Background prompts found: **{len(bgs)}**")
        if bgs:
            st.dataframe(
                [{"file_name": b.file_name, "prompt_preview": (b.prompt[:140] + ("..." if len(b.prompt) > 140 else ""))} for b in bgs],
                use_container_width=True
            )

        st.write(f"Canonical characters found: **{len(chars)}**")
        if chars:
            st.dataframe(
                [{"name": c.name, "canon_file_name": c.canon_file_name, "prompt_preview": (c.prompt[:140] + ("..." if len(c.prompt) > 140 else ""))} for c in chars.values()],
                use_container_width=True
            )


# ============================================================
# Tab 2: Preview
# ============================================================
with tabs[1]:
    st.subheader("Preview one image before batch generation")

    if not st.session_state["parsed_ok"]:
        st.warning("Go to 'Import + Parse' first and parse your text.")
    else:
        bgs: List[BackgroundItem] = st.session_state["backgrounds"]
        chars: Dict[str, CanonCharacter] = st.session_state["canon_chars"]

        preview_type = st.radio("Preview type", ["Background", "Character training"], horizontal=True)

        if preview_type == "Background":
            if not bgs:
                st.warning("No background prompts found.")
            else:
                bg_options = {f"{b.file_name}": b for b in bgs}
                selected_key = st.selectbox("Choose a background", list(bg_options.keys()))
                item = bg_options[selected_key]

                st.text_area("Prompt", item.prompt, height=140)

                if st.button("Generate 1 Preview", type="primary"):
                    if not api_key:
                        st.error("Add your Stability API key in the sidebar.")
                    else:
                        try:
                            stability_api = get_stability_client(api_key, engine, host)
                            imgs = generate_images(
                                stability_api,
                                prompt=item.prompt,
                                negative_prompt=negative_prompt,
                                samples=1,
                                steps=steps,
                                cfg_scale=cfg_scale,
                                width=width,
                                height=height,
                                style_preset=style_preset,
                            )
                            img = imgs[0]
                            st.image(img, caption=f"{item.file_name}.png", use_container_width=True)

                            png_bytes = pil_to_png_bytes(img)
                            st.download_button(
                                "Download PNG",
                                data=png_bytes,
                                file_name=f"{item.file_name}.png",
                                mime="image/png"
                            )
                        except Exception as e:
                            st.error(str(e))

        else:
            if not chars:
                st.warning("No canonical characters found.")
            else:
                names = list(chars.keys())
                selected_name = st.selectbox("Choose a character", names)
                char_item = chars[selected_name]

                variant = st.selectbox(
                    "Training variant",
                    [v[0] for v in TRAIN_VARIANTS],
                    index=0
                )

                composed_prompt = build_character_variant_prompt(char_item.prompt, variant)
                st.text_area("Composed prompt", composed_prompt, height=170)

                if st.button("Generate 1 Preview", type="primary"):
                    if not api_key:
                        st.error("Add your Stability API key in the sidebar.")
                    else:
                        try:
                            stability_api = get_stability_client(api_key, engine, host)
                            imgs = generate_images(
                                stability_api,
                                prompt=composed_prompt,
                                negative_prompt=negative_prompt,
                                samples=1,
                                steps=steps,
                                cfg_scale=cfg_scale,
                                width=width,
                                height=height,
                                style_preset=style_preset,
                            )

                            img = imgs[0]
                            img = apply_framing(img, variant, framing_tpl, width, height)

                            out_name = f"{char_item.canon_file_name}_{variant}.png"
                            st.image(img, caption=out_name, use_container_width=True)

                            png_bytes = pil_to_png_bytes(img)
                            st.download_button(
                                "Download PNG",
                                data=png_bytes,
                                file_name=out_name,
                                mime="image/png"
                            )
                        except Exception as e:
                            st.error(str(e))


# ============================================================
# Tab 3: Batch + Export ZIP
# ============================================================
with tabs[2]:
    st.subheader("Batch generation and export")

    if not st.session_state["parsed_ok"]:
        st.warning("Go to 'Import + Parse' first and parse your text.")
    else:
        bgs: List[BackgroundItem] = st.session_state["backgrounds"]
        chars: Dict[str, CanonCharacter] = st.session_state["canon_chars"]

        st.write("### Select what to generate")
        colA, colB = st.columns(2)

        with colA:
            bg_keys = [b.file_name for b in bgs]
            selected_bgs = st.multiselect("Backgrounds", bg_keys, default=bg_keys[: min(10, len(bg_keys))])

        with colB:
            char_names = list(chars.keys())
            selected_chars = st.multiselect("Canonical characters", char_names, default=char_names[: min(6, len(char_names))])

        variant_keys = [v[0] for v in TRAIN_VARIANTS]
        selected_variants = st.multiselect("Training variants", variant_keys, default=variant_keys)

        st.write("### Run batch")
        run_batch = st.button("Generate Batch and Build ZIP", type="primary")

        if run_batch:
            if not api_key:
                st.error("Add your Stability API key in the sidebar.")
            else:
                try:
                    stability_api = get_stability_client(api_key, engine, host)

                    zip_buf = io.BytesIO()
                    zf = zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED)

                    total_jobs = len(selected_bgs) + (len(selected_chars) * len(selected_variants))
                    if total_jobs == 0:
                        st.warning("Nothing selected.")
                    else:
                        progress = st.progress(0.0)
                        done = 0

                        gallery = st.columns(4)
                        shown = 0

                        # Backgrounds
                        bg_map = {b.file_name: b for b in bgs}
                        for file_name in selected_bgs:
                            item = bg_map[file_name]
                            n = 1 if preview_only else samples_per_prompt

                            imgs = generate_images(
                                stability_api,
                                prompt=item.prompt,
                                negative_prompt=negative_prompt,
                                samples=n,
                                steps=steps,
                                cfg_scale=cfg_scale,
                                width=width,
                                height=height,
                                style_preset=style_preset,
                            )

                            for j, img in enumerate(imgs):
                                suffix = f"_v{j+1:02d}" if n > 1 else ""
                                out_name = f"{file_name}{suffix}.png"
                                png_bytes = pil_to_png_bytes(img)

                                zf.writestr(f"backgrounds/{out_name}", png_bytes)

                                if shown < 12:
                                    with gallery[shown % 4]:
                                        st.image(img, caption=f"backgrounds/{out_name}", use_container_width=True)
                                    shown += 1

                            done += 1
                            progress.progress(done / total_jobs)

                        # Characters
                        for char_name in selected_chars:
                            c = chars[char_name]

                            for variant in selected_variants:
                                composed_prompt = build_character_variant_prompt(c.prompt, variant)
                                n = 1 if preview_only else samples_per_prompt

                                imgs = generate_images(
                                    stability_api,
                                    prompt=composed_prompt,
                                    negative_prompt=negative_prompt,
                                    samples=n,
                                    steps=steps,
                                    cfg_scale=cfg_scale,
                                    width=width,
                                    height=height,
                                    style_preset=style_preset,
                                )

                                for j, img in enumerate(imgs):
                                    img = apply_framing(img, variant, framing_tpl, width, height)

                                    suffix = f"_v{j+1:02d}" if n > 1 else ""
                                    out_name = f"{c.canon_file_name}_{variant}{suffix}.png"
                                    png_bytes = pil_to_png_bytes(img)

                                    zf.writestr(f"characters_training/{c.canon_file_name}/{out_name}", png_bytes)

                                    if shown < 12:
                                        with gallery[shown % 4]:
                                            st.image(img, caption=f"characters_training/{c.canon_file_name}/{out_name}", use_container_width=True)
                                        shown += 1

                                done += 1
                                progress.progress(done / total_jobs)

                        # Metadata for reproducibility
                        meta = {
                            "engine": engine,
                            "host": host,
                            "style_preset": style_preset,
                            "negative_prompt": negative_prompt,
                            "steps": steps,
                            "cfg_scale": cfg_scale,
                            "width": width,
                            "height": height,
                            "samples_per_prompt": 1 if preview_only else samples_per_prompt,
                            "backgrounds_selected": selected_bgs,
                            "characters_selected": selected_chars,
                            "training_variants_selected": selected_variants,
                            "character_pack_template": {
                                "enabled": framing_tpl.enabled,
                                "apply_to_face_variants_only": framing_tpl.apply_to_face_variants_only,
                                "crop_box": {"x0": framing_tpl.x0, "y0": framing_tpl.y0, "x1": framing_tpl.x1, "y1": framing_tpl.y1},
                                "resize_back_to_target": framing_tpl.resize_back_to_target,
                            },
                        }
                        zf.writestr("metadata/run_settings.json", json.dumps(meta, indent=2))

                        zf.close()
                        zip_buf.seek(0)

                        st.success("Batch complete.")
                        st.download_button(
                            "Download ZIP",
                            data=zip_buf.getvalue(),
                            file_name="outpaged_outputs.zip",
                            mime="application/zip"
                        )

                except Exception as e:
                    st.error(str(e))


# ============================================================
# Tab 4: Drive Import + Upload
# ============================================================
with tabs[3]:
    st.subheader("Google Drive import and upload (service account)")

    st.write(
        "Recommended setup:\n"
        "1) Create a service account.\n"
        "2) Share your Drive input folder and output folder with the service account email.\n"
        "3) Paste the service account JSON into the sidebar.\n"
    )

    if not sa_json_text.strip() or not drive_input_folder.strip():
        st.info("Add service account JSON and a Drive input folder ID in the sidebar to use this tab.")
    else:
        try:
            sa_info = json.loads(sa_json_text)
            service = drive_service_from_sa_json(sa_info)

            files = drive_list_files(service, drive_input_folder)
            doc_candidates = [
                f for f in files
                if f["mimeType"] in (
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/vnd.google-apps.document",
                )
            ]

            st.write(f"Docs found in input folder: **{len(doc_candidates)}**")
            if not doc_candidates:
                st.warning("No DOCX or Google Docs found. Check the folder ID and sharing permissions.")
            else:
                selected = st.selectbox(
                    "Choose a prompt document",
                    options=doc_candidates,
                    format_func=lambda x: f'{x["name"]} ({x["mimeType"]})'
                )

                if st.button("Load + Parse from Drive", type="primary"):
                    content = drive_download_file(service, selected["id"], selected["mimeType"])
                    text = read_docx_to_text(content)
                    st.session_state["raw_text"] = text
                    bgs, chars = parse_outpaged_text(text)
                    st.session_state["backgrounds"] = bgs
                    st.session_state["canon_chars"] = chars
                    st.session_state["parsed_ok"] = True
                    st.success(f"Parsed {len(bgs)} backgrounds and {len(chars)} canonical characters.")

                if st.session_state.get("parsed_ok", False):
                    st.divider()
                    st.write("### Generate and upload")

                    if not drive_output_folder.strip():
                        st.warning("Add a Drive output folder ID in the sidebar.")
                    else:
                        bgs: List[BackgroundItem] = st.session_state["backgrounds"]
                        chars: Dict[str, CanonCharacter] = st.session_state["canon_chars"]

                        bg_keys = [b.file_name for b in bgs]
                        selected_bgs = st.multiselect("Backgrounds to upload", bg_keys, default=bg_keys[: min(10, len(bg_keys))])

                        char_names = list(chars.keys())
                        selected_chars = st.multiselect("Characters to upload", char_names, default=char_names[: min(6, len(char_names))])

                        variant_keys = [v[0] for v in TRAIN_VARIANTS]
                        selected_variants = st.multiselect("Training variants to upload", variant_keys, default=variant_keys)

                        upload_btn = st.button("Generate and Upload to Drive", type="primary")

                        if upload_btn:
                            if not api_key:
                                st.error("Add your Stability API key in the sidebar.")
                            else:
                                try:
                                    stability_api = get_stability_client(api_key, engine, host)

                                    backgrounds_folder_id = drive_find_or_create_folder(service, drive_output_folder, "backgrounds")
                                    characters_folder_id = drive_find_or_create_folder(service, drive_output_folder, "characters_training")

                                    total_jobs = len(selected_bgs) + (len(selected_chars) * len(selected_variants))
                                    if total_jobs == 0:
                                        st.warning("Nothing selected.")
                                    else:
                                        progress = st.progress(0.0)
                                        done = 0
                                        uploaded = 0

                                        # Backgrounds
                                        bg_map = {b.file_name: b for b in bgs}
                                        for file_name in selected_bgs:
                                            item = bg_map[file_name]
                                            n = 1 if preview_only else samples_per_prompt

                                            imgs = generate_images(
                                                stability_api,
                                                prompt=item.prompt,
                                                negative_prompt=negative_prompt,
                                                samples=n,
                                                steps=steps,
                                                cfg_scale=cfg_scale,
                                                width=width,
                                                height=height,
                                                style_preset=style_preset,
                                            )

                                            for j, img in enumerate(imgs):
                                                suffix = f"_v{j+1:02d}" if n > 1 else ""
                                                out_name = f"{file_name}{suffix}.png"
                                                png_bytes = pil_to_png_bytes(img)
                                                drive_upload_png(service, backgrounds_folder_id, out_name, png_bytes)
                                                uploaded += 1

                                            done += 1
                                            progress.progress(done / total_jobs)

                                        # Characters
                                        for char_name in selected_chars:
                                            c = chars[char_name]
                                            per_char_folder_id = drive_find_or_create_folder(service, characters_folder_id, c.canon_file_name)

                                            for variant in selected_variants:
                                                composed_prompt = build_character_variant_prompt(c.prompt, variant)
                                                n = 1 if preview_only else samples_per_prompt

                                                imgs = generate_images(
                                                    stability_api,
                                                    prompt=composed_prompt,
                                                    negative_prompt=negative_prompt,
                                                    samples=n,
                                                    steps=steps,
                                                    cfg_scale=cfg_scale,
                                                    width=width,
                                                    height=height,
                                                    style_preset=style_preset,
                                                )

                                                for j, img in enumerate(imgs):
                                                    img = apply_framing(img, variant, framing_tpl, width, height)

                                                    suffix = f"_v{j+1:02d}" if n > 1 else ""
                                                    out_name = f"{c.canon_file_name}_{variant}{suffix}.png"
                                                    png_bytes = pil_to_png_bytes(img)
                                                    drive_upload_png(service, per_char_folder_id, out_name, png_bytes)
                                                    uploaded += 1

                                                done += 1
                                                progress.progress(done / total_jobs)

                                        st.success(f"Uploaded {uploaded} images to Drive.")

                                except Exception as e:
                                    st.error(str(e))

        except Exception as e:
            st.error(f"Drive error: {e}")
