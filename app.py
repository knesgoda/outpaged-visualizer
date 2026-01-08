import io
import os
import re
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image
from docx import Document


# ----------------------------
# OutPaged Visualizer (Streamlit)
# - Upload DOCX/TXT prompt docs
# - Pick a Scene Background prompt or Character prompt
# - Optional reference image (img2img style/structure)
# - Generate single preview or batch
# - Download a ZIP containing images + manifest.json
# ----------------------------

STABILITY_API_BASE = "https://api.stability.ai"
STABILITY_IMAGE_ENDPOINTS = {
    "Stable Image Ultra (best quality)": "/v2beta/stable-image/generate/ultra",
    "Stable Image Core (faster/cheaper)": "/v2beta/stable-image/generate/core",
}

ASPECT_RATIOS = {
    "16:9 (backgrounds)": "16:9",
    "1:1 (characters, faces)": "1:1",
    "9:16 (vertical)": "9:16",
    "4:3": "4:3",
    "3:2": "3:2",
    "2:3": "2:3",
}


@dataclass
class PromptItem:
    kind: str  # "scene" or "character"
    key: str   # filename or name
    title: str # display title
    prompt: str


def _get_stability_key() -> str:
    # Prefer Streamlit Secrets, then environment variables.
    key = ""
    if hasattr(st, "secrets") and "STABILITY_API_KEY" in st.secrets:
        key = str(st.secrets["STABILITY_API_KEY"]).strip()
    if not key:
        key = os.getenv("STABILITY_API_KEY", "").strip()
    if not key:
        key = os.getenv("STABILITY_KEY", "").strip()  # legacy name
    return key


def _read_docx_bytes(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    parts = [p.text for p in doc.paragraphs if p.text is not None]
    return "\n".join(parts).strip()


def _read_uploaded_prompt_doc(uploaded_file) -> str:
    raw = uploaded_file.read()
    name = uploaded_file.name.lower()
    if name.endswith(".docx"):
        return _read_docx_bytes(raw)
    # Treat as text
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def _parse_prompt_doc(text: str) -> Tuple[List[PromptItem], List[PromptItem]]:
    """
    Parses docs that look like your Gulliver prompt dumps:
    - Scenes with "Background file name: ch##bg##" + prompt line(s)
    - Canonical Master Prompts section for characters
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    scenes: List[PromptItem] = []
    chars: List[PromptItem] = []

    # ---- Scenes ----
    pending_scene_label = ""
    pending_scene_title = ""
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if re.match(r"^Scene\s+\d+", line, flags=re.IGNORECASE):
            pending_scene_label = line
            pending_scene_title = ""
            # next non-empty line that is not a "Background file name" becomes title (if present)
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and not lines[j].strip().lower().startswith("background file name"):
                pending_scene_title = lines[j].strip()
            i += 1
            continue

        if line.lower().startswith("background file name"):
            filename = line.split(":", 1)[1].strip() if ":" in line else line.replace("Background file name", "").strip()
            # prompt starts on next line(s) until blank line or next Scene/Canonical section
            prompt_lines = []
            j = i + 1
            while j < len(lines):
                nxt = lines[j].strip()
                if not nxt:
                    break
                if re.match(r"^Scene\s+\d+", nxt, flags=re.IGNORECASE):
                    break
                if nxt.lower().startswith("canonical master prompts"):
                    break
                if nxt.lower().startswith("background file name"):
                    break
                prompt_lines.append(nxt)
                j += 1

            prompt_text = " ".join(prompt_lines).strip()
            title = pending_scene_title or pending_scene_label or filename
            if prompt_text:
                scenes.append(
                    PromptItem(
                        kind="scene",
                        key=filename,
                        title=f"{filename} | {title}",
                        prompt=prompt_text,
                    )
                )
            i = j
            continue

        if line.lower().startswith("canonical master prompts"):
            break

        i += 1

    # ---- Characters ----
    # Find canonical section start
    canon_start = None
    for idx, ln in enumerate(lines):
        if ln.strip().lower().startswith("canonical master prompts"):
            canon_start = idx
            break

    if canon_start is not None:
        # heuristics: header lines are "short" and have no obvious punctuation
        def is_header(s: str) -> bool:
            s = s.strip()
            if not s:
                return False
            if len(s) > 50:
                return False
            if any(ch in s for ch in [",", ".", ":", "—", "–", "(", ")"]):
                return False
            # avoid generic lines
            lower = s.lower()
            if lower.startswith("use these"):
                return False
            if lower.startswith("canonical"):
                return False
            return True

        current_name = ""
        buffer: List[str] = []
        for ln in lines[canon_start + 1:]:
            s = ln.strip()
            if not s:
                continue
            if is_header(s):
                # flush prior
                if current_name and buffer:
                    chars.append(
                        PromptItem(
                            kind="character",
                            key=current_name,
                            title=current_name,
                            prompt=" ".join(buffer).strip(),
                        )
                    )
                current_name = s
                buffer = []
            else:
                if current_name:
                    buffer.append(s)

        if current_name and buffer:
            chars.append(
                PromptItem(
                    kind="character",
                    key=current_name,
                    title=current_name,
                    prompt=" ".join(buffer).strip(),
                )
            )

    return scenes, chars


def _stability_generate_png(
    api_key: str,
    endpoint_path: str,
    prompt: str,
    *,
    negative_prompt: str = "",
    aspect_ratio: str = "16:9",
    seed: Optional[int] = None,
    reference_image_bytes: Optional[bytes] = None,
    strength: float = 0.35,
    timeout_s: int = 120,
) -> bytes:
    url = f"{STABILITY_API_BASE}{endpoint_path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "image/*",  # <-- required by the API (or application/json)
    }

    data = {
        "prompt": prompt,
        "output_format": "png",
        "aspect_ratio": aspect_ratio,
    }
    if negative_prompt.strip():
        data["negative_prompt"] = negative_prompt.strip()
    if seed is not None:
        data["seed"] = str(int(seed))

    files = None
    if reference_image_bytes:
        files = {"image": ("reference.png", reference_image_bytes, "image/png")}
        data["strength"] = str(float(strength))  # required when image is provided

    resp = requests.post(url, headers=headers, data=data, files=files, timeout=timeout_s)

    if resp.status_code == 200:
        return resp.content

    # Helpful error surface
    try:
        err = resp.json()
    except Exception:
        err = {"error": resp.text}

    raise RuntimeError(f"Stability API error {resp.status_code}: {json.dumps(err)}")



def _make_zip(images: List[Tuple[str, bytes]], manifest: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        for fname, b in images:
            z.writestr(fname, b)
    return buf.getvalue()


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="OutPaged Visualizer", layout="wide")
st.title("📚 OutPaged Visualizer")
st.caption("Generate consistent backgrounds and character training shots from your prompt docs.")

api_key = _get_stability_key()
if not api_key:
    st.warning("Add your Stability key in Streamlit Secrets as STABILITY_API_KEY, then rerun.")

with st.sidebar:
    st.header("Generation")
    endpoint_label = st.selectbox("Model", list(STABILITY_IMAGE_ENDPOINTS.keys()), index=0)
    endpoint_path = STABILITY_IMAGE_ENDPOINTS[endpoint_label]

    aspect_label = st.selectbox("Aspect ratio", list(ASPECT_RATIOS.keys()), index=0)
    aspect_ratio = ASPECT_RATIOS[aspect_label]

    negative_prompt = st.text_area(
        "Negative prompt (optional)",
        value="text, watermark, logo, signature, low quality, blurry, extra limbs, malformed",
        height=90,
    )

    batch_size = st.slider("Batch size", 1, 20, 4)
    seed_mode = st.selectbox("Seed mode", ["Random", "Fixed seed"], index=0)
    seed_value = None
    if seed_mode == "Fixed seed":
        seed_value = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=12345, step=1)

    st.divider()
    st.subheader("Reference image (optional)")
    ref_image = st.file_uploader("Upload a reference image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    strength = st.slider(
        "Strength (only used with a reference image)",
        min_value=0.05,
        max_value=0.95,
        value=0.35,
        step=0.05,
        help="Lower keeps the original image more. Higher reimagines more. Required by Stability when an image is provided.",
    )

    st.divider()
    preview_btn = st.button("Preview 1", type="secondary", use_container_width=True)
    generate_btn = st.button("Generate batch", type="primary", use_container_width=True)

st.subheader("1) Prompt source")

col_a, col_b = st.columns([1, 1])

with col_a:
    uploaded_doc = st.file_uploader("Upload prompt document (DOCX or TXT)", type=["docx", "txt"])
    if uploaded_doc:
        doc_text = _read_uploaded_prompt_doc(uploaded_doc)
        scenes, chars = _parse_prompt_doc(doc_text)
        st.session_state["scenes"] = scenes
        st.session_state["chars"] = chars
        st.success(f"Loaded {len(scenes)} scenes and {len(chars)} character prompts.")

with col_b:
    manual_prompt = st.text_area(
        "Or paste a prompt directly",
        value="Rocky coastline after a violent storm, shattered mast and sailcloth tangled in seaweed, wet sand with oversized human footprints leading inland, tiny arrows and thread-thin ropes scattered near the prints, cold dawn fog rolling off the sea. Style: vivid chalk pastel illustration with deep 3D depth, rich layered chalk texture, crisp chalk lines, cinematic 16:9, no people, no text.",
        height=140,
    )

st.subheader("2) Choose from loaded items (optional)")

scenes: List[PromptItem] = st.session_state.get("scenes", [])
chars: List[PromptItem] = st.session_state.get("chars", [])

tab1, tab2 = st.tabs(["Scenes (backgrounds)", "Characters"])

selected_prompt_text = manual_prompt
selected_label = "manual"

with tab1:
    if scenes:
        scene_labels = [p.title for p in scenes]
        choice = st.selectbox("Scene prompt", ["(manual)"] + scene_labels, index=0)
        if choice != "(manual)":
            item = next(p for p in scenes if p.title == choice)
            selected_prompt_text = item.prompt
            selected_label = item.key
        st.code(selected_prompt_text, language="text")
    else:
        st.info("Upload a prompt doc to populate scenes.")

with tab2:
    if chars:
        char_labels = [p.title for p in chars]
        choice = st.selectbox("Character prompt", ["(manual)"] + char_labels, index=0)
        if choice != "(manual)":
            item = next(p for p in chars if p.title == choice)
            selected_prompt_text = item.prompt
            selected_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", item.key.strip())[:80] or "character"
        st.code(selected_prompt_text, language="text")
    else:
        st.info("Upload a prompt doc to populate characters.")

st.subheader("3) Generate")

ref_bytes = None
if ref_image:
    ref_bytes = ref_image.read()
    try:
        st.image(Image.open(io.BytesIO(ref_bytes)), caption="Reference image", use_container_width=True)
    except Exception:
        st.warning("Could not preview the uploaded reference image, but it will still be sent to the API.")


def _run_generation(n: int) -> List[Tuple[str, bytes]]:
    images_out: List[Tuple[str, bytes]] = []
    progress = st.progress(0)
    status = st.empty()

    for idx in range(n):
        status.write(f"Generating {idx+1} of {n}...")
        png_bytes = _stability_generate_png(
            api_key=api_key,
            endpoint_path=endpoint_path,
            prompt=selected_prompt_text,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            seed=(int(seed_value) + idx) if seed_value is not None else None,
            reference_image_bytes=ref_bytes,
            strength=strength,
        )
        fname = f"{selected_label}_{idx+1:02d}.png"
        images_out.append((fname, png_bytes))
        progress.progress((idx + 1) / n)

    status.empty()
    return images_out


if preview_btn or generate_btn:
    if not api_key:
        st.error("Missing STABILITY_API_KEY. Add it in Streamlit Secrets and rerun.")
    elif not selected_prompt_text.strip():
        st.error("Prompt is empty.")
    else:
        try:
            n = 1 if preview_btn else int(batch_size)
            imgs = _run_generation(n)

            st.success(f"Done. Generated {len(imgs)} image(s).")
            cols = st.columns(4)
            for i, (fname, b) in enumerate(imgs):
                with cols[i % 4]:
                    st.image(Image.open(io.BytesIO(b)), caption=fname, use_container_width=True)

            manifest = {
                "created_at_utc": datetime.utcnow().isoformat() + "Z",
                "model": endpoint_label,
                "endpoint": endpoint_path,
                "aspect_ratio": aspect_ratio,
                "seed_mode": seed_mode,
                "seed": int(seed_value) if seed_value is not None else None,
                "has_reference_image": bool(ref_bytes),
                "strength": float(strength) if ref_bytes else None,
                "prompt": selected_prompt_text,
                "negative_prompt": negative_prompt,
                "files": [f for f, _ in imgs],
            }
            zip_bytes = _make_zip(imgs, manifest)

            st.download_button(
                "Download ZIP (images + manifest)",
                data=zip_bytes,
                file_name=f"outpaged_{selected_label}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                use_container_width=True,
            )

        except Exception as e:
            st.exception(e)

st.divider()
st.subheader("Notes for character training shots")
st.markdown(
    "- For training: set aspect ratio to **1:1**, use a clean neutral prompt, and generate:\n"
    "  - full body neutral pose\n"
    "  - face: straight-on, 45°, profile\n"
    "- If you're using a reference image to match a style, start with **strength 0.25 to 0.45** and adjust.\n"
)
