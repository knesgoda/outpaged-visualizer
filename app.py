import io
import os
import re
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image

try:
    from docx import Document
except Exception:
    Document = None


# =========================
# Configuration
# =========================

STABILITY_API_BASE = "https://api.stability.ai"
ULTRA_ENDPOINT = f"{STABILITY_API_BASE}/v2beta/stable-image/generate/ultra"
CORE_ENDPOINT = f"{STABILITY_API_BASE}/v2beta/stable-image/generate/core"

DEFAULT_NEGATIVE = (
    "low quality, blurry, jpeg artifacts, watermark, signature, text, logo, "
    "extra limbs, deformed, bad anatomy, oversaturated"
)

ASPECT_RATIOS = ["16:9", "1:1", "3:2", "2:3", "4:5", "5:4", "9:16"]
OUTPUT_FORMATS = ["png", "webp", "jpg"]


@dataclass
class PromptItem:
    file_base: str
    prompt: str


# =========================
# Helpers
# =========================

def _get_api_key() -> Optional[str]:
    # Streamlit Cloud best practice: put this in App Settings -> Secrets
    # STABILITY_KEY="sk-..."
    key = None
    if hasattr(st, "secrets"):
        key = st.secrets.get("STABILITY_KEY", None)
    if not key:
        key = os.getenv("STABILITY_KEY")
    return key


def _read_uploaded_text(uploaded_file) -> str:
    name = (uploaded_file.name or "").lower()

    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if name.endswith(".docx"):
        if Document is None:
            raise RuntimeError("python-docx not available. Check requirements.txt.")
        doc = Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs])

    raise ValueError("Unsupported file type. Upload .txt or .docx")


def _parse_background_items(raw_text: str) -> List[PromptItem]:
    """
    Looks for blocks like:
      Background file name: ch01bg01
      <prompt line(s)>

    It will grab the file base and the next non-empty line(s) until a blank line
    or the next "Scene" block header.
    """
    lines = [ln.rstrip() for ln in raw_text.splitlines()]
    items: List[PromptItem] = []

    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.lower().startswith("background file name:"):
            file_base = ln.split(":", 1)[1].strip()
            i += 1

            # collect prompt lines until blank line OR next scene header-ish
            prompt_lines = []
            while i < len(lines):
                cur = lines[i].strip()
                if cur == "":
                    if prompt_lines:
                        break
                    i += 1
                    continue

                # stop if we hit a new scene label
                if re.match(r"^scene\s+\d+", cur.lower()):
                    break
                if cur.lower().startswith("background file name:"):
                    break

                prompt_lines.append(cur)
                i += 1

            prompt = " ".join(prompt_lines).strip()
            if file_base and prompt:
                items.append(PromptItem(file_base=file_base, prompt=prompt))
            continue

        i += 1

    return items


def _make_character_training_prompts(name: str, base_prompt: str) -> List[PromptItem]:
    """
    For training:
      1) neutral full body
      2) face straight
      3) face 45
      4) face profile
    """
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_").lower()
    if not clean:
        clean = "character"

    # You can tweak these strings any time to match your pipeline
    body_suffix = (
        "Neutral standing pose, arms relaxed at sides, full body visible, centered, "
        "no background, no shadow, no text."
    )
    face_common = (
        "Head and shoulders portrait, tight framing, neutral expression, "
        "no background, no shadow, no text."
    )

    items = [
        PromptItem(file_base=f"{clean}_body_neutral", prompt=f"{base_prompt} {body_suffix}"),
        PromptItem(file_base=f"{clean}_face_front", prompt=f"{base_prompt} {face_common} Facing camera straight on."),
        PromptItem(file_base=f"{clean}_face_45", prompt=f"{base_prompt} {face_common} 45 degree angle view."),
        PromptItem(file_base=f"{clean}_face_profile", prompt=f"{base_prompt} {face_common} Side profile view."),
    ]
    return items


def _stability_generate_one(
    api_key: str,
    endpoint: str,
    prompt: str,
    negative_prompt: str,
    aspect_ratio: str,
    output_format: str,
    seed: Optional[int] = None,
    reference_image_bytes: Optional[bytes] = None,
    image_denoise: float = 0.45,
    timeout_s: int = 120,
) -> bytes:
    """
    Uses Stability Stable Image REST endpoints (Ultra/Core).
    Official docs show Bearer auth and Accept: image/* patterns. :contentReference[oaicite:4]{index=4}
    Some clients also pass image_denoise for reference-image steering. :contentReference[oaicite:5]{index=5}
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "image/*",
    }

    data = {
        "prompt": prompt,
        "output_format": output_format,
        "aspect_ratio": aspect_ratio,
    }

    if negative_prompt.strip():
        data["negative_prompt"] = negative_prompt.strip()

    if seed is not None:
        data["seed"] = str(int(seed))

    files = None
    if reference_image_bytes is not None:
        # multipart with an image field named "image"
        files = {
            "image": ("reference.png", reference_image_bytes, "image/png")
        }
        # Many clients use image_denoise to control how strongly to follow the reference image. :contentReference[oaicite:6]{index=6}
        data["image_denoise"] = str(float(image_denoise))

    resp = requests.post(
        endpoint,
        headers=headers,
        data=data,
        files=files,
        timeout=timeout_s,
    )

    if not resp.ok:
        # Streamlit will redact some errors automatically; show useful status and body
        raise RuntimeError(f"Stability API error {resp.status_code}: {resp.text}")

    return resp.content


def _bytes_to_pil(img_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(img_bytes)).convert("RGBA")


def _zip_images(named_images: List[Tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, b in named_images:
            zf.writestr(name, b)
    return buf.getvalue()


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="OutPaged Visualizer", layout="wide")
st.title("📚 OutPaged Visualizer")
st.caption("Generate consistent background and character training images using Stability Stable Image (Ultra/Core).")

api_key = _get_api_key()
if not api_key:
    st.warning(
        "Add your API key as STABILITY_KEY in Streamlit Cloud Secrets, or as an environment variable. "
        "Example in Secrets:\n\nSTABILITY_KEY = \"sk-...\""
    )
    st.stop()

with st.sidebar:
    st.header("Generator Settings")

    model_choice = st.selectbox(
        "Model",
        ["Stable Image Ultra (best quality)", "Stable Image Core (faster)"],
        index=0,
    )
    endpoint = ULTRA_ENDPOINT if model_choice.startswith("Stable Image Ultra") else CORE_ENDPOINT

    aspect_ratio = st.selectbox("Aspect Ratio", ASPECT_RATIOS, index=0)
    output_format = st.selectbox("Output Format", OUTPUT_FORMATS, index=0)

    st.divider()
    use_seed = st.checkbox("Lock Seed (more consistent)", value=True)
    seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=123456, step=1) if use_seed else None

    variants = st.slider("Variants per prompt", 1, 8, 2)

    st.divider()
    st.subheader("Reference Image (optional)")
    ref_img = st.file_uploader("Upload reference image (PNG/JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
    image_denoise = st.slider(
        "Reference strength (image_denoise)",
        0.0, 1.0, 0.45,
        help="Lower keeps closer to the reference image. Higher reimagines more."
    )

    st.divider()
    st.subheader("Negative prompt")
    negative_prompt = st.text_area("Negative prompt", value=DEFAULT_NEGATIVE, height=120)

tab1, tab2, tab3 = st.tabs(["Single Prompt", "Upload Doc (Backgrounds)", "Character Training Pack"])


# -------------------------
# Tab 1: Single prompt
# -------------------------
with tab1:
    st.subheader("Single Prompt")
    prompt_text = st.text_area(
        "Prompt",
        value="Rocky coastline after a violent storm, shattered mast and sailcloth tangled in seaweed, wet sand with oversized human footprints leading inland, tiny arrows and thread-thin ropes scattered near the prints, cold dawn fog rolling off the sea. Style: vivid chalk pastel illustration with deep 3D depth, rich layered chalk texture, crisp chalk lines, cinematic 16:9, no people, no text.",
        height=160,
    )
    colA, colB = st.columns([1, 1])
    with colA:
        preview_btn = st.button("Preview 1", type="primary")
    with colB:
        run_btn = st.button("Generate Batch")

    if ref_img is not None:
        ref_bytes = ref_img.read()
        st.image(ref_bytes, caption="Reference image", use_container_width=True)
    else:
        ref_bytes = None

    if preview_btn:
        try:
            img_bytes = _stability_generate_one(
                api_key=api_key,
                endpoint=endpoint,
                prompt=prompt_text,
                negative_prompt=negative_prompt,
                aspect_ratio=aspect_ratio,
                output_format=output_format,
                seed=int(seed) if seed is not None else None,
                reference_image_bytes=ref_bytes,
                image_denoise=float(image_denoise),
            )
            img = _bytes_to_pil(img_bytes)
            st.image(img, caption="Preview", use_container_width=True)
        except Exception as e:
            st.error(str(e))

    if run_btn:
        named: List[Tuple[str, bytes]] = []
        progress = st.progress(0)
        gallery = st.container()

        for i in range(variants):
            try:
                img_bytes = _stability_generate_one(
                    api_key=api_key,
                    endpoint=endpoint,
                    prompt=prompt_text,
                    negative_prompt=negative_prompt,
                    aspect_ratio=aspect_ratio,
                    output_format=output_format,
                    seed=(int(seed) + i) if seed is not None else None,
                    reference_image_bytes=ref_bytes,
                    image_denoise=float(image_denoise),
                )
                fname = f"single_v{i+1:02d}.{output_format}"
                named.append((fname, img_bytes))
            except Exception as e:
                st.error(f"Variant {i+1} failed: {e}")

            progress.progress((i + 1) / variants)

        if named:
            cols = st.columns(4)
            for idx, (fname, b) in enumerate(named):
                with cols[idx % 4]:
                    st.image(_bytes_to_pil(b), caption=fname, use_container_width=True)

            zip_bytes = _zip_images(named)
            st.download_button(
                "Download ZIP",
                data=zip_bytes,
                file_name="outpaged_images.zip",
                mime="application/zip",
            )


# -------------------------
# Tab 2: Upload doc + parse backgrounds
# -------------------------
with tab2:
    st.subheader("Upload Doc and Batch Generate Backgrounds")
    st.caption("Upload a .docx or .txt that contains lines like 'Background file name: ch01bg01' followed by the prompt.")

    uploaded = st.file_uploader("Upload .docx or .txt", type=["docx", "txt"], accept_multiple_files=False)
    if uploaded:
        try:
            raw = _read_uploaded_text(uploaded)
            items = _parse_background_items(raw)

            if not items:
                st.warning("No background prompts found. Make sure the doc includes 'Background file name:' lines.")
            else:
                st.success(f"Found {len(items)} background prompts.")
                file_names = [it.file_base for it in items]

                selected = st.multiselect(
                    "Select scenes to generate",
                    options=file_names,
                    default=file_names[: min(10, len(file_names))],
                )

                preview_name = st.selectbox("Preview which one?", options=selected if selected else file_names)
                preview_item = next((it for it in items if it.file_base == preview_name), items[0])

                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Preview Selected", type="primary"):
                        ref_bytes = ref_img.read() if ref_img is not None else None
                        try:
                            img_bytes = _stability_generate_one(
                                api_key=api_key,
                                endpoint=endpoint,
                                prompt=preview_item.prompt,
                                negative_prompt=negative_prompt,
                                aspect_ratio=aspect_ratio,
                                output_format=output_format,
                                seed=int(seed) if seed is not None else None,
                                reference_image_bytes=ref_bytes,
                                image_denoise=float(image_denoise),
                            )
                            st.image(_bytes_to_pil(img_bytes), caption=preview_item.file_base, use_container_width=True)
                        except Exception as e:
                            st.error(str(e))

                with col2:
                    run_selected = st.button("Generate All Selected")

                if run_selected and selected:
                    ref_bytes = ref_img.read() if ref_img is not None else None

                    chosen_items = [it for it in items if it.file_base in selected]
                    named: List[Tuple[str, bytes]] = []

                    total = len(chosen_items) * variants
                    done = 0
                    progress = st.progress(0)

                    gallery_cols = st.columns(4)
                    thumb_count = 0

                    for it in chosen_items:
                        for v in range(variants):
                            try:
                                img_bytes = _stability_generate_one(
                                    api_key=api_key,
                                    endpoint=endpoint,
                                    prompt=it.prompt,
                                    negative_prompt=negative_prompt,
                                    aspect_ratio=aspect_ratio,
                                    output_format=output_format,
                                    seed=(int(seed) + v) if seed is not None else None,
                                    reference_image_bytes=ref_bytes,
                                    image_denoise=float(image_denoise),
                                )
                                fname = f"{it.file_base}_v{v+1:02d}.{output_format}"
                                named.append((fname, img_bytes))

                                # Show a few thumbnails so you see it working
                                if thumb_count < 12:
                                    with gallery_cols[thumb_count % 4]:
                                        st.image(_bytes_to_pil(img_bytes), caption=fname, use_container_width=True)
                                    thumb_count += 1

                            except Exception as e:
                                st.error(f"{it.file_base} v{v+1} failed: {e}")

                            done += 1
                            progress.progress(done / max(1, total))

                    if named:
                        zip_bytes = _zip_images(named)
                        st.download_button(
                            "Download ZIP",
                            data=zip_bytes,
                            file_name="outpaged_backgrounds.zip",
                            mime="application/zip",
                        )

        except Exception as e:
            st.error(str(e))


# -------------------------
# Tab 3: Character training pack
# -------------------------
with tab3:
    st.subheader("Character Training Pack")
    st.caption("Paste one character prompt, pick a name, then generate neutral body + 3 face angles.")

    char_name = st.text_input("Character name", value="Lemuel Gulliver")
    char_prompt = st.text_area(
        "Character master prompt",
        value=(
            "Lemuel Gulliver, an English ship’s surgeon and seasoned traveler in the early 1700s, "
            "lean build and weathered seafarer’s face, medium-length dark hair, clean-shaven or faint stubble, "
            "wearing period-accurate voyager clothing: dark long coat, waistcoat, linen shirt, neck cloth, breeches, "
            "stockings, buckled shoes, with a leather satchel and nautical wear and tear appropriate to harsh voyages. "
            "Realistic charcoal drawing with subtle colored chalk accents (muted blues, reds, ochres), textured paper grain, "
            "high detail, lifelike eyes and shading, slightly glossy highlights, full body cutout, no background, no shadow, no text."
        ),
        height=180,
    )

    st.info("Tip: For character training, use aspect ratio 1:1 in the sidebar.")

    colX, colY = st.columns([1, 1])
    with colX:
        preview_pack = st.button("Preview Pack (first image)", type="primary")
    with colY:
        run_pack = st.button("Generate Full Pack")

    ref_bytes = ref_img.read() if ref_img is not None else None

    pack_items = _make_character_training_prompts(char_name, char_prompt)

    st.write("Pack will generate:")
    for it in pack_items:
        st.code(f"{it.file_base}: {it.prompt[:120]}...", language="text")

    if preview_pack:
        try:
            img_bytes = _stability_generate_one(
                api_key=api_key,
                endpoint=endpoint,
                prompt=pack_items[0].prompt,
                negative_prompt=negative_prompt,
                aspect_ratio=aspect_ratio,
                output_format=output_format,
                seed=int(seed) if seed is not None else None,
                reference_image_bytes=ref_bytes,
                image_denoise=float(image_denoise),
            )
            st.image(_bytes_to_pil(img_bytes), caption=pack_items[0].file_base, use_container_width=True)
        except Exception as e:
            st.error(str(e))

    if run_pack:
        named: List[Tuple[str, bytes]] = []
        total = len(pack_items) * variants
        done = 0
        progress = st.progress(0)

        cols = st.columns(4)
        shown = 0

        for it in pack_items:
            for v in range(variants):
                try:
                    img_bytes = _stability_generate_one(
                        api_key=api_key,
                        endpoint=endpoint,
                        prompt=it.prompt,
                        negative_prompt=negative_prompt,
                        aspect_ratio=aspect_ratio,
                        output_format=output_format,
                        seed=(int(seed) + v) if seed is not None else None,
                        reference_image_bytes=ref_bytes,
                        image_denoise=float(image_denoise),
                    )
                    fname = f"{it.file_base}_v{v+1:02d}.{output_format}"
                    named.append((fname, img_bytes))

                    if shown < 8:
                        with cols[shown % 4]:
                            st.image(_bytes_to_pil(img_bytes), caption=fname, use_container_width=True)
                        shown += 1

                except Exception as e:
                    st.error(f"{it.file_base} v{v+1} failed: {e}")

                done += 1
                progress.progress(done / max(1, total))

        if named:
            zip_bytes = _zip_images(named)
            st.download_button(
                "Download ZIP",
                data=zip_bytes,
                file_name=f"{re.sub(r'[^a-zA-Z0-9_-]+','_',char_name).lower()}_training_pack.zip",
                mime="application/zip",
            )
