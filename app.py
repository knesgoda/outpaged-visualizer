import io
import os
import re
import json
import zipfile
import zlib
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image

try:
    from docx import Document
except Exception:
    Document = None


# =========================
# Stability API config
# =========================
STABILITY_API_BASE = "https://api.stability.ai"

ENDPOINTS = {
    "Stable Image Ultra (best quality)": "/v2beta/stable-image/generate/ultra",
    "Stable Image Core (faster)": "/v2beta/stable-image/generate/core",
}

ASPECT_RATIOS = ["16:9", "1:1", "3:2", "2:3", "4:5", "5:4", "9:16"]
OUTPUT_FORMATS = ["png", "webp", "jpg"]

# Seed range per Stability docs (commonly used range)
SEED_MOD = 2**32  # 0..4294967295


# =========================
# Data structures
# =========================
@dataclass
class PromptItem:
    file_base: str  # ch36bg01
    label: str      # display label
    prompt: str     # prompt text


# =========================
# API key + file helpers
# =========================
def _get_api_key() -> Optional[str]:
    if hasattr(st, "secrets"):
        k = st.secrets.get("STABILITY_API_KEY", None)
        if k:
            return str(k).strip()

    k = os.getenv("STABILITY_API_KEY", "").strip()
    if k:
        return k
    k = os.getenv("STABILITY_KEY", "").strip()
    if k:
        return k
    return None


def _read_uploaded_text(uploaded_file) -> str:
    name = (uploaded_file.name or "").lower()

    # IMPORTANT: use getvalue() so it is stable across reruns
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()

    if name.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore")

    if name.endswith(".docx"):
        if Document is None:
            raise RuntimeError("python-docx is not installed. Check requirements.txt.")
        doc = Document(io.BytesIO(raw))
        return "\n".join([p.text for p in doc.paragraphs])

    raise ValueError("Unsupported file type. Upload .txt or .docx")


def _safe_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s)
    return s.strip("_") or "image"


# =========================
# Per-scene seed logic
# =========================
def _crc32_int(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def _scene_seed(base_seed: Optional[int], scene_key: str, variant_index: int) -> Optional[int]:
    """
    If base_seed is None -> random seed (return None)
    If base_seed is set -> make unique deterministic seed per scene + variant
    """
    if base_seed is None:
        return None
    # Make each scene different but reproducible
    return int((base_seed + _crc32_int(scene_key) + variant_index) % SEED_MOD)


# =========================
# Parsing your scene format
# =========================
def _parse_background_items(raw_text: str) -> List[PromptItem]:
    lines = [ln.rstrip() for ln in raw_text.splitlines()]
    items: List[PromptItem] = []

    current_scene_label = ""
    current_scene_title = ""

    i = 0
    while i < len(lines):
        ln = lines[i].strip()

        if ln.lower().startswith("canonical master prompts"):
            break

        if re.match(r"^scene\s+\d+", ln, flags=re.IGNORECASE):
            current_scene_label = ln
            current_scene_title = ""
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                nxt = lines[j].strip()
                if not nxt.lower().startswith("background file name:") and not re.match(r"^scene\s+\d+", nxt, flags=re.IGNORECASE):
                    current_scene_title = nxt
            i += 1
            continue

        if ln.lower().startswith("background file name:"):
            file_base = ln.split(":", 1)[1].strip()
            file_base = _safe_filename(file_base)

            prompt_lines = []
            j = i + 1
            while j < len(lines):
                cur = lines[j].strip()

                if cur.lower().startswith("canonical master prompts"):
                    break

                if cur == "":
                    if prompt_lines:
                        break
                    j += 1
                    continue

                if re.match(r"^scene\s+\d+", cur, flags=re.IGNORECASE):
                    break
                if cur.lower().startswith("background file name:"):
                    break

                prompt_lines.append(cur)
                j += 1

            prompt = " ".join(prompt_lines).strip()
            label = (current_scene_title or current_scene_label or file_base).strip()

            if file_base and prompt:
                items.append(PromptItem(file_base=file_base, label=label, prompt=prompt))

            i = j
            continue

        i += 1

    return items


# =========================
# Stability call
# =========================
def _stability_generate(
    api_key: str,
    endpoint_path: str,
    prompt: str,
    negative_prompt: str,
    aspect_ratio: str,
    output_format: str,
    seed: Optional[int],
    reference_image_bytes: Optional[bytes],
    strength: float,
    timeout_s: int = 120,
) -> bytes:
    url = f"{STABILITY_API_BASE}{endpoint_path}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "image/*",
    }

    data = {
        "prompt": prompt,
        "output_format": output_format,
        "aspect_ratio": aspect_ratio,
        "mode": "image-to-image" if reference_image_bytes is not None else "text-to-image",
    }

    if negative_prompt.strip():
        data["negative_prompt"] = negative_prompt.strip()

    if seed is not None:
        data["seed"] = str(int(seed))

    files = None
    if reference_image_bytes is not None:
        files = {"image": ("reference.png", reference_image_bytes, "image/png")}
        data["strength"] = str(float(strength))  # required when 'image' is provided

    resp = requests.post(url, headers=headers, data=data, files=files, timeout=timeout_s)

    if resp.status_code == 200:
        return resp.content

    try:
        err = resp.json()
    except Exception:
        err = {"error": resp.text}

    raise RuntimeError(f"Stability API error {resp.status_code}: {json.dumps(err)}")


def _zip_images(named_images: List[Tuple[str, bytes]], manifest: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        for name, b in named_images:
            zf.writestr(name, b)
    return buf.getvalue()


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="OutPaged Visualizer", layout="wide")
st.title("📚 OutPaged Visualizer")
st.caption("Upload your prompt doc, then generate 1 image per Background file name (or more variants).")

api_key = _get_api_key()
if not api_key:
    st.warning(
        "Add your Stability key in Streamlit Cloud Secrets as:\n\n"
        'STABILITY_API_KEY = "sk-..."\n'
    )
    st.stop()

with st.sidebar:
    st.header("Settings")

    model_choice = st.selectbox("Model", list(ENDPOINTS.keys()), index=0, key="sb_model")
    endpoint_path = ENDPOINTS[model_choice]

    aspect_ratio = st.selectbox("Aspect ratio", ASPECT_RATIOS, index=0, key="sb_aspect")
    output_format = st.selectbox("Output format", OUTPUT_FORMATS, index=0, key="sb_format")

    variants_per_prompt = st.slider("Variants per prompt", 1, 5, 1, key="sb_variants")

    seed_mode = st.selectbox("Seed mode", ["Fixed (reproducible)", "Random"], index=0, key="sb_seed_mode")
    fixed_seed = (
        st.number_input("Base seed", min_value=0, max_value=2_147_483_647, value=123456, step=1, key="sb_seed")
        if seed_mode.startswith("Fixed")
        else None
    )

    st.divider()
    st.subheader("Reference image (optional)")
    ref_img = st.file_uploader("Upload reference image", type=["png", "jpg", "jpeg"], key="sb_ref_img")
    strength = st.slider(
        "Strength (only used if reference image provided)",
        0.05, 0.95, 0.35, 0.05,
        help="Lower = closer to reference. Higher = more reimagined.",
        key="sb_strength",
    )

    st.divider()
    st.subheader("Negative prompt")
    negative_prompt = st.text_area(
        "Negative prompt",
        value="text, watermark, logo, signature, low quality, blurry, malformed",
        height=100,
        key="sb_negative",
    )

# Reference bytes (use getvalue to avoid read pointer issues)
ref_bytes = None
if ref_img is not None:
    ref_bytes = ref_img.getvalue() if hasattr(ref_img, "getvalue") else ref_img.read()

tab_doc, tab_single = st.tabs(["Batch from Doc (Scenes)", "Single Prompt"])


# ===== TAB: DOC BATCH =====
with tab_doc:
    st.subheader("Batch from Doc")
    st.caption("This generates one image per scene file name. Your filenames are preserved.")

    uploaded = st.file_uploader("Upload your prompt doc (.docx or .txt)", type=["docx", "txt"], key="doc_uploader")

    if uploaded:
        raw_text = _read_uploaded_text(uploaded)
        items = _parse_background_items(raw_text)

        if not items:
            st.error("No scenes found. Make sure your doc includes lines like 'Background file name: ch36bg01'.")
        else:
            st.success(f"Found {len(items)} scenes (background prompts).")

            # Quick sanity check: show first few parsed prompts
            with st.expander("Show parsed prompts (sanity check)"):
                st.dataframe(
                    [{"file": it.file_base, "label": it.label, "prompt_preview": it.prompt[:140]} for it in items[:25]],
                    use_container_width=True,
                )

            all_keys = [it.file_base for it in items]

            if "selected_scene_keys" not in st.session_state:
                st.session_state["selected_scene_keys"] = all_keys

            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("Select all", use_container_width=True, key="doc_select_all"):
                    st.session_state["selected_scene_keys"] = all_keys
            with col2:
                if st.button("Select none", use_container_width=True, key="doc_select_none"):
                    st.session_state["selected_scene_keys"] = []

            selected = st.multiselect(
                "Scenes to generate",
                options=all_keys,
                default=st.session_state["selected_scene_keys"],
                key="doc_scene_multiselect",
            )
            st.session_state["selected_scene_keys"] = selected

            total_outputs = len(selected) * int(variants_per_prompt)
            st.info(f"Outputs to generate: {len(selected)} scenes × {variants_per_prompt} variants = {total_outputs} images")

            selected_items = [it for it in items if it.file_base in selected]
            preview_options = [it.file_base for it in selected_items] if selected_items else all_keys

            preview_key = st.selectbox("Preview which scene?", options=preview_options, key="doc_preview_scene")
            preview_item = next(it for it in items if it.file_base == preview_key)

            cA, cB = st.columns([1, 1])
            with cA:
                preview_btn = st.button("Preview 1", type="secondary", use_container_width=True, key="doc_preview_btn")
            with cB:
                gen_btn = st.button("Generate full batch", type="primary", use_container_width=True, key="doc_generate_btn")

            if preview_btn:
                try:
                    seed_for_preview = _scene_seed(fixed_seed, preview_item.file_base, 0)
                    img_bytes = _stability_generate(
                        api_key=api_key,
                        endpoint_path=endpoint_path,
                        prompt=preview_item.prompt,
                        negative_prompt=negative_prompt,
                        aspect_ratio=aspect_ratio,
                        output_format=output_format,
                        seed=seed_for_preview,
                        reference_image_bytes=ref_bytes,
                        strength=float(strength),
                    )
                    st.image(
                        Image.open(io.BytesIO(img_bytes)),
                        caption=f"{preview_item.file_base}.{output_format} (seed={seed_for_preview})",
                        use_container_width=True,
                    )
                    st.caption(preview_item.label)
                    st.code(preview_item.prompt, language="text")
                except Exception as e:
                    st.error(str(e))

            if gen_btn:
                if not selected_items:
                    st.warning("Select at least one scene.")
                else:
                    named_images: List[Tuple[str, bytes]] = []
                    rows_for_manifest = []

                    progress = st.progress(0)
                    status = st.empty()

                    total = len(selected_items) * int(variants_per_prompt)
                    done = 0

                    thumbs_to_show = 12
                    thumbs_shown = 0
                    grid = st.columns(4)

                    for it in selected_items:
                        for v in range(int(variants_per_prompt)):
                            status.write(f"Generating {done+1} of {total}: {it.file_base}")

                            # 🔥 KEY FIX: unique deterministic seed per scene (+ variant)
                            seed = _scene_seed(fixed_seed, it.file_base, v)

                            img_bytes = _stability_generate(
                                api_key=api_key,
                                endpoint_path=endpoint_path,
                                prompt=it.prompt,
                                negative_prompt=negative_prompt,
                                aspect_ratio=aspect_ratio,
                                output_format=output_format,
                                seed=seed,
                                reference_image_bytes=ref_bytes,
                                strength=float(strength),
                            )

                            if int(variants_per_prompt) == 1:
                                fname = f"{it.file_base}.{output_format}"
                            else:
                                fname = f"{it.file_base}_v{v+1:02d}.{output_format}"

                            named_images.append((fname, img_bytes))
                            rows_for_manifest.append({
                                "file": fname,
                                "file_base": it.file_base,
                                "label": it.label,
                                "prompt": it.prompt,
                                "variant": v + 1,
                                "seed": seed,
                            })

                            if thumbs_shown < thumbs_to_show:
                                with grid[thumbs_shown % 4]:
                                    st.image(
                                        Image.open(io.BytesIO(img_bytes)),
                                        caption=f"{fname}",
                                        use_container_width=True,
                                    )
                                    st.caption(f"{it.label}")
                                thumbs_shown += 1

                            done += 1
                            progress.progress(done / max(1, total))

                    status.empty()
                    st.success(f"Generated {len(named_images)} images.")

                    manifest = {
                        "created_at_utc": datetime.utcnow().isoformat() + "Z",
                        "model": model_choice,
                        "endpoint": endpoint_path,
                        "aspect_ratio": aspect_ratio,
                        "output_format": output_format,
                        "variants_per_prompt": int(variants_per_prompt),
                        "seed_mode": seed_mode,
                        "base_seed": int(fixed_seed) if fixed_seed is not None else None,
                        "has_reference_image": bool(ref_img),
                        "strength": float(strength) if ref_img else None,
                        "negative_prompt": negative_prompt,
                        "items": rows_for_manifest,
                    }

                    zip_bytes = _zip_images(named_images, manifest)

                    st.download_button(
                        "Download ZIP (images + manifest.json)",
                        data=zip_bytes,
                        file_name=f"outpaged_scenes_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True,
                        key="doc_download_zip",
                    )

                    st.subheader("Generated file list")
                    st.dataframe(
                        [{"file": r["file"], "label": r["label"], "seed": r["seed"]} for r in rows_for_manifest],
                        use_container_width=True,
                    )


# ===== TAB: SINGLE PROMPT =====
with tab_single:
    st.subheader("Single Prompt")
    st.caption("Generate variants of one prompt.")

    prompt_text = st.text_area(
        "Prompt",
        value="Laputan observatory walkway: chalk-marked geometric diagrams on flagstones, star charts pinned to rails, a dangling flapper’s bladder on a stick left resting against a post, and brass instruments scattered under cold sky light. Style: vivid chalk pastel illustration with deep 3D depth, rich layered chalk texture, crisp chalk lines, cinematic 16:9, no people, no text.",
        height=150,
        key="single_prompt_text",
    )

    c1, c2 = st.columns([1, 1])
    preview_btn = c1.button("Preview 1", type="secondary", use_container_width=True, key="single_preview_btn")
    batch_btn = c2.button("Generate batch", type="primary", use_container_width=True, key="single_generate_btn")

    if preview_btn:
        try:
            seed_for_preview = _scene_seed(fixed_seed, prompt_text, 0)
            img_bytes = _stability_generate(
                api_key=api_key,
                endpoint_path=endpoint_path,
                prompt=prompt_text,
                negative_prompt=negative_prompt,
                aspect_ratio=aspect_ratio,
                output_format=output_format,
                seed=seed_for_preview,
                reference_image_bytes=ref_bytes,
                strength=float(strength),
            )
            st.image(
                Image.open(io.BytesIO(img_bytes)),
                caption=f"preview.{output_format} (seed={seed_for_preview})",
                use_container_width=True,
            )
        except Exception as e:
            st.error(str(e))

    if batch_btn:
        named_images: List[Tuple[str, bytes]] = []
        grid = st.columns(4)
        progress = st.progress(0)

        for v in range(int(variants_per_prompt)):
            seed = _scene_seed(fixed_seed, prompt_text, v)
            try:
                img_bytes = _stability_generate(
                    api_key=api_key,
                    endpoint_path=endpoint_path,
                    prompt=prompt_text,
                    negative_prompt=negative_prompt,
                    aspect_ratio=aspect_ratio,
                    output_format=output_format,
                    seed=seed,
                    reference_image_bytes=ref_bytes,
                    strength=float(strength),
                )
                fname = f"single_v{v+1:02d}.{output_format}"
                named_images.append((fname, img_bytes))

                with grid[v % 4]:
                    st.image(
                        Image.open(io.BytesIO(img_bytes)),
                        caption=f"{fname} (seed={seed})",
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Variant {v+1} failed: {e}")

            progress.progress((v + 1) / int(variants_per_prompt))

        if named_images:
            manifest = {
                "created_at_utc": datetime.utcnow().isoformat() + "Z",
                "model": model_choice,
                "endpoint": endpoint_path,
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
                "variants_per_prompt": int(variants_per_prompt),
                "prompt": prompt_text,
                "negative_prompt": negative_prompt,
                "seed_mode": seed_mode,
                "base_seed": int(fixed_seed) if fixed_seed is not None else None,
            }
            zip_bytes = _zip_images(named_images, manifest)
            st.download_button(
                "Download ZIP",
                data=zip_bytes,
                file_name=f"outpaged_single_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                use_container_width=True,
                key="single_download_zip",
            )
