import io
import os
import re
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image

from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation


# =========================
# Helpers
# =========================

def _get_cfg() -> Tuple[str, str, str]:
    """
    Reads config from Streamlit Secrets first, then environment variables.
    """
    host = st.secrets.get("STABILITY_HOST", os.environ.get("STABILITY_HOST", "grpc.stability.ai:443"))
    key = st.secrets.get("STABILITY_KEY", os.environ.get("STABILITY_KEY", "")).strip()
    engine = st.secrets.get("STABILITY_ENGINE", os.environ.get("STABILITY_ENGINE", "stable-diffusion-xl-1024-v1-0"))
    return host, key, engine


@st.cache_resource(show_spinner=False)
def _get_stability_client(host: str, key: str, engine: str) -> client.StabilityInference:
    # Stability SDK reads these env vars internally too.
    os.environ["STABILITY_HOST"] = host
    os.environ["STABILITY_KEY"] = key

    return client.StabilityInference(
        key=key,
        verbose=False,
        engine=engine,
    )


def _build_weighted_prompts(prompt_text: str, negative_text: str) -> List[generation.Prompt]:
    prompts = [generation.Prompt(text=prompt_text, weight=1.0)]
    neg = (negative_text or "").strip()
    if neg:
        prompts.append(generation.Prompt(text=neg, weight=-1.0))
    return prompts


def _generate_images(
    stability_api: client.StabilityInference,
    prompt_text: str,
    negative_text: str,
    samples: int,
    steps: int,
    cfg_scale: float,
    width: int,
    height: int,
    style_preset: str,
) -> List[Tuple[Image.Image, Optional[int]]]:
    """
    Returns list of (PIL Image, seed).
    Uses weighted prompts to support negative prompts without negative_prompt=.
    """
    prompts = _build_weighted_prompts(prompt_text, negative_text)

    results: List[Tuple[Image.Image, Optional[int]]] = []
    answers = stability_api.generate(
        prompt=prompts,
        samples=samples,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        style_preset=style_preset,
    )

    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                st.warning("Safety filter triggered for one output. Try adjusting the prompt.")
                continue

            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary)).convert("RGBA")
                seed = getattr(artifact, "seed", None)
                results.append((img, seed))

    return results


def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "asset"


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _zip_bytes(named_pngs: List[Tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in named_pngs:
            z.writestr(name, data)
    return buf.getvalue()


BACKGROUND_BLOCK_RE = re.compile(
    r"Background file name:\s*(?P<fname>\S+)\s*\n(?P<prompt>.*?)(?=\n\s*\nScene\s+\d+|\n\s*\nCanonical|\Z)",
    re.IGNORECASE | re.DOTALL
)


def _parse_background_doc(text: str) -> List[Dict[str, str]]:
    """
    Extracts:
      - file_name from "Background file name:"
      - prompt from the text that follows until next Scene / Canonical / EOF

    Returns list of {file_name, prompt, label}.
    """
    items: List[Dict[str, str]] = []
    if not (text or "").strip():
        return items

    for m in BACKGROUND_BLOCK_RE.finditer(text):
        fname = m.group("fname").strip()
        prompt = m.group("prompt").strip()

        # Light cleanup: collapse internal whitespace
        prompt = re.sub(r"[ \t]+\n", "\n", prompt)
        prompt = re.sub(r"\n{3,}", "\n\n", prompt)

        # Try to capture a nearby scene label by searching backwards a bit
        start = max(0, m.start() - 250)
        context = text[start:m.start()]
        scene_label = ""
        scene_match = re.search(r"(Scene\s+\d+.*?)(?:\n|$)", context, re.IGNORECASE)
        if scene_match:
            scene_label = scene_match.group(1).strip()

        label = f"{fname}" + (f" | {scene_label}" if scene_label else "")
        items.append({"file_name": fname, "prompt": prompt, "label": label})

    return items


def _require_auth():
    app_pw = st.secrets.get("APP_PASSWORD", "").strip()
    if not app_pw:
        return

    if "authed" not in st.session_state:
        st.session_state["authed"] = False

    if not st.session_state["authed"]:
        st.subheader("🔒 Locked")
        pw = st.text_input("Password", type="password")
        if st.button("Unlock"):
            st.session_state["authed"] = (pw == app_pw)
        if not st.session_state["authed"]:
            st.stop()


# =========================
# App UI
# =========================

st.set_page_config(page_title="OutPaged Visualizer", layout="wide")
st.title("📚 OutPaged Visualizer")
st.caption("Preview and batch-generate background concepts + character training images.")

_require_auth()

host, key, engine = _get_cfg()
if not key:
    st.error("Missing STABILITY_KEY. Add it in Streamlit Advanced settings → Secrets.")
    st.stop()

stability_api = _get_stability_client(host, key, engine)

with st.sidebar:
    st.header("Settings")

    style_preset = st.selectbox(
        "Style preset",
        ["cinematic", "fantasy-art", "photographic", "comic-book"],
        index=0
    )

    steps = st.slider("Steps", 10, 50, 30)
    cfg_scale = st.slider("CFG scale", 1.0, 15.0, 7.0, 0.5)

    size_mode = st.selectbox(
        "Output size",
        ["Background 16:9 (1024x576)", "Square (1024x1024)", "Portrait (768x1024)"],
        index=0
    )

    if size_mode == "Background 16:9 (1024x576)":
        width, height = 1024, 576
    elif size_mode == "Portrait (768x1024)":
        width, height = 768, 1024
    else:
        width, height = 1024, 1024

    st.divider()
    negative_prompt = st.text_area(
        "Negative prompt (optional)",
        value="blurry, low detail, watermark, text, logo, extra limbs, bad anatomy",
        height=90
    )


tab_bg, tab_char = st.tabs(["Backgrounds", "Characters"])


# =========================
# Backgrounds tab
# =========================
with tab_bg:
    st.subheader("Background generation")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 1) Load prompts")
        st.write("Paste your background document (the Scenes list) below and click Parse.")
        bg_text = st.text_area("Background prompt document", height=260)

        parse_btn = st.button("Parse backgrounds", type="secondary")

        parsed_items: List[Dict[str, str]] = []
        if parse_btn or ("bg_items" in st.session_state and not bg_text.strip() == ""):
            parsed_items = _parse_background_doc(bg_text)
            st.session_state["bg_items"] = parsed_items

        if "bg_items" in st.session_state:
            parsed_items = st.session_state["bg_items"]

        if parsed_items:
            labels = [x["label"] for x in parsed_items]
            selected_label = st.selectbox("Select scene to preview", labels, index=0)
            selected = next(x for x in parsed_items if x["label"] == selected_label)
            preview_prompt = selected["prompt"]
            file_base = selected["file_name"]
        else:
            st.info("No parsed scenes yet. You can still use manual prompt mode.")
            preview_prompt = st.text_area(
                "Manual background prompt",
                value="Rocky coastline after a violent storm... Style: vivid chalk pastel illustration...",
                height=140
            )
            file_base = st.text_input("File base (used for naming)", value="bg_preview")

    with col_right:
        st.markdown("### 2) Preview")
        st.write("Generate a single image to confirm look and feel.")

        preview_btn = st.button("Generate preview", type="primary")

        if preview_btn:
            with st.spinner("Generating preview..."):
                imgs = _generate_images(
                    stability_api=stability_api,
                    prompt_text=preview_prompt,
                    negative_text=negative_prompt,
                    samples=1,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    width=width,
                    height=height,
                    style_preset=style_preset,
                )

            if not imgs:
                st.error("No image returned. Try adjusting prompt or settings.")
            else:
                img, seed = imgs[0]
                st.image(img, caption=f"Preview {file_base} | seed={seed}")
                png = _png_bytes(img)

                st.download_button(
                    "Download preview PNG",
                    data=png,
                    file_name=f"{file_base}_preview.png",
                    mime="image/png"
                )

    st.divider()
    st.markdown("### 3) Batch generate")

    batch_cols = st.columns([1, 1, 2])
    with batch_cols[0]:
        batch_n = st.slider("Batch size", 1, 20, 4)
    with batch_cols[1]:
        naming_mode = st.selectbox("Naming", ["Use file base", "Auto timestamp"], index=0)

    gen_batch_btn = st.button("Generate batch", type="primary")

    if gen_batch_btn:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        prefix = file_base if naming_mode == "Use file base" else f"bg_{ts}"

        with st.spinner(f"Generating {batch_n} images..."):
            imgs = _generate_images(
                stability_api=stability_api,
                prompt_text=preview_prompt,
                negative_text=negative_prompt,
                samples=batch_n,
                steps=steps,
                cfg_scale=cfg_scale,
                width=width,
                height=height,
                style_preset=style_preset,
            )

        if not imgs:
            st.error("No images returned. Try adjusting prompt or settings.")
        else:
            gallery_cols = st.columns(4)
            zip_payload: List[Tuple[str, bytes]] = []

            for i, (img, seed) in enumerate(imgs, start=1):
                name = f"{prefix}_{i:02d}.png"
                data = _png_bytes(img)
                zip_payload.append((name, data))

                with gallery_cols[(i - 1) % 4]:
                    st.image(img, caption=f"{name} | seed={seed}")
                    st.download_button(
                        "Download",
                        data=data,
                        file_name=name,
                        mime="image/png",
                        key=f"dl_bg_{name}_{seed}_{i}"
                    )

            z = _zip_bytes(zip_payload)
            st.download_button(
                "Download ALL as ZIP",
                data=z,
                file_name=f"{prefix}_batch.zip",
                mime="application/zip"
            )


# =========================
# Characters tab
# =========================
with tab_char:
    st.subheader("Character training set generator")

    st.write("This generates: 1 full-body neutral + 3 face angles (front, 45°, profile).")

    char_col1, char_col2 = st.columns([1, 1])

    with char_col1:
        char_name = st.text_input("Character name", value="Lemuel Gulliver")
        char_file_base = st.text_input("Character file base (used for naming)", value="gulliver")

        char_master_prompt = st.text_area(
            "Character master prompt (full description)",
            height=260,
            value=(
                "Lemuel Gulliver, an English ship’s surgeon and seasoned traveler in the early 1700s, "
                "lean build and weathered seafarer’s face, medium-length dark hair, clean-shaven or faint stubble, "
                "wearing period-accurate voyager clothing: dark long coat, waistcoat, linen shirt, neck cloth, "
                "breeches, stockings, buckled shoes, with a leather satchel and nautical wear and tear appropriate "
                "to harsh voyages. Realistic charcoal drawing with subtle colored chalk accents (muted blues, reds, ochres), "
                "textured paper grain, high detail, lifelike eyes and shading, slightly glossy highlights, full body cutout, "
                "no background, no shadow, no text."
            )
        )

    with char_col2:
        st.markdown("### Preview")
        st.write("Preview the neutral full-body image first.")

        preview_char_btn = st.button("Preview neutral full-body", type="primary")

        if preview_char_btn:
            prompt_full = (
                f"{char_master_prompt}\n"
                "Neutral standing pose, arms relaxed at sides, feet shoulder-width apart, centered composition."
            )

            with st.spinner("Generating character preview..."):
                imgs = _generate_images(
                    stability_api=stability_api,
                    prompt_text=prompt_full,
                    negative_text=negative_prompt,
                    samples=1,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    width=1024,
                    height=1024,
                    style_preset=style_preset,
                )

            if imgs:
                img, seed = imgs[0]
                st.image(img, caption=f"{char_name} | neutral full-body | seed={seed}")
                data = _png_bytes(img)
                st.download_button(
                    "Download preview PNG",
                    data=data,
                    file_name=f"{_slugify(char_file_base)}_neutral_full_preview.png",
                    mime="image/png",
                    key="dl_char_preview"
                )
            else:
                st.error("No image returned. Try adjusting prompt or settings.")

    st.divider()
    st.markdown("### Generate full training set")

    gen_set_btn = st.button("Generate 4-image training set", type="primary")

    if gen_set_btn:
        base = _slugify(char_file_base)

        prompts = [
            ("neutral_full", f"{char_master_prompt}\nNeutral standing pose, arms relaxed at sides, centered, full body."),
            ("face_front", f"{char_master_prompt}\nClose-up face portrait, straight-on front view, head and shoulders, centered."),
            ("face_45", f"{char_master_prompt}\nClose-up face portrait, 45-degree three-quarter view, head and shoulders, centered."),
            ("face_profile", f"{char_master_prompt}\nClose-up face portrait, side profile view, head and shoulders, centered."),
        ]

        out_files: List[Tuple[str, bytes]] = []
        gallery = st.columns(4)

        with st.spinner("Generating training images..."):
            for idx, (suffix, p) in enumerate(prompts, start=1):
                imgs = _generate_images(
                    stability_api=stability_api,
                    prompt_text=p,
                    negative_text=negative_prompt,
                    samples=1,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    width=1024,
                    height=1024,
                    style_preset=style_preset,
                )

                if not imgs:
                    st.warning(f"Missing output for {suffix}. Try again or adjust settings.")
                    continue

                img, seed = imgs[0]
                fname = f"{base}_{suffix}.png"
                data = _png_bytes(img)
                out_files.append((fname, data))

                with gallery[(idx - 1) % 4]:
                    st.image(img, caption=f"{fname} | seed={seed}")
                    st.download_button(
                        "Download",
                        data=data,
                        file_name=fname,
                        mime="image/png",
                        key=f"dl_{fname}_{seed}_{idx}"
                    )

        if out_files:
            z = _zip_bytes(out_files)
            st.download_button(
                "Download training set ZIP",
                data=z,
                file_name=f"{base}_training_set.zip",
                mime="application/zip",
                key="dl_training_zip"
            )
