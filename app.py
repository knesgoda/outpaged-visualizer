import os
import io
import re
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import streamlit as st
from PIL import Image, ImageDraw

from docx import Document

from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation


# =========================
# Config
# =========================
APP_TITLE = "📚 OutPaged Visualizer"
DEFAULT_ENGINE = "stable-diffusion-xl-1024-v1-0"
DEFAULT_STYLE_PRESET = "cinematic"

# gRPC endpoint commonly used by the SDK (the README references the production endpoint). :contentReference[oaicite:2]{index=2}
os.environ.setdefault("STABILITY_HOST", "grpc.stability.ai:443")


# =========================
# Data models
# =========================
@dataclass
class ScenePrompt:
    filename: str
    title: str
    prompt: str


@dataclass
class CharacterPrompt:
    name: str
    prompt: str


# =========================
# Helpers: reading uploads
# =========================
def _read_txt(uploaded_file) -> str:
    return uploaded_file.read().decode("utf-8", errors="replace")


def _read_docx(uploaded_file) -> str:
    # Streamlit uploaded_file is file-like; python-docx needs a path or file-like bytes
    data = uploaded_file.read()
    bio = io.BytesIO(data)
    doc = Document(bio)
    parts = []
    for p in doc.paragraphs:
        txt = (p.text or "").strip()
        if txt:
            parts.append(txt)
    return "\n".join(parts)


def read_uploaded_text(uploaded_file) -> str:
    name = (uploaded_file.name or "").lower()
    if name.endswith(".txt"):
        return _read_txt(uploaded_file)
    if name.endswith(".docx"):
        return _read_docx(uploaded_file)
    raise ValueError("Unsupported file type. Please upload a .txt or .docx file.")


# =========================
# Helpers: parsing prompts
# =========================
def parse_scenes_from_text(text: str) -> List[ScenePrompt]:
    """
    Looks for blocks containing:
      - "Background file name:" <filename>
      - Prompt text (until next Scene / next Background file name / end)

    Works with your pasted format.
    """
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    scenes: List[ScenePrompt] = []

    current_title = ""
    current_filename = ""
    collecting_prompt: List[str] = []
    in_prompt = False

    def flush():
        nonlocal current_title, current_filename, collecting_prompt, in_prompt
        if current_filename and collecting_prompt:
            prompt = " ".join([p.strip() for p in collecting_prompt if p.strip()]).strip()
            if prompt:
                scenes.append(ScenePrompt(filename=current_filename.strip(),
                                          title=current_title.strip() or current_filename.strip(),
                                          prompt=prompt))
        current_title = ""
        current_filename = ""
        collecting_prompt = []
        in_prompt = False

    for ln in lines:
        stripped = ln.strip()

        # Start of a new scene section
        if re.match(r"^Scene\s+\d+", stripped, flags=re.IGNORECASE):
            flush()
            current_title = stripped
            continue

        # Sometimes you have a second line with richer title
        if stripped.lower().startswith("scene ") and "-" in stripped:
            # Example: "Scene 01 - Part I - ..."
            current_title = stripped
            continue

        # Filename marker
        if stripped.lower().startswith("background file name:"):
            flush()
            current_filename = stripped.split(":", 1)[1].strip()
            in_prompt = True
            continue

        # Stop conditions if we hit another filename marker etc are handled by flush on new markers
        if in_prompt:
            if stripped == "":
                # allow blank lines inside prompt blocks; but if prompt already collecting, keep going
                # We'll just ignore empties.
                continue
            collecting_prompt.append(stripped)

    flush()
    return scenes


def parse_characters_from_text(text: str) -> List[CharacterPrompt]:
    """
    Parses "Canonical Master Prompts" style blocks like:
      Lemuel Gulliver
      Lemuel Gulliver, an English ship’s surgeon ...

    Heuristic:
      - A "name line" is short-ish, has no commas, and is not a header.
      - Everything after is prompt until next name line.
    """
    raw_lines = [ln.rstrip() for ln in (text or "").splitlines()]
    lines = [ln.strip() for ln in raw_lines if ln.strip()]

    headers = {
        "canonical master prompts",
        "use these master prompts to keep characters consistent across scenes.",
    }

    chars: List[CharacterPrompt] = []
    current_name = ""
    current_prompt: List[str] = []

    def is_name_line(s: str) -> bool:
        sl = s.lower()
        if sl in headers:
            return False
        if sl.startswith("scene "):
            return False
        if sl.startswith("background file name"):
            return False
        if len(s) > 60:
            return False
        if "," in s:
            return False
        # Title case-ish names are common, but do not enforce
        return True

    def flush():
        nonlocal current_name, current_prompt
        if current_name and current_prompt:
            prompt = " ".join([p.strip() for p in current_prompt if p.strip()]).strip()
            if prompt:
                chars.append(CharacterPrompt(name=current_name.strip(), prompt=prompt))
        current_name = ""
        current_prompt = []

    for ln in lines:
        if is_name_line(ln):
            flush()
            current_name = ln
            continue
        if current_name:
            current_prompt.append(ln)

    flush()
    return chars


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return s or "item"


# =========================
# Helpers: Stability SDK prompt building
# =========================
def build_weighted_prompts(prompt_text: str, negative_text: str) -> List[generation.Prompt]:
    """
    stability-sdk weights live under PromptParameters(weight=...). :contentReference[oaicite:3]{index=3}
    Negative prompting is done with negative weights (multi-prompting). :contentReference[oaicite:4]{index=4}
    """
    prompt_text = (prompt_text or "").strip()
    negative_text = (negative_text or "").strip()

    prompts: List[generation.Prompt] = []
    if prompt_text:
        prompts.append(
            generation.Prompt(
                text=prompt_text,
                parameters=generation.PromptParameters(weight=1.0),
            )
        )
    if negative_text:
        prompts.append(
            generation.Prompt(
                text=negative_text,
                parameters=generation.PromptParameters(weight=-1.0),
            )
        )
    return prompts


# =========================
# Helpers: init image
# =========================
def build_sample_init_image(width: int, height: int) -> Image.Image:
    """
    Simple built-in sample so you can test img2img without uploading.
    """
    img = Image.new("RGB", (width, height), (245, 242, 235))
    d = ImageDraw.Draw(img)
    # soft frame
    d.rectangle([10, 10, width - 10, height - 10], outline=(180, 175, 165), width=4)
    # subtle diagonal strokes
    step = max(20, min(width, height) // 20)
    for i in range(-height, width, step):
        d.line([(i, 0), (i + height, height)], fill=(225, 220, 210), width=2)
    d.text((24, 24), "OutPaged sample init image", fill=(90, 85, 75))
    return img


# =========================
# Stability client
# =========================
def get_stability_key_from_secrets_or_env() -> str:
    key = ""
    try:
        # Streamlit Cloud: set this in App -> Settings -> Secrets
        key = st.secrets.get("STABILITY_KEY", "")
    except Exception:
        key = ""

    if not key:
        key = os.getenv("STABILITY_KEY", "") or os.getenv("STABILITY_API_KEY", "")

    return key.strip()


@st.cache_resource
def get_stability_client(api_key: str, engine: str):
    return client.StabilityInference(
        key=api_key,
        verbose=False,
        engine=engine,
    )


def generate_images(
    stability_api,
    prompt_text: str,
    negative_text: str,
    samples: int,
    steps: int,
    cfg_scale: float,
    width: int,
    height: int,
    style_preset: str,
    seed: Optional[int],
    init_image: Optional[Image.Image],
    start_schedule: float,
) -> List[Tuple[str, bytes]]:
    """
    Returns list of (suggested_filename, png_bytes)
    """
    weighted_prompts = build_weighted_prompts(prompt_text, negative_text)

    kwargs = dict(
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        samples=samples,
        style_preset=style_preset,
    )

    if seed is not None:
        kwargs["seed"] = int(seed)

    if init_image is not None:
        # The SDK supports passing an init image for image-to-image flows. :contentReference[oaicite:5]{index=5}
        kwargs["init_image"] = init_image
        # Start schedule is commonly used with init images (0..1). (Docs/examples refer to start_schedule with init_image.) :contentReference[oaicite:6]{index=6}
        kwargs["start_schedule"] = float(start_schedule)

    # Prefer multi-prompting via `prompts=...` (SDK uses PromptParameters weights). :contentReference[oaicite:7]{index=7}
    try:
        answers = stability_api.generate(prompts=weighted_prompts, **kwargs)
    except TypeError:
        # Fallback: if this engine/SDK build does not accept prompts=,
        # run without negative prompt.
        answers = stability_api.generate(prompt=prompt_text, **kwargs)

    out: List[Tuple[str, bytes]] = []
    idx = 0

    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                raise ValueError("Safety filter triggered. Try adjusting the prompt.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                png_bytes = artifact.binary
                idx += 1
                out.append((f"image_{idx}.png", png_bytes))

    return out


def make_zip(files: List[Tuple[str, bytes]]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in files:
            z.writestr(name, data)
    return bio.getvalue()


# =========================
# UI
# =========================
st.set_page_config(page_title="OutPaged Visualizer", layout="wide")
st.title(APP_TITLE)
st.caption("Generate batch concepts for AR/XR book assets (backgrounds, characters, training sets).")

# API key handling
api_key = get_stability_key_from_secrets_or_env()
if not api_key:
    st.warning("No Stability API key found in Streamlit Secrets or environment.")
    api_key_input = st.text_input("Paste Stability API Key (not saved)", type="password")
    api_key = (api_key_input or "").strip()

if not api_key:
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header("Generation Settings")

    engine = st.selectbox("Engine", [DEFAULT_ENGINE], index=0)
    style_preset = st.selectbox(
        "Style preset",
        ["cinematic", "fantasy-art", "photographic", "comic-book", "digital-art", "3d-model"],
        index=0 if DEFAULT_STYLE_PRESET == "cinematic" else 0,
    )

    aspect = st.selectbox(
        "Aspect ratio",
        ["16:9 (1024x576)", "1:1 (1024x1024)", "4:5 (896x1120)", "9:16 (576x1024)"],
        index=0,
    )
    if aspect.startswith("16:9"):
        width, height = 1024, 576
    elif aspect.startswith("1:1"):
        width, height = 1024, 1024
    elif aspect.startswith("4:5"):
        width, height = 896, 1120
    else:
        width, height = 576, 1024

    steps = st.slider("Steps", 10, 60, 30)
    cfg_scale = st.slider("CFG scale", 1.0, 14.0, 7.0, 0.5)
    seed = st.number_input("Seed (optional)", min_value=0, max_value=2_147_483_647, value=0, step=1)
    use_seed = st.checkbox("Use seed", value=False)

    negative_text = st.text_area(
        "Negative prompt (optional)",
        "low quality, blurry, watermark, text, logo, extra fingers, deformed",
        height=90,
    )

    st.divider()
    st.subheader("Reference image (optional)")

    ref_upload = st.file_uploader("Upload reference image (png/jpg)", type=["png", "jpg", "jpeg"])
    use_sample_init = st.checkbox("Use built-in sample init image", value=False)
    start_schedule = st.slider("Init strength (start_schedule)", 0.0, 1.0, 0.65, 0.01)

# Build Stability client
stability_api = get_stability_client(api_key, engine)

# Resolve init image
init_image: Optional[Image.Image] = None
if ref_upload is not None:
    init_image = Image.open(ref_upload).convert("RGB")
elif use_sample_init:
    init_image = build_sample_init_image(width, height)

if init_image is not None:
    st.sidebar.image(init_image, caption="Init image", use_container_width=True)

seed_value = int(seed) if use_seed else None

tabs = st.tabs(["Backgrounds", "Characters", "Quick Prompt"])


# =========================
# TAB: Backgrounds
# =========================
with tabs[0]:
    st.subheader("Background generation")

    uploaded_bg = st.file_uploader("Upload background prompt doc (.txt or .docx)", type=["txt", "docx"], key="bg_doc")
    pasted_bg = st.text_area(
        "Or paste background prompt text here",
        height=220,
        placeholder="Paste your Scene blocks here...",
        key="bg_paste",
    )

    source_text = ""
    if uploaded_bg is not None:
        try:
            source_text = read_uploaded_text(uploaded_bg)
        except Exception as e:
            st.error(f"Could not read file: {type(e).__name__}: {e}")
            st.stop()
    else:
        source_text = pasted_bg or ""

    scenes = parse_scenes_from_text(source_text) if source_text.strip() else []
    if not scenes:
        st.info("Upload or paste background prompts to begin.")
    else:
        st.write(f"Found {len(scenes)} background prompts.")

        options = [f"{s.filename} | {s.title}" for s in scenes]
        chosen = st.multiselect("Select scenes to generate", options, default=options[:1])

        preview_only = st.checkbox("Preview only (1 image per scene)", value=True)
        per_scene = 1 if preview_only else st.slider("Images per scene", 1, 12, 4)

        colA, colB = st.columns([1, 1])
        with colA:
            go = st.button("Generate selected backgrounds", type="primary")
        with colB:
            st.caption("Tip: keep preview on until the style looks right, then turn it off for batch.")

        if go and chosen:
            all_files: List[Tuple[str, bytes]] = []
            for label in chosen:
                filename = label.split("|", 1)[0].strip()
                scene = next((s for s in scenes if s.filename == filename), None)
                if not scene:
                    continue

                st.markdown(f"### {scene.title}")
                st.code(scene.prompt)

                try:
                    imgs = generate_images(
                        stability_api=stability_api,
                        prompt_text=scene.prompt,
                        negative_text=negative_text,
                        samples=per_scene,
                        steps=steps,
                        cfg_scale=cfg_scale,
                        width=width,
                        height=height,
                        style_preset=style_preset,
                        seed=seed_value,
                        init_image=init_image,
                        start_schedule=start_schedule,
                    )
                except Exception as e:
                    st.error(f"Generation failed: {type(e).__name__}: {e}")
                    continue

                # Show grid
                cols = st.columns(4)
                for i, (tmp_name, png_bytes) in enumerate(imgs):
                    out_name = f"{scene.filename}_v{i+1:02d}.png"
                    all_files.append((out_name, png_bytes))
                    with cols[i % 4]:
                        st.image(png_bytes, caption=out_name, use_container_width=True)

            if all_files:
                zip_bytes = make_zip(all_files)
                st.download_button(
                    "Download backgrounds ZIP",
                    data=zip_bytes,
                    file_name="outpaged_backgrounds.zip",
                    mime="application/zip",
                )


# =========================
# TAB: Characters
# =========================
with tabs[1]:
    st.subheader("Character training set generation")
    st.caption("Generates 4 images per character: neutral full-body, face front, face 45°, face profile.")

    uploaded_ch = st.file_uploader("Upload character prompt doc (.txt or .docx)", type=["txt", "docx"], key="ch_doc")
    pasted_ch = st.text_area(
        "Or paste character master prompt text here",
        height=220,
        placeholder="Paste Canonical Master Prompts here...",
        key="ch_paste",
    )

    ch_text = ""
    if uploaded_ch is not None:
        try:
            ch_text = read_uploaded_text(uploaded_ch)
        except Exception as e:
            st.error(f"Could not read file: {type(e).__name__}: {e}")
            st.stop()
    else:
        ch_text = pasted_ch or ""

    chars = parse_characters_from_text(ch_text) if ch_text.strip() else []
    if not chars:
        st.info("Upload or paste character master prompts to begin.")
    else:
        st.write(f"Found {len(chars)} characters.")
        char_names = [c.name for c in chars]
        selected_chars = st.multiselect("Select characters", char_names, default=char_names[:1])

        st.markdown("#### Training prompt templates")
        full_body_suffix = st.text_input(
            "Full-body suffix",
            "Neutral standing pose, arms relaxed at sides, full body cutout, no background, no shadow, no text.",
        )
        face_front_suffix = st.text_input(
            "Face front suffix",
            "Tight face close-up, straight-on, neutral expression, consistent identity, no background, no text.",
        )
        face_45_suffix = st.text_input(
            "Face 45° suffix",
            "Tight face close-up, 45 degree angle, neutral expression, consistent identity, no background, no text.",
        )
        face_profile_suffix = st.text_input(
            "Face profile suffix",
            "Tight face close-up, profile view, neutral expression, consistent identity, no background, no text.",
        )

        preview_only = st.checkbox("Preview only (1 image per view)", value=True, key="ch_preview")
        per_view = 1 if preview_only else st.slider("Images per view", 1, 8, 2, key="ch_per_view")

        go = st.button("Generate training sets", type="primary", key="ch_go")

        if go and selected_chars:
            all_files: List[Tuple[str, bytes]] = []

            for name in selected_chars:
                ch = next((c for c in chars if c.name == name), None)
                if not ch:
                    continue

                st.markdown(f"### {ch.name}")
                base = ch.prompt.strip()
                slug = slugify(ch.name)

                variants = [
                    ("body_neutral", f"{base} {full_body_suffix}"),
                    ("face_front", f"{base} {face_front_suffix}"),
                    ("face_45", f"{base} {face_45_suffix}"),
                    ("face_profile", f"{base} {face_profile_suffix}"),
                ]

                for tag, prompt in variants:
                    st.markdown(f"**{tag}**")
                    st.code(prompt)

                    try:
                        imgs = generate_images(
                            stability_api=stability_api,
                            prompt_text=prompt,
                            negative_text=negative_text,
                            samples=per_view,
                            steps=steps,
                            cfg_scale=cfg_scale,
                            width=width,
                            height=height,
                            style_preset=style_preset,
                            seed=seed_value,
                            init_image=init_image,
                            start_schedule=start_schedule,
                        )
                    except Exception as e:
                        st.error(f"Generation failed: {type(e).__name__}: {e}")
                        continue

                    cols = st.columns(4)
                    for i, (tmp_name, png_bytes) in enumerate(imgs):
                        out_name = f"{slug}_{tag}_v{i+1:02d}.png"
                        all_files.append((out_name, png_bytes))
                        with cols[i % 4]:
                            st.image(png_bytes, caption=out_name, use_container_width=True)

            if all_files:
                zip_bytes = make_zip(all_files)
                st.download_button(
                    "Download character training ZIP",
                    data=zip_bytes,
                    file_name="outpaged_character_training.zip",
                    mime="application/zip",
                )


# =========================
# TAB: Quick Prompt
# =========================
with tabs[2]:
    st.subheader("Quick prompt")
    st.caption("Use this to test style quickly without parsing a document.")

    prompt_text = st.text_area(
        "Prompt",
        "Rocky coastline after a violent storm, shattered mast and sailcloth tangled in seaweed, wet sand with oversized human footprints leading inland, cold dawn fog rolling off the sea. Style: vivid chalk pastel illustration with deep 3D depth, rich layered chalk texture, crisp chalk lines, cinematic 16:9, no people, no text.",
        height=160,
    )

    preview_only = st.checkbox("Preview only (1 image)", value=True, key="qp_preview")
    count = 1 if preview_only else st.slider("Batch size", 1, 20, 4, key="qp_count")

    if st.button("Generate", type="primary", key="qp_go"):
        try:
            imgs = generate_images(
                stability_api=stability_api,
                prompt_text=prompt_text,
                negative_text=negative_text,
                samples=count,
                steps=steps,
                cfg_scale=cfg_scale,
                width=width,
                height=height,
                style_preset=style_preset,
                seed=seed_value,
                init_image=init_image,
                start_schedule=start_schedule,
            )
        except Exception as e:
            st.error(f"Generation failed: {type(e).__name__}: {e}")
            st.stop()

        cols = st.columns(4)
        files = []
        for i, (tmp_name, png_bytes) in enumerate(imgs):
            out_name = f"quick_v{i+1:02d}.png"
            files.append((out_name, png_bytes))
            with cols[i % 4]:
                st.image(png_bytes, caption=out_name, use_container_width=True)

        if files:
            st.download_button(
                "Download quick ZIP",
                data=make_zip(files),
                file_name="outpaged_quick.zip",
                mime="application/zip",
            )
