import os
import io
import re
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import streamlit as st
from PIL import Image

from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

from docx import Document


# =========================
# Helpers
# =========================

def _get_stability_key() -> str:
    # Prefer Streamlit Secrets, fallback to env var
    key = ""
    try:
        key = st.secrets.get("STABILITY_KEY", "")
    except Exception:
        key = ""
    if not key:
        key = os.getenv("STABILITY_KEY", "")
    return (key or "").strip()


def _init_stability(engine: str, key: str) -> client.StabilityInference:
    # Host can be set in secrets or env, default to official
    host = ""
    try:
        host = st.secrets.get("STABILITY_HOST", "")
    except Exception:
        host = ""
    if not host:
        host = os.getenv("STABILITY_HOST", "grpc.stability.ai")

    os.environ["STABILITY_HOST"] = host
    os.environ["STABILITY_KEY"] = key

    return client.StabilityInference(
        key=key,
        verbose=False,
        engine=engine,
    )


def _build_weighted_prompts(positive_text: str, negative_text: str) -> List[generation.Prompt]:
    """
    stability-sdk expects prompt weights via PromptParameters(weight=...),
    not generation.Prompt(..., weight=...).
    """
    positive_text = (positive_text or "").strip()
    negative_text = (negative_text or "").strip()

    prompts: List[generation.Prompt] = []
    if positive_text:
        prompts.append(
            generation.Prompt(
                text=positive_text,
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


def _docx_to_text(file_bytes: bytes) -> str:
    bio = io.BytesIO(file_bytes)
    doc = Document(bio)
    parts = []
    for p in doc.paragraphs:
        t = (p.text or "").rstrip()
        if t:
            parts.append(t)
    return "\n".join(parts).strip()


def _bytes_to_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    raw = uploaded_file.read()
    name = (uploaded_file.name or "").lower()
    if name.endswith(".docx"):
        return _docx_to_text(raw)
    # assume text
    try:
        return raw.decode("utf-8")
    except Exception:
        return raw.decode("latin-1", errors="ignore")


@dataclass
class ScenePrompt:
    scene_title: str
    file_name: str
    prompt: str


def _parse_scenes(raw_text: str) -> List[ScenePrompt]:
    """
    Parses blocks like:
    Scene 01 ...
    Background file name: ch01bg01
    <prompt...>

    It is tolerant to extra lines.
    """
    txt = (raw_text or "").strip()
    if not txt:
        return []

    # Split on "Scene" headings
    chunks = re.split(r"\n(?=Scene\s+\d+)", txt, flags=re.IGNORECASE)
    scenes: List[ScenePrompt] = []

    for chunk in chunks:
        c = chunk.strip()
        if not c.lower().startswith("scene"):
            continue

        # File name
        m_fn = re.search(r"Background file name:\s*([A-Za-z0-9_\-]+)", c, flags=re.IGNORECASE)
        if not m_fn:
            continue
        file_name = m_fn.group(1).strip()

        # Title: first non-empty line after "Scene X"
        lines = [ln.strip() for ln in c.splitlines() if ln.strip()]
        scene_title = lines[0] if lines else file_name

        # Prompt: take text after the background file name line
        # Find the line index containing "Background file name:"
        prompt_text = ""
        for i, ln in enumerate(lines):
            if re.search(r"^Background file name:", ln, flags=re.IGNORECASE):
                prompt_text = "\n".join(lines[i + 1 :]).strip()
                break

        # Stop prompt if it runs into "Canonical Master Prompts"
        prompt_text = re.split(r"\nCanonical Master Prompts", prompt_text, flags=re.IGNORECASE)[0].strip()

        if prompt_text:
            scenes.append(ScenePrompt(scene_title=scene_title, file_name=file_name, prompt=prompt_text))

    return scenes


def _parse_canonical_characters(raw_text: str) -> Dict[str, str]:
    """
    Parses Canonical Master Prompts section into a dict: name -> prompt.
    Assumes format:
    Canonical Master Prompts
    <optional lines>
    Character Name
    Character prompt paragraph...
    Next Character Name
    ...
    """
    txt = (raw_text or "")
    m = re.search(r"Canonical Master Prompts(.+)$", txt, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return {}

    body = m.group(1).strip()
    if not body:
        return {}

    lines = [ln.rstrip() for ln in body.splitlines()]
    # Remove "Use these master prompts..." helper line(s)
    cleaned = []
    for ln in lines:
        if not ln.strip():
            cleaned.append("")
            continue
        if re.search(r"Use these master prompts", ln, flags=re.IGNORECASE):
            continue
        cleaned.append(ln)

    # Heuristic: a character name is a non-empty line with no comma and short-ish
    # then prompt is following paragraph(s) until next name line.
    chars: Dict[str, str] = {}
    i = 0
    while i < len(cleaned):
        name = cleaned[i].strip()
        if not name:
            i += 1
            continue

        # probable name line
        is_name = (len(name) <= 60) and ("," not in name) and (not name.lower().startswith("use these"))
        if not is_name:
            i += 1
            continue

        # collect prompt lines after name until next probable name
        j = i + 1
        prompt_lines = []
        while j < len(cleaned):
            nxt = cleaned[j].strip()
            if nxt and (len(nxt) <= 60) and ("," not in nxt) and (not nxt.lower().startswith("use these")):
                # looks like next name
                break
            prompt_lines.append(cleaned[j])
            j += 1

        prompt = "\n".join([p for p in prompt_lines if p.strip()]).strip()
        if prompt:
            chars[name] = prompt

        i = j

    return chars


def _safe_filename(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "asset"


def _generate_images(
    stability_api: client.StabilityInference,
    positive_text: str,
    negative_text: str,
    width: int,
    height: int,
    samples: int,
    steps: int,
    cfg_scale: float,
    seed: Optional[int],
    init_image: Optional[Image.Image],
    start_schedule: Optional[float],
    use_style_preset: bool,
    style_preset: str,
) -> List[Image.Image]:
    """
    Returns list of PIL Images.
    """
    if not positive_text.strip():
        return []

    prompt_payload = positive_text
    prompts = _build_weighted_prompts(positive_text, negative_text)
    if len(prompts) >= 2:
        # For multi-prompting (including negative prompt), pass prompt as list of generation.Prompt
        prompt_payload = prompts

    kwargs = dict(
        prompt=prompt_payload,
        samples=int(samples),
        steps=int(steps),
        cfg_scale=float(cfg_scale),
        width=int(width),
        height=int(height),
    )

    if seed is not None:
        kwargs["seed"] = int(seed)

    if init_image is not None:
        kwargs["init_image"] = init_image
        # start_schedule controls how much the init_image influences the result
        # lower values keep more of the init_image; 1.0 means text-only
        if start_schedule is not None:
            kwargs["start_schedule"] = float(start_schedule)

    # Only pass style_preset if you explicitly want it.
    if use_style_preset and style_preset:
        kwargs["style_preset"] = style_preset

    answers = stability_api.generate(**kwargs)

    out: List[Image.Image] = []
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                # safety filter hit
                continue
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary)).convert("RGB")
                out.append(img)
    return out


def _zip_images(named_images: List[Tuple[str, Image.Image]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, img in named_images:
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            zf.writestr(name, img_bytes.getvalue())
    return buf.getvalue()


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="OutPaged Visualizer", layout="wide")
st.title("📚 OutPaged Visualizer")
st.caption("Generate backgrounds, characters, and training packs from your prompts. Built for StoryTech AI workflows.")

stability_key = _get_stability_key()

with st.sidebar:
    st.header("Connection")
    if stability_key:
        st.success("STABILITY_KEY found")
    else:
        st.error("Missing STABILITY_KEY (Streamlit Secrets or env var)")

    st.header("Generation Settings")
    engine = st.selectbox(
        "Engine",
        [
            "stable-diffusion-xl-1024-v1-0",
            "stable-diffusion-v1-6",
        ],
        index=0,
    )

    steps = st.slider("Steps", 10, 60, 30)
    cfg_scale = st.slider("CFG Scale", 1.0, 15.0, 7.0, 0.5)
    seed_input = st.text_input("Seed (optional)", value="")
    seed = None
    if seed_input.strip():
        try:
            seed = int(seed_input.strip())
        except Exception:
            st.warning("Seed must be an integer or blank.")
            seed = None

    st.header("Style Preset (optional)")
    use_style_preset = st.checkbox(
        "Use style_preset (can override prompt style)",
        value=False,
        help="If off, your prompt text fully controls style.",
    )
    style_preset = st.selectbox(
        "Preset",
        ["cinematic", "fantasy-art", "photographic", "comic-book", "pixel-art"],
        index=0,
        disabled=(not use_style_preset),
    )

    st.header("Reference Image (optional)")
    ref_file = st.file_uploader("Upload reference image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    init_image = None
    if ref_file is not None:
        init_image = Image.open(ref_file).convert("RGB")
        st.image(init_image, caption="Reference image", use_container_width=True)

    start_schedule = None
    if init_image is not None:
        start_schedule = st.slider(
            "Init influence (start_schedule)",
            0.1, 1.0, 0.6, 0.05,
            help="Lower = more like the reference image. 1.0 = text-only.",
        )


st.divider()

tab_bg, tab_char = st.tabs(["🌄 Backgrounds (Scenes)", "🧍 Characters (Training Pack)"])

# ---------- Backgrounds ----------
with tab_bg:
    st.subheader("Background generation from Scene prompts")

    colA, colB = st.columns([1, 1])

    with colA:
        uploaded_doc = st.file_uploader("Upload a DOCX or TXT with Scene prompts", type=["docx", "txt"], key="bg_doc")
        raw_text = _bytes_to_text(uploaded_doc) if uploaded_doc else ""
        pasted = st.text_area(
            "Or paste prompts here",
            value=raw_text,
            height=280,
            placeholder="Paste your Scene blocks here...",
        )

        negative_text = st.text_input(
            "Negative prompt (optional)",
            value="text, watermark, logo, signature, blurry, low quality",
            help="This SDK does not accept negative_prompt=..., so we apply it via weighted prompts.",
        )

    with colB:
        st.markdown("**Output size**")
        ratio = st.selectbox("Aspect ratio", ["16:9 (cinematic)", "1:1 (square)"], index=0)
        if ratio.startswith("16:9"):
            width, height = 1024, 576
        else:
            width, height = 1024, 1024

        samples = st.slider("Images per scene", 1, 12, 2)
        preview_only = st.checkbox("Preview mode (only 1 scene)", value=True)

        st.info("Tip: Put your full style in the prompt (ex: chalk pastel, textured paper, crisp chalk lines). Leave style_preset off.")

    scenes = _parse_scenes(pasted)
    if not scenes:
        st.warning("No scenes parsed yet. Make sure your text contains 'Background file name: ...' lines.")
    else:
        st.write(f"Parsed scenes: **{len(scenes)}**")

        # Scene picker
        scene_labels = [f"{s.file_name} | {s.scene_title}" for s in scenes]
        picked = st.multiselect(
            "Select scenes to generate",
            options=scene_labels,
            default=[scene_labels[0]] if scene_labels else [],
        )

        selected_scenes = [scenes[i] for i, lab in enumerate(scene_labels) if lab in picked]

        col1, col2 = st.columns([1, 1])
        with col1:
            go_preview = st.button("Generate preview", type="primary", disabled=(not stability_key or not selected_scenes))
        with col2:
            go_batch = st.button("Generate batch", disabled=(not stability_key or not selected_scenes))

        if (go_preview or go_batch) and stability_key:
            stability_api = _init_stability(engine=engine, key=stability_key)

            run_scenes = selected_scenes[:1] if (preview_only or go_preview) else selected_scenes

            named_images: List[Tuple[str, Image.Image]] = []
            prog = st.progress(0)
            total = len(run_scenes)

            for idx, scene in enumerate(run_scenes, start=1):
                st.markdown(f"### {scene.file_name}")
                st.code(scene.prompt, language="text")

                imgs = _generate_images(
                    stability_api=stability_api,
                    positive_text=scene.prompt,
                    negative_text=negative_text,
                    width=width,
                    height=height,
                    samples=samples if go_batch else 1,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    seed=seed,
                    init_image=init_image,
                    start_schedule=start_schedule,
                    use_style_preset=use_style_preset,
                    style_preset=style_preset,
                )

                if not imgs:
                    st.error("No images returned (possible safety filter or API error). Check Streamlit logs.")
                else:
                    cols = st.columns(min(4, len(imgs)))
                    for i, img in enumerate(imgs):
                        fname = f"{scene.file_name}_v{str(i+1).zfill(2)}.png"
                        named_images.append((fname, img))
                        with cols[i % len(cols)]:
                            st.image(img, caption=fname, use_container_width=True)

                prog.progress(int((idx / total) * 100))

            if named_images:
                zip_bytes = _zip_images(named_images)
                st.download_button(
                    "Download ZIP",
                    data=zip_bytes,
                    file_name="outpaged_backgrounds.zip",
                    mime="application/zip",
                )


# ---------- Characters ----------
with tab_char:
    st.subheader("Character training pack: full body + 3 face angles")

    uploaded_doc_c = st.file_uploader("Upload a DOCX or TXT with Canonical Master Prompts", type=["docx", "txt"], key="char_doc")
    raw_text_c = _bytes_to_text(uploaded_doc_c) if uploaded_doc_c else ""

    pasted_c = st.text_area(
        "Or paste Canonical Master Prompts here",
        value=raw_text_c,
        height=220,
        placeholder="Paste the Canonical Master Prompts section here...",
    )

    chars = _parse_canonical_characters(pasted_c)
    if not chars:
        st.warning("No canonical characters parsed yet. Paste the 'Canonical Master Prompts' section (with names + paragraphs).")
        character_name = st.text_input("Character name (manual)", value="Custom Character")
        character_prompt = st.text_area("Character prompt (manual)", value="", height=180)
    else:
        character_name = st.selectbox("Pick a character", options=list(chars.keys()))
        character_prompt = chars[character_name]
        st.code(character_prompt, language="text")

    negative_text_c = st.text_input(
        "Negative prompt (optional)",
        value="text, watermark, logo, signature, blurry, low quality, extra fingers, deformed",
    )

    st.markdown("**Pack outputs**")
    pack = st.multiselect(
        "Select outputs",
        ["Full body neutral pose", "Face front (straight)", "Face 45 degrees", "Face profile"],
        default=["Full body neutral pose", "Face front (straight)", "Face 45 degrees", "Face profile"],
    )

    samples_c = st.slider("Images per output", 1, 8, 2)
    width_c, height_c = 1024, 1024

    run_char_preview = st.button("Generate character preview", type="primary", disabled=(not stability_key))
    run_char_batch = st.button("Generate character pack", disabled=(not stability_key))

    if (run_char_preview or run_char_batch) and stability_key:
        if not character_prompt.strip():
            st.error("Character prompt is empty.")
        else:
            stability_api = _init_stability(engine=engine, key=stability_key)

            # Build prompt variants
            variants: List[Tuple[str, str]] = []
            base = character_prompt.strip()

            safe_base_name = _safe_filename(character_name)

            if "Full body neutral pose" in pack:
                variants.append((
                    f"{safe_base_name}_fullbody",
                    base + "\nNeutral standing pose, arms relaxed, full body centered, consistent proportions, clean silhouette."
                ))
            if "Face front (straight)" in pack:
                variants.append((
                    f"{safe_base_name}_face_front",
                    base + "\nTight head-and-shoulders portrait, face straight toward camera, neutral expression, high detail eyes."
                ))
            if "Face 45 degrees" in pack:
                variants.append((
                    f"{safe_base_name}_face_45",
                    base + "\nTight head-and-shoulders portrait, face turned 45 degrees, neutral expression, consistent facial structure."
                ))
            if "Face profile" in pack:
                variants.append((
                    f"{safe_base_name}_face_profile",
                    base + "\nTight head-and-shoulders portrait, full profile view, neutral expression, clear nose and jawline."
                ))

            named_images: List[Tuple[str, Image.Image]] = []
            prog = st.progress(0)
            total = len(variants)

            for idx, (stem, prompt_txt) in enumerate(variants, start=1):
                st.markdown(f"### {stem}")
                st.code(prompt_txt, language="text")

                imgs = _generate_images(
                    stability_api=stability_api,
                    positive_text=prompt_txt,
                    negative_text=negative_text_c,
                    width=width_c,
                    height=height_c,
                    samples=(1 if run_char_preview else samples_c),
                    steps=steps,
                    cfg_scale=cfg_scale,
                    seed=seed,
                    init_image=init_image,
                    start_schedule=start_schedule,
                    use_style_preset=use_style_preset,
                    style_preset=style_preset,
                )

                if not imgs:
                    st.error("No images returned (possible safety filter or API error). Check Streamlit logs.")
                else:
                    cols = st.columns(min(4, len(imgs)))
                    for i, img in enumerate(imgs):
                        fname = f"{stem}_v{str(i+1).zfill(2)}.png"
                        named_images.append((fname, img))
                        with cols[i % len(cols)]:
                            st.image(img, caption=fname, use_container_width=True)

                prog.progress(int((idx / total) * 100))

            if named_images:
                zip_bytes = _zip_images(named_images)
                st.download_button(
                    "Download ZIP",
                    data=zip_bytes,
                    file_name="outpaged_character_pack.zip",
                    mime="application/zip",
                )
