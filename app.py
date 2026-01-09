import base64
import io
import os
import re
import time
import zipfile

import requests
import streamlit as st
from docx import Document

BLOCKADE_BASE = "https://backend.blockadelabs.com/api/v1"
STABILITY_CORE_URL = "https://api.stability.ai/v2beta/stable-image/generate/core"

st.set_page_config(page_title="OutPaged Visualizer", layout="wide")
st.title("OutPaged Visualizer")
st.caption("Bulk-generate backgrounds, character training sets, and skyboxes from your DOCX prompts.")

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
        files["image"] = ("init.png", init_image_bytes, "image/png")

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

def read_docx_text(uploaded_file) -> str:
    document = Document(uploaded_file)
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)

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
                items.append(
                    {
                        "filename": current_name,
                        "prompt": "\n".join(l for l in current_lines if l.strip()).strip(),
                    }
                )
            current_name = match.group("filename").strip()
            current_lines = []
        elif current_name:
            current_lines.append(line)
    if current_name:
        items.append(
            {
                "filename": current_name,
                "prompt": "\n".join(l for l in current_lines if l.strip()).strip(),
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
        st.dataframe(
            [{"filename": item["filename"], "prompt": item["prompt"][:120]} for item in bg_items],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No background entries parsed yet. Ensure your DOCX includes 'Background file name: <name>'.")

    bg_negative = st.text_input(
        "Negative prompt",
        "text, watermark, logo, low quality, blurry",
        key="background_negative",
    )
    bg_aspect = st.selectbox(
        "Aspect ratio",
        ["1:1", "2:3", "3:2", "9:16", "16:9"],
        index=4,
        key="background_aspect",
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
        stability_key = stability_api_key()
        items_to_run = bg_items[:1] if preview_bg else bg_items
        all_outputs: list[tuple[str, bytes]] = []
        init_bytes = bg_ref_image.getvalue() if bg_ref_image else None
        with st.status("Generating backgrounds…", expanded=True) as status:
            try:
                for idx, item in enumerate(items_to_run):
                    seed = build_seed(bg_seed, idx)
                    status.write(f"Generating {item['filename']}…")
                    images = stability_generate_images(
                        prompt=item["prompt"],
                        negative_prompt=bg_negative.strip(),
                        seed=seed,
                        aspect_ratio=bg_aspect,
                        image_count=bg_count,
                        api_key=stability_key,
                        init_image_bytes=init_bytes,
                        init_strength=bg_strength if init_bytes else None,
                    )
                    for image_index, image_bytes in enumerate(images, start=1):
                        if bg_count > 1:
                            filename = f"{item['filename']}_{image_index:02d}.png"
                        else:
                            filename = f"{item['filename']}.png"
                        all_outputs.append((filename, image_bytes))
                        st.image(image_bytes, caption=filename, use_container_width=True)

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
        st.dataframe(
            [{"filename": item["filename"], "prompt": item["prompt"][:120]} for item in char_items],
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
        "text, watermark, logo, blurry, low quality, cropped, extra limbs",
        key="char_negative",
    )
    char_aspect = st.selectbox(
        "Aspect ratio",
        ["1:1", "2:3", "3:2", "9:16", "16:9"],
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
        stability_key = stability_api_key()
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
                    for variant_index, (variant_key, variant_suffix) in enumerate(variants):
                        seed_offset = item_index * 100 + variant_index * 10
                        seed = build_seed(char_seed, seed_offset)
                        view_prompt = f"{item['prompt']}, {variant_suffix}"
                        status.write(f"Generating {item['filename']} {variant_key}…")
                        images = stability_generate_images(
                            prompt=view_prompt,
                            negative_prompt=char_negative.strip(),
                            seed=seed,
                            aspect_ratio=char_aspect,
                            image_count=char_count,
                            api_key=stability_key,
                            init_image_bytes=init_bytes,
                            init_strength=char_strength if init_bytes else None,
                        )
                        for image_index, image_bytes in enumerate(images, start=1):
                            if char_count > 1:
                                filename = f"{item['filename']}_{variant_key}_{image_index:02d}.png"
                            else:
                                filename = f"{item['filename']}_{variant_key}.png"
                            all_outputs.append((filename, image_bytes))
                            st.image(image_bytes, caption=filename, use_container_width=True)

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
            st.dataframe(
                [{"filename": item["filename"], "prompt": item["prompt"][:120]} for item in skybox_items],
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
        negative = st.text_input("Negative text (optional)", "people, text, watermark", key="skybox_negative")
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
            init_b64 = _b64_of_uploaded_file(init_img) if init_img else None
            control_b64 = _b64_of_uploaded_file(control_img) if control_img else None
            items_to_run = skybox_items[:1] if preview_skybox else skybox_items
            all_outputs: list[tuple[str, bytes]] = []
            with st.status("Generating skyboxes…", expanded=True) as status:
                try:
                    for idx, item in enumerate(items_to_run):
                        skybox_seed = build_seed(int(seed), idx)
                        status.write(f"Generating {item['filename']}…")
                        gen = blockade_generate_skybox(
                            prompt=item["prompt"],
                            style_id=style_id,
                            negative_text=negative,
                            seed=skybox_seed,
                            enhance_prompt=enhance,
                            init_image_b64=init_b64,
                            init_strength=float(init_strength),
                            control_image_b64=control_b64,
                            control_model="remix",
                            api_key=blockade_key,
                        )
                        skybox_oid = gen["obfuscated_id"]
                        done = blockade_poll_generation(skybox_oid)
                        skybox_png = download_url_bytes(done["file_url"])
                        filename = f"{item['filename']}.png"
                        all_outputs.append((filename, skybox_png))
                        st.image(
                            skybox_png,
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
            init_b64 = _b64_of_uploaded_file(init_img) if init_img else None
            control_b64 = _b64_of_uploaded_file(control_img) if control_img else None
            skybox_seed = None if seed == 0 else int(seed)

            with st.status("Generating skybox…", expanded=True) as status:
                try:
                    gen = blockade_generate_skybox(
                        prompt=prompt,
                        style_id=style_id,
                        negative_text=negative,
                        seed=skybox_seed,
                        enhance_prompt=enhance,
                        init_image_b64=init_b64,
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
    
                    st.image(skybox_png, caption="Skybox (equirectangular preview)", use_container_width=True)
                    st.download_button(
                        "Download equirectangular (base)",
                        data=skybox_png,
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
    
                        st.download_button(
                            "Download equirectangular PNG (export)",
                            data=png_bytes,
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
