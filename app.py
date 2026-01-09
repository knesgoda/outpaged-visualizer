import base64
import io
import time
import zipfile
import requests
import streamlit as st

BLOCKADE_BASE = "https://backend.blockadelabs.com/api/v1"

def _b64_of_uploaded_file(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    return base64.b64encode(data).decode("utf-8")

def blockade_headers() -> dict:
    api_key = st.secrets.get("BLOCKADE_API_KEY", "")
    if not api_key:
        st.error("Missing BLOCKADE_API_KEY in Streamlit Secrets.")
        st.stop()
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
        headers=blockade_headers(),
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
        resp = requests.get(url, headers=blockade_headers(), timeout=60)
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
    seed: int = 0,
    enhance_prompt: bool = False,
    init_image_b64: str | None = None,
    init_strength: float = 0.5,
    control_image_b64: str | None = None,
    control_model: str = "remix",
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
    if seed is not None:
        payload["seed"] = int(seed)

    # Prefer init_image if provided; control_image is more "structure only".
    if init_image_b64:
        payload["init_image"] = init_image_b64
        payload["init_strength"] = float(init_strength)

    if control_image_b64:
        payload["control_image"] = control_image_b64
        payload["control_model"] = control_model  # required for remix from control image :contentReference[oaicite:12]{index=12}

    resp = requests.post(f"{BLOCKADE_BASE}/skybox", headers=blockade_headers(), json=payload, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"Blockade error {resp.status_code}: {resp.text}")
    return resp.json()

def blockade_poll_generation(obfuscated_id: str, sleep_s: float = 2.0, max_wait_s: int = 300):
    # Docs mention tracking generation progress; you can poll by obfuscated id. :contentReference[oaicite:13]{index=13}
    # Endpoint shown in docs nav: "Get Skybox by Obfuscated id"
    url = f"{BLOCKADE_BASE}/skybox/{obfuscated_id}"
    headers = blockade_headers()
    waited = 0
    while waited < max_wait_s:
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
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

    resp = requests.post(f"{BLOCKADE_BASE}/skybox/export", headers=blockade_headers(), json=payload, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"Export error {resp.status_code}: {resp.text}")
    return resp.json()

def blockade_poll_export(export_id: str, sleep_s: float = 2.0, max_wait_s: int = 300):
    # Docs: GET /skybox/export/{export.id} returns file_url + status. :contentReference[oaicite:15]{index=15}
    url = f"{BLOCKADE_BASE}/skybox/export/{export_id}"
    headers = blockade_headers()
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
        "accept": "application/json",
    }

def stability_api_key() -> str:
    api_key = st.secrets.get("STABILITY_API_KEY", "")
    if not api_key:
        st.error(
            "Missing STABILITY_API_KEY in Streamlit Secrets. The key needs access to "
            "Stability.ai Stable Image Core (v2beta) generation."
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
):
    url = "https://api.stability.ai/v2beta/stable-image/generate/core"
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "aspect_ratio": aspect_ratio,
        "output_format": "png",
        "samples": image_count,
    }
    if seed is not None:
        payload["seed"] = int(seed)

    resp = requests.post(url, headers=stability_headers(api_key), files=payload, timeout=180)
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

# ---------------- UI: Skybox Tab ----------------

st.subheader("🌐 Skybox Generator (Blockade Labs)")

headers = blockade_headers()
try:
    styles = blockade_get_styles(model_version=3, api_key=headers["x-api-key"])
except (requests.exceptions.RequestException, RuntimeError) as exc:
    st.error(f"Unable to load Blockade styles: {exc}")
    st.stop()
style_options = {f"{s['name']} (id {s['id']})": s["id"] for s in styles}
style_label = st.selectbox("Skybox Style (Model 3)", list(style_options.keys()))
style_id = style_options[style_label]

prompt = st.text_area("Skybox Prompt", "Laputan observatory walkway under cold sky light, chalk-marked geometric diagrams, star charts, brass instruments, vivid chalk pastel look")
negative = st.text_input("Negative text (optional)", "people, text, watermark")
seed = st.number_input("Seed (0 = random)", min_value=0, max_value=2147483647, value=0)
enhance = st.checkbox("Enhance prompt (Blockade)", value=False)

colA, colB = st.columns(2)
with colA:
    init_img = st.file_uploader("Optional INIT image (2:1 equirectangular)", type=["png", "jpg", "jpeg"])
    init_strength = st.slider("Init strength (lower = more influence)", min_value=0.11, max_value=0.90, value=0.50, step=0.01)
with colB:
    control_img = st.file_uploader("Optional CONTROL image (2:1 equirectangular)", type=["png", "jpg", "jpeg"])
    st.caption("Control image preserves structure/perspective more than color. Requires control_model='remix'.")

try:
    export_meta = blockade_get_export_types(api_key=headers["x-api-key"])
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

    res_choice = st.selectbox("Export resolution", resolution_labels, index=2)
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
    )
    export_cubemap_type_id = st.number_input(
        "Cubemap ZIP export type ID",
        min_value=1,
        value=2,
        step=1,
    )
    res_choice = st.text_input("Export resolution label (for filenames)", value="custom")
    resolution_id = st.number_input(
        "Export resolution ID",
        min_value=1,
        value=1,
        step=1,
    )

gen_btn = st.button("Generate Skybox", type="primary", use_container_width=True)

if gen_btn:
    init_b64 = _b64_of_uploaded_file(init_img) if init_img else None
    control_b64 = _b64_of_uploaded_file(control_img) if control_img else None

    with st.status("Generating skybox…", expanded=True) as status:
        try:
            gen = blockade_generate_skybox(
                prompt=prompt,
                style_id=style_id,
                negative_text=negative,
                seed=int(seed),
                enhance_prompt=enhance,
                init_image_b64=init_b64,
                init_strength=float(init_strength),
                control_image_b64=control_b64,
                control_model="remix",
            )
            skybox_oid = gen["obfuscated_id"]
            status.write(f"Generation started. obfuscated_id: {skybox_oid}")

            done = blockade_poll_generation(skybox_oid)
            status.write("Skybox complete. Fetching base image…")
            skybox_png = download_url_bytes(done["file_url"])

            st.image(skybox_png, caption="Skybox (equirectangular preview)", use_container_width=True)
            st.download_button("Download equirectangular (base)", data=skybox_png, file_name="skybox_equirectangular_base.png", mime="image/png")

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

st.divider()

# ---------------- UI: Character Training Images (Stability.ai) ----------------

st.subheader("🧍 Character Training Images (Stability.ai)")
st.caption(
    "Requires Streamlit Secret STABILITY_API_KEY with permission to call "
    "Stability.ai Stable Image Core (v2beta) image generation."
)

char_prompt = st.text_area(
    "Character prompt",
    "A stylized sci-fi explorer, full body, detailed outfit, neutral pose, consistent lighting, plain studio background",
)
view_options = {
    "Front": "front",
    "3/4": "three-quarter",
    "Profile": "side profile",
    "Back": "back",
    "T-pose": "T-pose",
}
selected_view_labels = st.multiselect(
    "Views to generate",
    list(view_options.keys()),
    default=["Front", "3/4", "Profile", "Back"],
    help="Select one or more views to include in the training set.",
)
lock_seed = st.checkbox(
    "Lock seed across views",
    value=True,
    help="When off, each view will use a random seed for more variety.",
)
st.markdown("**Prompt builder**")
turnaround_fragment = st.checkbox(
    "Add turnaround fragment",
    value=True,
    help="Adds a consistent turnaround phrase covering common views and pose.",
)
lighting_fragment = st.checkbox(
    "Add lighting/background fragment",
    value=True,
    help="Adds neutral lighting and plain background guidance.",
)
training_fragment = st.checkbox(
    "Add training boilerplate fragment",
    value=True,
    help="Adds full-body, turntable training, and neutral pose guidance.",
)
prompt_fragments = []
if turnaround_fragment:
    prompt_fragments.append("turnaround: front, three-quarter, profile, back, T-pose")
if lighting_fragment:
    prompt_fragments.append("neutral lighting, orthographic look, plain background")
if training_fragment:
    prompt_fragments.append("full-body character, turntable training image, neutral pose, consistent lighting")

default_prompt_builder = ", ".join(
    [segment for segment in [char_prompt.strip(), *prompt_fragments] if segment]
).strip()
reset_prompt_builder = st.button("Reset prompt builder to defaults", use_container_width=True)
if "prompt_builder" not in st.session_state or reset_prompt_builder:
    st.session_state.prompt_builder = default_prompt_builder
prompt_builder = st.text_area(
    "Prompt builder (editable)",
    key="prompt_builder",
    height=120,
    help="Edit this prompt to fit your character style. View-specific text is appended automatically.",
)
st.caption("Tip: edit the prompt builder to fine-tune your character style. Use reset if you change fragments.")
char_negative = st.text_input(
    "Negative prompt",
    "text, watermark, logo, blurry, low quality, cropped, extra limbs",
)
char_seed = st.number_input("Seed (leave 0 for random)", min_value=0, max_value=2147483647, value=0)
aspect_ratio = st.selectbox(
    "Aspect ratio",
    ["1:1", "2:3", "3:2", "9:16", "16:9"],
    index=0,
)
image_count = st.slider("Images per view", min_value=1, max_value=4, value=2)

char_btn = st.button("Generate Character Views", type="primary", use_container_width=True)

if char_btn:
    selected_views = [view_options[label] for label in selected_view_labels]
    if not selected_views:
        st.error("Please select at least one view to generate.")
        st.stop()
    stability_key = stability_api_key()

    all_images: list[tuple[str, bytes]] = []
    with st.status("Generating character views…", expanded=True) as status:
        try:
            for view in selected_views:
                view_prompt = f"{prompt_builder.strip()}, {view} view"
                status.write(f"Requesting {view} view…")
                images = stability_generate_images(
                    prompt=view_prompt,
                    negative_prompt=char_negative.strip(),
                    seed=(char_seed if char_seed > 0 else None) if lock_seed else None,
                    aspect_ratio=aspect_ratio,
                    image_count=image_count,
                    api_key=stability_key,
                )
                for idx, image_bytes in enumerate(images, start=1):
                    filename = f"character_{view.replace(' ', '_')}_{idx}.png"
                    all_images.append((filename, image_bytes))

            for filename, image_bytes in all_images:
                st.image(image_bytes, caption=filename, use_container_width=True)
                st.download_button(
                    f"Download {filename}",
                    data=image_bytes,
                    file_name=filename,
                    mime="image/png",
                )

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for filename, image_bytes in all_images:
                    zip_file.writestr(filename, image_bytes)
            zip_buffer.seek(0)
            st.download_button(
                "Download all images (ZIP)",
                data=zip_buffer,
                file_name="character_training_images.zip",
                mime="application/zip",
            )

            status.update(label="Done", state="complete", expanded=False)
        except requests.exceptions.RequestException as exc:
            status.update(label="Failed", state="error", expanded=True)
            st.error(f"Request failed while generating images: {exc}")
            st.exception(exc)
        except RuntimeError as exc:
            status.update(label="Failed", state="error", expanded=True)
            st.error(f"Image generation failed: {exc}")
            st.exception(exc)
