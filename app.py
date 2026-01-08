import base64
import time
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
def blockade_get_styles(model_version: int = 3):
    # Docs: Get Skybox Styles. :contentReference[oaicite:10]{index=10}
    url = f"{BLOCKADE_BASE}/skybox/styles"
    resp = requests.get(url, headers={"x-api-key": st.secrets["BLOCKADE_API_KEY"], "accept": "application/json"}, params={"model_version": model_version}, timeout=60)
    resp.raise_for_status()
    return resp.json()

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
    headers = {"x-api-key": st.secrets["BLOCKADE_API_KEY"], "accept": "application/json"}
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
    headers = {"x-api-key": st.secrets["BLOCKADE_API_KEY"], "accept": "application/json"}
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

# ---------------- UI: Skybox Tab ----------------

st.subheader("🌐 Skybox Generator (Blockade Labs)")

styles = blockade_get_styles(model_version=3)
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

# Export choices: equirectangular PNG + Cube Map Default
# Docs show export types list includes "equirectangular-png" and "cube-map-default-png". :contentReference[oaicite:16]{index=16}
export_png_type_id = 2   # typically PNG in docs example list (verify via GET export types if you want dynamic)
export_cubemap_type_id = 10

res_choice = st.selectbox("Export resolution", ["2K", "4K", "8K", "16K"], index=2)
res_map = {"1K": 1, "2K": 2, "4K": 3, "8K": 4, "16K": 7}  # based on docs example ids :contentReference[oaicite:17]{index=17}
resolution_id = res_map[res_choice]

gen_btn = st.button("Generate Skybox", type="primary", use_container_width=True)

if gen_btn:
    init_b64 = _b64_of_uploaded_file(init_img) if init_img else None
    control_b64 = _b64_of_uploaded_file(control_img) if control_img else None

    with st.status("Generating skybox…", expanded=True) as status:
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

        status.write("Requesting exports…")
        exp_png = blockade_request_export(skybox_oid, type_id=export_png_type_id, resolution_id=resolution_id)
        exp_cube = blockade_request_export(skybox_oid, type_id=export_cubemap_type_id, resolution_id=resolution_id)

        exp_png_done = blockade_poll_export(exp_png["id"])
        exp_cube_done = blockade_poll_export(exp_cube["id"])

        png_bytes = download_url_bytes(exp_png_done["file_url"])
        cube_bytes = download_url_bytes(exp_cube_done["file_url"])

        st.download_button("Download equirectangular PNG (export)", data=png_bytes, file_name=f"skybox_{res_choice}_equirectangular.png", mime="image/png")
        st.download_button("Download cubemap (export)", data=cube_bytes, file_name=f"skybox_{res_choice}_cubemap.zip", mime="application/zip")

        status.update(label="Done", state="complete", expanded=False)
