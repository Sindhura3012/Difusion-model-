# app.py
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

st.set_page_config(page_title="AI Image Chat", layout="wide")
st.title("💬 AI Image Generator (Diffusion Model)")

st.write("Type anything and get a realistic AI-generated image!")

# --- Sidebar settings ---
with st.sidebar:
    st.header("Settings")
    model_id = st.selectbox(
        "Choose Model",
        ["runwayml/stable-diffusion-v1-5", "hakurei/waifu-diffusion"]  # add more public models
    )
    num_steps = st.slider("Inference Steps", min_value=10, max_value=50, value=30)
    guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=7.5)

# --- Cache the model loading ---
@st.cache_resource(show_spinner=True)
def load_model(model_id):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    pipe.enable_attention_slicing()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return pipe.to(device)

pipe = load_model(model_id)

# --- Chat-style input ---
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:", placeholder="Type your prompt here...")

if st.button("Generate"):
    if user_input.strip() != "":
        with st.spinner("Generating image..."):
            image = pipe(
                user_input,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale
            ).images[0]

        # Save to session history
        st.session_state.history.append({"prompt": user_input, "image": image})
        user_input = ""

# --- Display history like chat ---
for chat in reversed(st.session_state.history):
    st.markdown(f"**Prompt:** {chat['prompt']}")
    st.image(chat["image"], use_column_width=True)
    st.markdown("---")
