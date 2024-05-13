import gradio as gr
import requests
import io
import random
import os
from PIL import Image

list_models = [
    "SDXL-1.0",
    "SD-1.5",
    "OpenJourney-V4",
    "Anything-V4",
    "Disney-Pixar-Cartoon",
    "Pixel-Art-XL",
    "Dalle-3-XL",
    "Midjourney-V4-XL",
]

def generate_txt2img(current_model, prompt, is_negative=False, image_style="None style", steps=50, cfg_scale=7,
                     seed=None):

    if current_model == "SD-1.5":
        API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    elif current_model == "SDXL-1.0":
        API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    elif current_model == "OpenJourney-V4":
        API_URL = "https://api-inference.huggingface.co/models/prompthero/openjourney"       
    elif current_model == "Anything-V4":
        API_URL = "https://api-inference.huggingface.co/models/xyn-ai/anything-v4.0" 
    elif current_model == "Disney-Pixar-Cartoon":
        API_URL = "https://api-inference.huggingface.co/models/stablediffusionapi/disney-pixar-cartoon"
    elif current_model == "Pixel-Art-XL":
        API_URL = "https://api-inference.huggingface.co/models/nerijs/pixel-art-xl"
    elif current_model == "Dalle-3-XL":
        API_URL = "https://api-inference.huggingface.co/models/openskyml/dalle-3-xl"
    elif current_model == "Midjourney-V4-XL":
        API_URL = "https://api-inference.huggingface.co/models/openskyml/midjourney-v4-xl"    

    API_TOKEN = os.environ.get("HF_READ_TOKEN")
    headers = {"Authorization": f"Bearer {API_TOKEN}"}


    if image_style == "None style":
        payload = {
            "inputs": prompt + ", 8k",
            "is_negative": is_negative,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed if seed is not None else random.randint(-1, 2147483647)
        }
    elif image_style == "Cinematic":
        payload = {
            "inputs": prompt + ", realistic, detailed, textured, skin, hair, eyes, by Alex Huguet, Mike Hill, Ian Spriggs, JaeCheol Park, Marek Denko",
            "is_negative": is_negative + ", abstract, cartoon, stylized",
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed if seed is not None else random.randint(-1, 2147483647)
        }
    elif image_style == "Digital Art":
        payload = {
            "inputs": prompt + ", faded , vintage , nostalgic , by Jose Villa , Elizabeth Messina , Ryan Brenizer , Jonas Peterson , Jasmine Star",
            "is_negative": is_negative + ", sharp , modern , bright",
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed if seed is not None else random.randint(-1, 2147483647)
        }
    elif image_style == "Portrait":
        payload = {
            "inputs": prompt + ", soft light, sharp, exposure blend, medium shot, bokeh, (hdr:1.4), high contrast, (cinematic, teal and orange:0.85), (muted colors, dim colors, soothing tones:1.3), low saturation, (hyperdetailed:1.2), (noir:0.4), (natural skin texture, hyperrealism, soft light, sharp:1.2)",
            "is_negative": is_negative,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed if seed is not None else random.randint(-1, 2147483647)
        }

    image_bytes = requests.post(API_URL, headers=headers, json=payload).content
    image = Image.open(io.BytesIO(image_bytes))
    return image


css = """
/* General Container Styles */
.gradio-container {
    font-family: 'IBM Plex Sans', sans-serif;
    max-width: 730px !important;
    margin: auto;
    padding-top: 1.5rem;
}

/* Button Styles */
.gr-button {
    color: white;
    border-color: black;
    background: black;
    white-space: nowrap;
}

.gr-button:focus {
    border-color: rgb(147 197 253 / var(--tw-border-opacity));
    outline: none;
    box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
    --tw-border-opacity: 1;
    --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
    --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
    --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
    --tw-ring-opacity: .5;
}

/* Footer Styles */
.footer, .dark .footer {
    margin-bottom: 45px;
    margin-top: 35px;
    text-align: center;
    border-bottom: 1px solid #e5e5e5;
}

.footer > p, .dark .footer > p {
    font-size: .8rem;
    display: inline-block;
    padding: 0 10px;
    transform: translateY(10px);
    background: white;
}

.dark .footer {
    border-color: #303030;
}

.dark .footer > p {
    background: #0b0f19;
}

/* Share Button Styles */
#share-btn-container {
    padding: 0 0.5rem !important;
    background-color: #000000;
    justify-content: center;
    align-items: center;
    border-radius: 9999px !important;
    max-width: 13rem;
    margin-left: auto;
}

#share-btn-container:hover {
    background-color: #060606;
}

#share-btn {
    all: initial;
    color: #ffffff;
    font-weight: 600;
    cursor: pointer;
    font-family: 'IBM Plex Sans', sans-serif;
    margin-left: 0.5rem !important;
    padding: 0.5rem !important;
    right: 0;
}

/* Animation Styles */
.animate-spin {
    animation: spin 1s linear infinite;
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

/* Other Styles */
#gallery {
    min-height: 22rem;
    margin-bottom: 15px;
    margin-left: auto;
    margin-right: auto;
    border-bottom-right-radius: .5rem !important;
    border-bottom-left-radius: .5rem !important;
}
"""

with gr.Blocks(css=css) as demo:
    
    favicon = '<img src="" width="48px" style="display: inline">'
    gr.Markdown(
        f"""<h1><center>{favicon} AI Diffusion</center></h1>
            """
    )
    
    with gr.Row(elem_id="prompt-container"):
        current_model = gr.Dropdown(label="Current Model", choices=list_models, value=list_models[1])
        
    with gr.Row(elem_id="prompt-container"):
        text_prompt = gr.Textbox(label="Prompt", placeholder="a cute dog", lines=1, elem_id="prompt-text-input")
        text_button = gr.Button("Generate", variant='primary', elem_id="gen-button")
        
    with gr.Row():
        image_output = gr.Image(type="pil", label="Output Image", elem_id="gallery")
        
    with gr.Accordion("Advanced settings", open=False):
        negative_prompt = gr.Textbox(label="Negative Prompt", value="text, blurry, fuzziness", lines=1, elem_id="negative-prompt-text-input")
        image_style = gr.Dropdown(label="Style", choices=["None style", "Cinematic", "Digital Art", "Portrait"], value="None style", allow_custom_value=False)

    text_button.click(generate_txt2img, inputs=[current_model, text_prompt, negative_prompt, image_style], outputs=image_output)

demo.launch(show_api=False)