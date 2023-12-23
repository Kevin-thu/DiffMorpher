import os
import torch
import numpy as np
import cv2
import gradio as gr
from PIL import Image
from datetime import datetime
from model import DiffMorpherPipeline
from utils.lora_utils import train_lora

LENGTH=450

def train_lora_interface(
    image,
    prompt,
    model_path,
    output_path,
    lora_steps,
    lora_rank,
    lora_lr,
    num
):
    os.makedirs(output_path, exist_ok=True)
    train_lora(image, prompt, output_path, model_path,
               lora_steps=lora_steps, lora_lr=lora_lr, lora_rank=lora_rank, weight_name=f"lora_{num}.ckpt", progress=gr.Progress())
    return f"Train LoRA {'A' if num == 0 else 'B'} Done!"

def run_diffmorpher(
    image_0,
    image_1,
    prompt_0,
    prompt_1,
    model_path,
    lora_mode,
    lamb,
    use_adain,
    use_reschedule,
    num_frames,
    fps,
    save_inter,
    load_lora_path_0,
    load_lora_path_1,
    output_path
):
    run_id = datetime.now().strftime("%H%M") + "_" +  datetime.now().strftime("%Y%m%d")
    os.makedirs(output_path, exist_ok=True)
    morpher_pipeline = DiffMorpherPipeline.from_pretrained(model_path, torch_dtype=torch.float32).to("cuda")
    if lora_mode == "Fix LoRA 0":
        fix_lora = 0
    elif lora_mode == "Fix LoRA 1":
        fix_lora = 1
    else:
        fix_lora = None
    if not load_lora_path_0:
        load_lora_path_0 = f"{output_path}/lora_0.ckpt"
    if not load_lora_path_1:
        load_lora_path_1 = f"{output_path}/lora_1.ckpt"
    images = morpher_pipeline(
        img_0=image_0,
        img_1=image_1,
        prompt_0=prompt_0,
        prompt_1=prompt_1,
        load_lora_path_0=load_lora_path_0,
        load_lora_path_1=load_lora_path_1,
        lamb=lamb,
        use_adain=use_adain,
        use_reschedule=use_reschedule,
        num_frames=num_frames,
        fix_lora=fix_lora,
        save_intermediates=save_inter,
        progress=gr.Progress()
    )
    video_path = f"{output_path}/{run_id}.mp4"
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (512, 512))
    for i, image in enumerate(images):
        # image.save(f"{output_path}/{i}.png")
        video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    video.release()
    cv2.destroyAllWindows()
    return gr.Video(value=video_path, format="mp4", label="Output video", show_label=True, height=LENGTH, width=LENGTH, interactive=False)

def run_all(
    image_0,
    image_1,
    prompt_0,
    prompt_1,
    model_path,
    lora_mode,
    lamb,
    use_adain,
    use_reschedule,
    num_frames,
    fps,
    save_inter,
    load_lora_path_0,
    load_lora_path_1,
    output_path,
    lora_steps,
    lora_rank,
    lora_lr
):
    os.makedirs(output_path, exist_ok=True)
    train_lora(image_0, prompt_0, output_path, model_path,
        lora_steps=lora_steps, lora_lr=lora_lr, lora_rank=lora_rank, weight_name=f"lora_0.ckpt", progress=gr.Progress())
    train_lora(image_1, prompt_1, output_path, model_path,
        lora_steps=lora_steps, lora_lr=lora_lr, lora_rank=lora_rank, weight_name=f"lora_1.ckpt", progress=gr.Progress())
    return run_diffmorpher(
        image_0,
        image_1,
        prompt_0,
        prompt_1,
        model_path,
        lora_mode,
        lamb,
        use_adain,
        use_reschedule,
        num_frames,
        fps,
        save_inter,
        load_lora_path_0,
        load_lora_path_1,
        output_path
    )

with gr.Blocks() as demo:
    
    with gr.Row():
        gr.Markdown("""
        # Official Implementation of [DiffMorpher](https://kevin-thu.github.io/DiffMorpher_page/)
        """)

    original_image_0, original_image_1 = gr.State(Image.open("assets/Trump.jpg").convert("RGB").resize((512,512), Image.BILINEAR)), gr.State(Image.open("assets/Biden.jpg").convert("RGB").resize((512,512), Image.BILINEAR))
    # key_points_0, key_points_1 = gr.State([]), gr.State([])
    # to_change_points = gr.State([])
    
    with gr.Row():
        with gr.Column():
            input_img_0 = gr.Image(type="numpy", label="Input image A", value="assets/Trump.jpg", show_label=True, height=LENGTH, width=LENGTH, interactive=True)
            prompt_0 = gr.Textbox(label="Prompt for image A", value="a photo of an American man", interactive=True)
            with gr.Row():
                train_lora_0_button = gr.Button("Train LoRA A")
                train_lora_1_button = gr.Button("Train LoRA B")
            # show_correspond_button = gr.Button("Show correspondence points")
        with gr.Column():
            input_img_1 = gr.Image(type="numpy", label="Input image B ", value="assets/Biden.jpg", show_label=True, height=LENGTH, width=LENGTH, interactive=True)
            prompt_1 = gr.Textbox(label="Prompt for image B", value="a photo of an American man", interactive=True)
            with gr.Row():
                clear_button = gr.Button("Clear All")
                run_button = gr.Button("Run w/o LoRA training")
        with gr.Column():
            output_video = gr.Video(format="mp4", label="Output video", show_label=True, height=LENGTH, width=LENGTH, interactive=False)
            lora_progress_bar = gr.Textbox(label="Display LoRA training progress", interactive=False)
            run_all_button = gr.Button("Run!")
        # with gr.Column():
        #     output_video = gr.Video(label="Output video", show_label=True, height=LENGTH, width=LENGTH)
        
    with gr.Row():
        gr.Markdown("""
        ### Usage:
        1. Upload two images (with correspondence) and fill out the prompts. 
           (It's recommended to change `[Output path]` accordingly.)
        2. Click **"Run!"**
        
        Or:
        1. Upload two images (with correspondence) and fill out the prompts.
        2. Click the **"Train LoRA A/B"** button to fit two LoRAs for two images respectively. <br> &nbsp;&nbsp;
           If you have trained LoRA A or LoRA B before, you can skip the step and fill the specific LoRA path in LoRA settings. <br> &nbsp;&nbsp;
           Trained LoRAs are saved to `[Output Path]/lora_0.ckpt` and `[Output Path]/lora_1.ckpt` by default.
        3. You might also change the settings below.
        4. Click **"Run w/o LoRA training"**
        
        ### Note: 
        1. To speed up the generation process, you can **ruduce the number of frames** or **turn off "Use Reschedule"**.
        2. You can try the influence of different prompts. It seems that using the same prompts or aligned prompts works better.
        ### Have fun!
        """)
        
    with gr.Accordion(label="Algorithm Parameters"):
        with gr.Tab("Basic Settings"):
            with gr.Row():
                # local_models_dir = 'local_pretrained_models'
                # local_models_choice = \
                #     [os.path.join(local_models_dir,d) for d in os.listdir(local_models_dir) if os.path.isdir(os.path.join(local_models_dir,d))]
                model_path = gr.Text(value="stabilityai/stable-diffusion-2-1-base",
                    label="Diffusion Model Path", interactive=True
                )
                lamb = gr.Slider(value=0.6, minimum=0, maximum=1, step=0.1, label="Lambda for attention replacement", interactive=True)
                lora_mode = gr.Dropdown(value="LoRA Interp",
                    label="LoRA Interp. or Fix LoRA",
                    choices=["LoRA Interp", "Fix LoRA A", "Fix LoRA B"],
                    interactive=True
                )
                use_adain = gr.Checkbox(value=True, label="Use AdaIN", interactive=True)
                use_reschedule = gr.Checkbox(value=True, label="Use Reschedule", interactive=True)
            with gr.Row():
                num_frames = gr.Number(value=16, minimum=0, label="Number of Frames", precision=0, interactive=True)
                fps = gr.Number(value=8, minimum=0, label="FPS (Frame rate)", precision=0, interactive=True)
                save_inter = gr.Checkbox(value=False, label="Save Intermediate Images", interactive=True)
                output_path = gr.Text(value="./results", label="Output Path", interactive=True)
                
        with gr.Tab("LoRA Settings"):
            with gr.Row():
                lora_steps = gr.Number(value=200, label="LoRA training steps", precision=0, interactive=True)
                lora_lr = gr.Number(value=0.0002, label="LoRA learning rate", interactive=True)
                lora_rank = gr.Number(value=16, label="LoRA rank", precision=0, interactive=True)
                # save_lora_dir = gr.Text(value="./lora", label="LoRA model save path", interactive=True)
                load_lora_path_0 = gr.Text(value="", label="LoRA model load path for image A", interactive=True)
                load_lora_path_1 = gr.Text(value="", label="LoRA model load path for image B", interactive=True)
    
    def store_img(img):
        image = Image.fromarray(img).convert("RGB").resize((512,512), Image.BILINEAR)
        # resize the input to 512x512
        # image = image.resize((512,512), Image.BILINEAR)
        # image = np.array(image)
        # when new image is uploaded, `selected_points` should be empty
        return image
    input_img_0.upload(
        store_img,
        [input_img_0],
        [original_image_0]
    )
    input_img_1.upload(
        store_img,
        [input_img_1],
        [original_image_1]
    )
    
    def clear(LENGTH):
        return gr.Image.update(value=None, width=LENGTH, height=LENGTH), \
            gr.Image.update(value=None, width=LENGTH, height=LENGTH), \
            None, None, None, None
    clear_button.click(
        clear,
        [gr.Number(value=LENGTH, visible=False, precision=0)],
        [input_img_0, input_img_1, original_image_0, original_image_1, prompt_0, prompt_1]
    )
        
    train_lora_0_button.click(
        train_lora_interface,
        [
         original_image_0,
         prompt_0,
         model_path,
         output_path,
         lora_steps,
         lora_rank,
         lora_lr,
         gr.Number(value=0, visible=False, precision=0)
        ],
        [lora_progress_bar]
    )
    
    train_lora_1_button.click(
        train_lora_interface,
        [
         original_image_1,
         prompt_1,
         model_path,
         output_path,
         lora_steps,
         lora_rank,
         lora_lr,
         gr.Number(value=1, visible=False, precision=0)
        ],
        [lora_progress_bar]
    )
    
    run_button.click(
        run_diffmorpher,
        [
         original_image_0,
         original_image_1,
         prompt_0,
         prompt_1,
         model_path,
         lora_mode,
         lamb,
         use_adain,
         use_reschedule,
         num_frames,
         fps,
         save_inter,
         load_lora_path_0,
         load_lora_path_1,
         output_path
        ],
        [output_video]
    )
    
    run_all_button.click(
        run_all,
        [
         original_image_0,
         original_image_1,
         prompt_0,
         prompt_1,
         model_path,
         lora_mode,
         lamb,
         use_adain,
         use_reschedule,
         num_frames,
         fps,
         save_inter,
         load_lora_path_0,
         load_lora_path_1,
         output_path,
         lora_steps,
         lora_rank,
         lora_lr
        ],
        [output_video]
    )
        
demo.queue().launch(debug=True)