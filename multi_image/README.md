## Update

Add support for multi-image input. Now you can get the morphing output among more than 2 images.

## Run the code

You can run the code with the following command:

```
python main.py \
  --image_paths [image_path_0] ... [image_path_n] \ 
  --prompts [prompt_0] ... [prompt_n] \
  --output_path [output_path] \
  --use_adain --use_reschedule --save_inter
```

This modification add support for the following options:

- `--image_paths`: Paths of the input images
- `--prompts`: Prompts of the images
- `--load_lora_paths`: Paths of the lora directory of the images

## Example

Run the code:
```
python main.py \
--image_paths ./assets/realdog0.jpg ./assets/realdog1.jpg ./assets/realdog2.jpg \
--prompts "A photo of a dog" "A photo of a dog" "A photo of a dog"  \
--output_path "./results/dog" \
--use_adain --use_reschedule --save_inter
```

Output:
<div  align="center">
<img src="assets/realdog.gif" width="50%" height="50%">
</div>