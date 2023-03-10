import base64
import pathlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch

from scripts.gligen_pluggable import PluggableGLIGEN
from scripts.sketch_helper import get_high_freq_colors, color_quantization, create_binary_matrix_base64, create_binary_mask
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json

from modules import devices, shared, processing

import modules.scripts as scripts
import gradio as gr

from gradio import processing_utils

import natsort

from modules.script_callbacks import CFGDenoisedParams, on_cfg_denoised, on_after_ui

from modules.processing import StableDiffusionProcessing
gradio_compat = True
MAX_COLORS = 12
switch_values_symbol = '\U000021C5' # ⇅

# rescale_js = """
# function(x) {
#     const root = document.querySelector('gradio-app').shadowRoot || document.querySelector('gradio-app');
#     let image_scale = parseFloat(root.querySelector('#image_scale input').value) || 1.0;
#     const image_width = root.querySelector('#img2img_image').clientWidth;
#     const target_height = parseInt(image_width * image_scale);
#     document.body.style.setProperty('--height', `${target_height}px`);
#     root.querySelectorAll('button.justify-center.rounded')[0].style.display='none';
#     root.querySelectorAll('button.justify-center.rounded')[1].style.display='none';
#     return x;
# }
# """

class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        if x is None:
            return x
        if self.tool == "sketch" and self.source in ["upload", "webcam"] and type(x) != dict:
            decode_image = processing_utils.decode_base64_to_image(x)
            width, height = decode_image.size
            mask = np.zeros((height, width, 4), dtype=np.uint8)
            mask[..., -1] = 255
            mask = self.postprocess(mask)
            x = {'image': x, 'mask': mask}
        return super().preprocess(x)


def binarize(x):
    return (x != 0).astype('uint8') * 255

def sized_center_crop(img, cropx, cropy):
    y, x = img.shape[:2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty+cropy, startx:startx+cropx]

def sized_center_fill(img, fill, cropx, cropy):
    y, x = img.shape[:2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    img[starty:starty+cropy, startx:startx+cropx] = fill
    return img

def sized_center_mask(img, cropx, cropy):
    y, x = img.shape[:2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    center_region = img[starty:starty+cropy, startx:startx+cropx].copy()
    img = (img * 0.2).astype('uint8')
    img[starty:starty+cropy, startx:startx+cropx] = center_region
    return img

def center_crop(img, HW=None, tgt_size=(512, 512)):
    if HW is None:
        H, W = img.shape[:2]
        HW = min(H, W)
    img = sized_center_crop(img, HW, HW)
    img = Image.fromarray(img)
    img = img.resize(tgt_size)
    return np.array(img)


def draw_box(boxes=[], texts=[], img=None, width=512, height=512):
    if len(boxes) == 0 and img is None:
        return None

    if img is None:
        img = Image.new('RGB', (height, width), (255, 255, 255))
    colors = ["red", "olive", "blue", "green", "orange", "brown", "cyan", "purple"]
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(Path(__file__).parent.parent / 'DejaVuSansMono.ttf'), size=18)
    for bid, box in enumerate(boxes):
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=colors[bid % len(colors)], width=4)
        anno_text = texts[bid]
        draw.rectangle([box[0], box[3] - int(font.size * 1.2), box[0] + int((len(anno_text) + 0.8) * font.size * 0.6), box[3]], outline=colors[bid % len(colors)], fill=colors[bid % len(colors)], width=4)
        draw.text([box[0] + int(font.size * 0.2), box[3] - int(font.size*1.2)], anno_text, font=font, fill=(255,255,255))
    return img


def draw(task, input, grounding_texts, new_image_trigger, state):
    if type(input) == dict:
        image = input['image']
        width, height = image.shape[:2]
        mask = input['mask']
    else:
        mask = input

    if mask.ndim == 3:
        mask = mask[..., 0]

    image_scale = 1.0

    # resize trigger
    # if task == "Grounded Inpainting":
    #     mask_cond = mask.sum() == 0
    #     # size_cond = mask.shape != (512, 512)
    #     if mask_cond and 'original_image' not in state:
    #         image = Image.fromarray(image)
    #         width, height = image.size
    #         scale = 600 / min(width, height)
    #         image = image.resize((int(width * scale), int(height * scale)))
    #         state['original_image'] = np.array(image).copy()
    #         image_scale = float(height / width)
    #         return [None, new_image_trigger + 1, image_scale, state]
    #     else:
    # original_image = state['original_image']
    # H, W = original_image.shape[:2]
    # image_scale = float(H / W)

    mask = binarize(mask)
    # if mask.shape != (512, 512):
    #     # assert False, "should not receive any non- 512x512 masks."
    #     if 'original_image' in state and state['original_image'].shape[:2] == mask.shape:
    #         mask = center_crop(mask, state['inpaint_hw'])
    #         image = center_crop(state['original_image'], state['inpaint_hw'])
    #     else:
    #         mask = np.zeros((512, 512), dtype=np.uint8)
    # mask = center_crop(mask)
    # mask = binarize(mask)

    if type(mask) != np.ndarray:
        mask = np.array(mask)

    if mask.sum() == 0 and task != "Grounded Inpainting":
        state = {}

    if task != 'Grounded Inpainting':
        image = None
    else:
        image = Image.fromarray(image)

    if 'boxes' not in state:
        state['boxes'] = []

    if 'masks' not in state or len(state['masks']) == 0:
        state['masks'] = []
        last_mask = np.zeros_like(mask)
    else:
        last_mask = state['masks'][-1]

    if type(mask) == np.ndarray and mask.size > 1:
        diff_mask = mask - last_mask
    else:
        diff_mask = np.zeros([])

    if diff_mask.sum() > 0:
        x1x2 = np.where(diff_mask.max(0) != 0)[0]
        y1y2 = np.where(diff_mask.max(1) != 0)[0]
        y1, y2 = y1y2.min(), y1y2.max()
        x1, x2 = x1x2.min(), x1x2.max()

        if (x2 - x1 > 5) and (y2 - y1 > 5):
            state['masks'].append(mask.copy())
            state['boxes'].append((x1, y1, x2, y2))

    grounding_texts = [x.strip() for x in grounding_texts.split(';')]
    grounding_texts = [x for x in grounding_texts if len(x) > 0]
    if len(grounding_texts) < len(state['boxes']):
        grounding_texts += [f'Obj. {bid+1}' for bid in range(len(grounding_texts), len(state['boxes']))]

    box_image = draw_box(state['boxes'], grounding_texts, image, width, height)

    if box_image is not None and state.get('inpaint_hw', None):
        inpaint_hw = state['inpaint_hw']
        box_image_resize = np.array(box_image.resize((inpaint_hw, inpaint_hw)))
        original_image = state['original_image'].copy()
        box_image = sized_center_fill(original_image, box_image_resize, inpaint_hw, inpaint_hw)

    return [box_image, new_image_trigger, image_scale, state]

def clear(task, sketch_pad_trigger, state, switch_task=False):
    if task != 'Grounded Inpainting':
        sketch_pad_trigger = sketch_pad_trigger + 1

    state = {}
    return [None, sketch_pad_trigger, None, 1.0] + [state]


shared.pluggable_gli = None


class Script(scripts.Script):

    def __init__(self):
        model_path = pathlib.Path(__file__).parent.parent / 'models' / 'gligen_textbox_delta.pth'
        gligen_state_dict = torch.load(str(model_path), map_location='cuda')
        if not shared.pluggable_gli:
            shared.pluggable_gli = PluggableGLIGEN(shared.sd_model.model.diffusion_model,gligen_state_dict)
        return


    def title(self):
        return "GLIGEN extension"

    def show(self, is_img2img):
        return scripts.AlwaysVisible


    def ui(self, is_img2img):
        process_script_params = []
        with gr.Group() as group_gligen_root:
            with gr.Accordion("Pluggable GLIGEN", open=False):
                enabled = gr.Checkbox(value=False, label="Enabled")
                with gr.Row():
                    with gr.Column(scale=4):
                        sketch_pad_trigger = gr.Number(value=0, visible=False)
                        sketch_pad_resize_trigger = gr.Number(value=0, visible=False)
                        init_white_trigger = gr.Number(value=0, visible=False)
                        image_scale = gr.Number(value=0, elem_id="image_scale", visible=False)
                        new_image_trigger = gr.Number(value=0, visible=False)

                        task = gr.Radio(
                            choices=["Grounded Generation", 'Grounded Inpainting'],
                            type="value",
                            value="Grounded Generation",
                            label="Task",
                            visible=False
                        )
                        # language_instruction = gr.Textbox(
                        #     label="Language instruction",
                        # )
                        grounding_instruction = gr.Textbox(
                            label="Grounding instruction (Separated by semicolon)",
                        )
                        strength = gr.Slider(label="Strength", labminimum=0.0, maximum=2.0, value=1.0, step=0.01, interactive=True)
                        stage_one = gr.Slider(label="Stage One", labminimum=0.0, maximum=1.0, value=0.2, step=0.01, interactive=True)
                        stage_two = gr.Slider(label="Stage Two", minimum=0.0, maximum=1.0, value=0.5, step=0.01, interactive=True)

                        # stage_three = gr.Slider(label="Stage Three", minimum=0.0, maximum=1.0, value=0.2, step=0.01, interactive=True)
                        with gr.Row():
                            sketch_pad = ImageMask(label="Sketch Pad", elem_id="img2img_image")
                            out_imagebox = gr.Image(type="pil", label="Parsed Sketch Pad")
                        with gr.Row():
                            clear_btn = gr.Button(value='Clear')
                            # gen_btn = gr.Button(value='Generate', elem_id="generate-btn")

                        def create_canvas(h, w):
                            return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255
                        with gr.Row():
                            with gr.Column():
                                canvas_width = gr.Slider(label="Canvas Width", minimum=256, maximum=1024, value=512,
                                                         step=64)
                                canvas_height = gr.Slider(label="Canvas Height", minimum=256, maximum=1024, value=512,
                                                          step=64)

                            if gradio_compat:
                                canvas_swap_res = ToolButton(value=switch_values_symbol)
                                canvas_swap_res.click(lambda w, h: (h, w), inputs=[canvas_width, canvas_height],
                                                      outputs=[canvas_width, canvas_height])

                        create_button = gr.Button(value="Create blank canvas")
                        create_button.click(fn=create_canvas, inputs=[canvas_height, canvas_width],
                                            outputs=[sketch_pad])

        state = gr.State({})

        class Controller:
            def __init__(self):
                self.calls = 0
                self.tracks = 0
                self.resizes = 0
                self.scales = 0

            def init_white(self, init_white_trigger):
                self.calls += 1
                return np.ones((512, 512), dtype='uint8') * 255, 1.0, init_white_trigger + 1

            def change_n_samples(self, n_samples):
                blank_samples = n_samples % 2 if n_samples > 1 else 0
                return [gr.Image.update(visible=True) for _ in range(n_samples + blank_samples)] \
                    + [gr.Image.update(visible=False) for _ in range(4 - n_samples - blank_samples)]

            def resize_centercrop(self, state):
                self.resizes += 1
                image = state['original_image'].copy()
                inpaint_hw = int(0.9 * min(*image.shape[:2]))
                state['inpaint_hw'] = inpaint_hw
                image_cc = center_crop(image, inpaint_hw)
                # print(f'resize triggered {self.resizes}', image.shape, '->', image_cc.shape)
                return image_cc, state

            def resize_masked(self, state):
                self.resizes += 1
                image = state['original_image'].copy()
                inpaint_hw = int(0.9 * min(*image.shape[:2]))
                state['inpaint_hw'] = inpaint_hw
                image_mask = sized_center_mask(image, inpaint_hw, inpaint_hw)
                state['masked_image'] = image_mask.copy()
                # print(f'mask triggered {self.resizes}')
                return image_mask, state

            def switch_task_hide_cond(self, task):
                cond = False
                if task == "Grounded Generation":
                    cond = True

                return gr.Checkbox.update(visible=cond, value=False), gr.Image.update(value=None,
                                                                                      visible=False), gr.Slider.update(
                    visible=cond), gr.Checkbox.update(visible=(not cond), value=False)

        controller = Controller()
        sketch_pad.edit(
            draw,
            inputs=[task, sketch_pad, grounding_instruction, sketch_pad_resize_trigger, state],
            outputs=[out_imagebox, sketch_pad_resize_trigger, image_scale, state],
            queue=False,
        )
        grounding_instruction.change(
            draw,
            inputs=[task, sketch_pad, grounding_instruction, sketch_pad_resize_trigger, state],
            outputs=[out_imagebox, sketch_pad_resize_trigger, image_scale, state],
            queue=False,
        )
        clear_btn.click(
            clear,
            inputs=[task, sketch_pad_trigger, state],
            outputs=[sketch_pad, sketch_pad_trigger, out_imagebox, image_scale, state],
            queue=False)
        sketch_pad_trigger.change(
            controller.init_white,
            inputs=[init_white_trigger],
            outputs=[sketch_pad, image_scale, init_white_trigger],
            queue=False)
        sketch_pad_resize_trigger.change(
            controller.resize_masked,
            inputs=[state],
            outputs=[sketch_pad, state],
            queue=False)
        # sketch_pad_resize_trigger.change(
        #     None,
        #     None,
        #     sketch_pad_resize_trigger,
        #     _js=rescale_js,
        #     queue=False)
        # init_white_trigger.change(
        #     None,
        #     None,
        #     init_white_trigger,
        #     _js=rescale_js,
        #     queue=False)
        process_script_params.append(enabled)
        process_script_params.extend([
            task,  grounding_instruction, sketch_pad,

            state,
            strength, stage_one, stage_two
        ])

        return process_script_params







    def process(self, p: StableDiffusionProcessing, *args, **kwargs):
        enabled, task, grounding_texts, sketch_pad, state, \
            strength, stage_one, stage_two = args
        if not enabled:
            return
        if 'boxes' not in state:
            state['boxes'] = []
        sketch_image = sketch_pad['image']
        boxes = state['boxes']
        height_width_arr = np.array([sketch_image.shape[1], sketch_image.shape[0], sketch_image.shape[1], sketch_image.shape[0]])
        grounding_texts = [x.strip() for x in grounding_texts.split(';')]
        assert len(boxes) == len(grounding_texts)
        boxes = (np.asarray(boxes) / height_width_arr).tolist()
        grounding_instruction ={obj: box for obj, box in zip(grounding_texts, boxes)}
        phrase_list, location_list = [], []
        for k, v in grounding_instruction.items():
            phrase_list.append(k)
            location_list.append(v)
        has_text_mask = 1
        has_image_mask = 0
        # get last sep token embedding
        max_objs = 30
        boxes = torch.zeros(max_objs, 4)
        masks = torch.zeros(max_objs)
        text_embeddings = torch.zeros(max_objs, 768)
        text_features = []
        clipembedder = shared.sd_model.cond_stage_model

        for phrase in phrase_list:
            # token_ids = clipembedder.tokenize(phrase)

            token_ids = clipembedder.wrapped.tokenizer(phrase, max_length=clipembedder.wrapped.max_length,return_length=True, return_overflowing_tokens=False, return_tensors="pt")
            clip_skip = processing.opts.data['CLIP_stop_at_last_layers']
            outputs = clipembedder.wrapped.transformer(input_ids=token_ids.input_ids.to(device=shared.device), output_hidden_states=-clip_skip)
            if clip_skip > 1:
                z = outputs.hidden_states[-clip_skip]
                z = clipembedder.wrapped.transformer.text_model.final_layer_norm(z)
            else:
                z = outputs.last_hidden_state

            feature = z[[0], [-2],:]
            text_features.append(feature)
        if len(text_features) > 0:
            text_features = torch.cat(text_features, dim=0)

            for idx, (box, text_feature) in enumerate(
                    zip(location_list, text_features)):
                boxes[idx] = torch.tensor(box)
                masks[idx] = 1
                text_embeddings[idx] = text_feature
        shared.pluggable_gli.update_stages(strength, stage_one, stage_two)
        shared.pluggable_gli.update_objs(boxes.unsqueeze(0), masks.unsqueeze(0), text_embeddings.unsqueeze(0), p.batch_size)
        shared.pluggable_gli.attach_all()



        return

    def postprocess(self, *args):
        shared.pluggable_gli.detach_all()
        return


