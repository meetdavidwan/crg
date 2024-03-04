import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from contrastive_generate import generate
from PIL import Image
import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
    images = kwargs.pop("images", None)
    image_sizes = kwargs.pop("image_sizes", None)
    inputs = self.prepare_inputs_for_generation(
        input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
    )
    if images is not None:
        inputs['images'] = images
    if image_sizes is not None:
        inputs['image_sizes'] = image_sizes
    return inputs

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,cache_dir=args.cache_dir)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        bboxs = line["bboxs"] if "bboxs" in line else None

        # prepare prompt
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()


        # prepare original image
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        image_tensor = image_tensor.unsqueeze(0)
        inputs,position_ids,attention_mask,_,inputs_embeds,_ = model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            None,
            None,
            None,
            image_tensor.half().cuda(),
            image_sizes=[image.size]
        )

        model_kwargs = {"postion_ids":position_ids,"attention_mask":attention_mask, "inputs_embeds": inputs_embeds}

        # prepare black out image if present
        image_blackout = None
        if args.black_out_image_folder is not None: # We take either the image from another prepared folder
            image_blackout = os.path.join(args.black_out_image_folder, image_file)
            image_blackout = Image.open(image_blackout)
        elif bboxs is not None: # Or we directly black-out here 
            image_blackout = image.copy()
            pixels = image_blackout.load()
            for bbox in bboxs:
                box = [int(b) for b in bbox]
                for i in range(bbox[0], bbox[2]):
                    for j in range(bbox[1],bbox[3]):
                        pixels[i,j] = (0,0,0)
        
        if image_blackout is not None:
            image_tensor_blackout = process_images([image_blackout], image_processor, model.config)[0]
            image_tensor_blackout = image_tensor_blackout.unsqueeze(0)
            inputs,position_ids,attention_mask,_,inputs_embeds,_ = model.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                None,
                None,
                None,
                image_tensor_blackout.half().cuda(),
                image_sizes=[image_blackout.size]
            )

            model_kwargs.update( {"postion_ids_blackout":position_ids,"attention_mask_blackout":attention_mask, "inputs_embeds_blackout": inputs_embeds} )
        
        with torch.inference_mode():
            output_ids = generate(
                model,
                input_ids=None,
                # images=image_tensor.half().cuda(),
                # image_sizes=[image.size],
                do_sample=False,
                # temperature=args.temperature,
                # top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,
                alpha=args.alpha,
                **model_kwargs,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.6-34b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="./answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="chatml_direct")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    
    parser.add_argument("--alpha",type=float,default=1.0)
    parser.add_argument("--black-out-image-folder", type=str)
    parser.add_argument("--cache-dir",type=str)

    args = parser.parse_args()

    eval_model(args)