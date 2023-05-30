import json
import argparse
from diffusers import StableDiffusionInpaintPipeline
import torch
from tqdm import tqdm
import os
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=16)
args = parser.parse_args()


pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
cache_dir="pretrained_model/"
# state_dict=state_dict
)
pipe = pipe.to("cuda")


img_path = "../../views_img_sd/"
output_path = "../../views_img_sd_inpaint"

with open("R2R_train_enc.json", 'r') as f:
    train_data = json.load(f)
f.close()

scene_ids = set()

for data in train_data:
    scene_ids.add(data["scan"])

scans_ = sorted(list(scene_ids))

scans = scans_[args.start:args.end]

with open("../BLIP-2/captions.json", "r") as f:
    results = json.load(f)

results_new = dict()
print(len(results))
for key_, value in results.items():
    scan = key_.split("_")[0]
    if scan in scans:
        results_new[key_] = value

print(len(results_new))

batch_size = args.batch_size

sequence = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
start_index = 12

for index in tqdm(sequence):
    print("Processing index:", index)
    if index != start_index:
        img_path = output_path
    output_list = []
    count = 0
    for key_, value in tqdm(results_new.items()):
        new_item = dict()
        new_item["scan"] = key_.split("_")[0]
        new_item["vp"] = key_.split("_")[1]
        new_item["idx"] = int(key_.split("_")[2])
        if new_item["idx"] == index:
            for d in ["left", "up", "down"]:
                if d == "left":
                    new_item["direction"] = "left"
                    new_item["target_idx"] = index + 1
                    if index + 1 == 24:
                        continue
                    if index + 1 == start_index:
                        continue
                    target_key = new_item["scan"] + "_" + new_item["vp"] + "_" + str(new_item["target_idx"])
                    new_item["prompt"] = results_new[target_key][0]
                    output_list.append(dict(new_item))
                elif d == "up":
                    new_item["direction"] = "up"
                    new_item["target_idx"] = index + 12
                    target_key = new_item["scan"] + "_" + new_item["vp"] + "_" + str(new_item["target_idx"])
                    new_item["prompt"] = results_new[target_key][0]
                    output_list.append(dict(new_item))
                elif d == "down":
                    new_item["direction"] = "down"
                    new_item["target_idx"] = index - 12
                    target_key = new_item["scan"] + "_" + new_item["vp"] + "_" + str(new_item["target_idx"])
                    new_item["prompt"] = results_new[target_key][0]
                    output_list.append(dict(new_item))
            if index <= start_index:
                new_item["direction"] = "right"
                new_item["target_idx"] = index - 1
                if index - 1 == 11:
                    continue
                target_key = new_item["scan"] + "_" + new_item["vp"] + "_" + str(new_item["target_idx"])
                new_item["prompt"] = results_new[target_key][0]
                output_list.append(dict(new_item))

    print("Processing %d images in the %d round" % (len(output_list), index))

    prompts = []
    targets = []
    inputs = []
    directions = []
    masks = []
    for j, item in tqdm(enumerate(output_list)):
        # print("Processing item", j, len(output_list))
        prompt = item["prompt"]
        prompts.append(prompt)
        target = "_".join([item["scan"], item["vp"], str(item["target_idx"])])
        targets.append(target)
        input_im = np.array(Image.open(os.path.join(img_path, item["scan"], item["vp"], str(item["idx"])+".jpg")))
        direction = item["direction"]
        if direction == "left":
            input_im_rotate = np.ones(input_im.shape, dtype=input_im.dtype) * 255
            input_im_rotate[:, :256, :] = input_im[:, 256:, :]
            mask = np.zeros(input_im.shape, dtype=input_im.dtype)
            mask[:, 256:, :] = 255
        elif direction == "right":
            input_im_rotate = np.ones(input_im.shape, dtype=input_im.dtype) * 255
            input_im_rotate[:, 256:, :] = input_im[:, :256, :]
            mask = np.zeros(input_im.shape, dtype=input_im.dtype)
            mask[:, :256, :] = 255
        elif direction == "up":
            input_im_rotate = np.ones(input_im.shape, dtype=input_im.dtype) * 255
            input_im_rotate[256:, :, :] = input_im[:256, :, :]
            mask = np.zeros(input_im.shape, dtype=input_im.dtype)
            mask[:256, :, :] = 255
        else:
            input_im_rotate = np.ones(input_im.shape, dtype=input_im.dtype) * 255
            input_im_rotate[:256, :, :] = input_im[256:, :, :]
            mask = np.zeros(input_im.shape, dtype=input_im.dtype)
            mask[256:, :, :] = 255
        inputs.append(Image.fromarray(input_im_rotate))
        masks.append(Image.fromarray(mask))

        if len(prompts) == batch_size:
            with torch.autocast("cuda"):
                images = pipe(prompt=prompts, image=inputs, mask_image=masks).images
            for i in range(batch_size):
                target = targets[i]
                scan = target.split("_")[0]
                vp = target.split("_")[1]
                idx = target.split("_")[-1]
                save_path = os.path.join(output_path, scan)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_path = os.path.join(output_path, scan, vp)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_path = os.path.join(output_path, scan, vp, idx + ".jpg")
                images[i].save(save_path)
            prompts = []
            targets = []
            inputs = []
            directions = []
            masks = []
        elif j == len(output_list) - 1:
            with torch.autocast("cuda"):
                images = pipe(prompt=prompts, image=inputs, mask_image=masks).images
            for i in range(len(inputs)):
                target = targets[i]
                scan = target.split("_")[0]
                vp = target.split("_")[1]
                idx = target.split("_")[-1]
                save_path = os.path.join(output_path, scan)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_path = os.path.join(output_path, scan, vp)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_path = os.path.join(output_path, scan, vp, idx + ".jpg")
                images[i].save(save_path)
            prompts = []
            targets = []
            inputs = []
            directions = []
            masks = []
