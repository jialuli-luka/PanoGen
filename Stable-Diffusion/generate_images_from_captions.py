import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import json
import os
from tqdm import tqdm
import torch.multiprocessing as mp
import argparse
# import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=None)
args = parser.parse_args()

generate_unseen = True
# train_data = []
with open("annotations/R2R_train_enc.json", 'r') as f:
    train_data = json.load(f)
f.close()

scene_ids = set()

for data in train_data:
    scene_ids.add(data["scan"])

scene_ids = sorted(list(scene_ids))


with open("../BLIP-2/captions.json", "r") as f:
    results = json.load(f)

if generate_unseen:
    unseen_scans = set()
    for key_, value in results.items():
        scan = key_.split("_")[0]
        if scan not in scene_ids:
            unseen_scans.add(scan)

    unseen_scans = sorted(list(unseen_scans))

    scans = unseen_scans[args.start:args.end]
else:
    scans = scene_ids[args.start:args.end]

results_new = dict()
print(len(results))
for key_, value in results.items():
    scan = key_.split("_")[0]
    if scan in scans:
        results_new[key_] = value

print(len(results_new))

model_id = "stabilityai/stable-diffusion-2-1-base"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", cache_dir="pretrained_model/")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16, cache_dir="pretrained_model/")
pipe = pipe.to("cuda")

img_path = "../../views_img_sd/"

batch_size = 24
prompts = []
keys = []
for key_, value in tqdm(results_new.items()):
    scan = key_.split("_")[0]
    # if scan not in scans:
    #     continue
    prompts.append(value[0])
    keys.append(key_)
    if len(prompts) == batch_size:
        images = pipe(prompts).images
        for i in range(batch_size):
            key = keys[i]
            scan = key.split("_")[0]
            vp = key.split("_")[1]
            idx = key.split("_")[-1]
            save_path = os.path.join(img_path, scan)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_path = os.path.join(img_path, scan, vp)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_path = os.path.join(img_path, scan, vp, idx + ".jpg")
            images[i].save(save_path)
        prompts = []
        keys = []
