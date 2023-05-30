import torch
from PIL import Image
import os
import json
from tqdm import tqdm

from lavis.models import load_model_and_preprocess


# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# loads BLIP-2 pre-trained model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)
# load sample image

results = dict()
path = "views_img/"
scans = os.listdir(path)
for j, scan in enumerate(scans):
    print("processing scan:", scan, j, len(scans))
    if os.path.isdir(path+scan):
        vps = os.listdir(path+scan)
        for vp in tqdm(vps):
            if os.path.isdir(os.path.join(path, scan, vp)):
                for i in range(36):
                    raw_image = Image.open(os.path.join(path, scan, vp, str(i)+".jpg")).convert("RGB")
                    # prepare the image
                    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                    caption = model.generate({"image": image, "prompt": "a photo of"})
                    results[scan+"_"+vp+"_"+str(i)] = caption


with open("captions.json", "w") as f:
    json.dump(results, f)

