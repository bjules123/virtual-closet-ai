import glob, os, torch
from PIL import Image
from transformers import YolosForObjectDetection, YolosImageProcessor

model_name = "valentinafeve/yolos-fashionpedia"
model     = YolosForObjectDetection.from_pretrained(model_name, use_safetensors=True)
processor = YolosImageProcessor.from_pretrained(model_name)
model.eval()

imgs = sorted(glob.glob("assets/closet-images/clothestest/*.png"))[:5]
for path in imgs:
    img    = Image.open(path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    probs              = outputs.logits.softmax(-1)[0, :, :-1]
    scores, label_ids  = probs.max(-1)
    bboxes             = outputs.pred_boxes[0]

    print(f"\n=== {os.path.basename(path)} {img.size} ===")
    hits = []
    for s, lid, box in zip(scores, label_ids, bboxes):
        if float(s) > 0.25:
            hits.append((float(s), model.config.id2label[lid.item()], [round(x, 3) for x in box.tolist()]))
    hits.sort(reverse=True)
    for s, lbl, box in hits[:12]:
        print(f"  {s:.3f}  {lbl:35s}  cx={box[0]:.2f} cy={box[1]:.2f} w={box[2]:.2f} h={box[3]:.2f}")
    if not hits:
        print("  (nothing above 0.25)")
