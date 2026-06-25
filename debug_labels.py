from transformers import YolosForObjectDetection
model = YolosForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia", use_safetensors=True)
print("\n".join(sorted(model.config.id2label.values())))
