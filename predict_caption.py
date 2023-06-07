
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base")


def predict_step(img_url):
   # raw_image = Image.open(img_url).convert('RGB')
    raw_image = img_url

    # conditional image captioning
   # text = "a photography of"
    #inputs = processor(raw_image, text, return_tensors="pt")

    #out = model.generate(**inputs, max_length=60)
    #print(processor.decode(out[0], skip_special_tokens=True))

    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs, max_length=60
                         )
    out2 = processor.decode(out[0], skip_special_tokens=True)

    return out2
