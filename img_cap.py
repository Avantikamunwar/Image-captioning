import requests
from PIL import Image
#PIL: python imaging library used for open, edit and process image in python libarary
from transformers  import AutoProcessor, BlipForConditionalGeneration #these are the vision language model available in hugging face library
#autoprocessor: prepares image and txt for giving input to BLIP model
#BlipforConditionalGeneration: to generate text from input
#BLIP: bootstraping language for image pre processing

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

img_path = r"C:\Users\kamun\Desktop\image_des\formal photo.jpg"
image = Image.open(img_path).convert('RGB')

text = "the image is of"
inputs = processor(images=image, text=text, return_tensors="pt")
#we give processed image as tensor so that deep learning models can use it

outputs = model.generate(**inputs, max_length=50)
#** tells to pass all the input to the model and then produce caption of only 50 words

caption = processor.decode(outputs[0], skip_special_tokens=True)
#to convert the tokenized output of the processor into human readable form
print(caption)