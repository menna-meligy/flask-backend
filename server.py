from flask import Flask, request, jsonify
from enum import Enum
from PIL import Image
import pytesseract
from flask_cors import CORS
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer 
import torch
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning") 
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model.to(device) 
max_length = 16 
num_beams = 4 
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class OS(Enum):
    Mac = 0
    Windows = 1

class Language(Enum):
    ENG = 'eng'

class ImageReader:
    def __init__(self, os: OS):
        if os == OS.Mac:
            print("Running on: MAC")
        elif os == OS.Windows:
            windows_path = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
            pytesseract.pytesseract.tesseract_cmd = windows_path
            print('Running on: Windows')

    def extract_text(self, image, lang):
        img = Image.open(image)
        extracted_text = pytesseract.image_to_string(img, lang=lang.value)
        return extracted_text

ir = ImageReader(OS.Windows)

@app.route("/extract-text", methods=["POST"])
def extract_text():
    try:
        image_file = request.files["image"]
        lang = Language.ENG
        extracted_text = ir.extract_text(image_file, Language(lang))
        return jsonify({"text": extracted_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict_caption():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        
        # Perform caption prediction
        captions = predict_caption_from_image(filename)
        
        # Delete the uploaded file
        os.remove(filename)
        
        return jsonify({'captions': captions})
    
    return jsonify({'error': 'Invalid file'})

def predict_caption_from_image(image_path):
    image = Image.open(image_path)
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True) 
    preds = [pred.strip() for pred in preds]
    
    return preds

if __name__ == "__main__":
    app.run(debug=True)
