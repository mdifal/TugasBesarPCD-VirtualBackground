import numpy as np
import os
import cv2
import torch
from flask import Flask, render_template, request, make_response, jsonify, Response
from datetime import datetime
from functools import wraps, update_wrapper
from shutil import copyfile
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
from PIL import Image
import threading

app = Flask(__name__)

# Initialize model
weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
model = deeplabv3_mobilenet_v3_large(weights=weights)
model.eval()

# Preprocess function
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load initial background image

background_image_path = os.path.join(os.path.dirname(__file__), 'background.jpg')
background_image = cv2.imread(background_image_path)

lock = threading.Lock()

# Initialize webcam
cap = cv2.VideoCapture(0)

def process_frame(frame, background_image):
    background = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))
    input_tensor = preprocess(frame).unsqueeze(0)
    
    if torch.cuda.is_available():
        input_tensor = input_tensor.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    mask = (output_predictions == 15).astype(np.uint8) * 255
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_inv = cv2.bitwise_not(mask)
    person = cv2.bitwise_and(frame, frame, mask=mask)
    new_background = cv2.bitwise_and(background, background, mask=mask_inv)
    combined = cv2.add(person, new_background)
    
    return combined

def generate_frames():
    global background_image
    while True:
        success, frame = cap.read()
        if not success:
            break

        with lock:
            combined_frame = process_frame(frame, background_image)

        ret, buffer = cv2.imencode('.jpg', combined_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return update_wrapper(no_cache, view)


@app.route("/about")
@nocache
def about():
    return render_template('about.html')

@app.route("/index")
@app.route("/")
@nocache
def index():
    return render_template("home.html", file_path="img/image_here.jpg")

@app.route("/virtual-bg")
@nocache
def virtualBG():
    return render_template('virtualBG.html')

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

# Gallery-related routes and functions
project_dir = os.path.dirname(os.path.abspath(__file__))
GALLERY_DIR = os.path.join(project_dir, 'static', 'gallery')

@app.route('/capture', methods=['POST'])
def capture():
    success, frame = cap.read()
    if success:
        with lock:
            combined_frame = process_frame(frame, background_image)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'captured_photo_{timestamp}.png'
        filepath = os.path.join(GALLERY_DIR, filename)
        cv2.imwrite(filepath, combined_frame)
        return f"Photo captured and saved as '{filename}'", 200
    return "Failed to capture photo", 500

@app.route('/gallery')
def gallery():
    image_files = [f for f in os.listdir(GALLERY_DIR) if f.endswith('.png')]
    image_paths = [os.path.join('gallery', filename) for filename in image_files]

    image_info = []
    for filename in image_files:
        image_path = os.path.join(GALLERY_DIR, filename)
        
        with Image.open(image_path) as img:
            resolution = f"{img.width}x{img.height}"
            size = os.path.getsize(image_path) // 1024
            image_info.append({'filename': filename, 'resolution': resolution, 'size': size})
    print(image_info);        

    return render_template('gallery.html', image_paths=image_paths, image_info=image_info)



@app.route('/delete-image', methods=['POST'])
def delete_image():
    data = request.get_json()
    filename = data.get('image_path')  # This should actually be just filename
    if filename:
        full_image_path = os.path.join(app.static_folder, 'gallery', filename)
        if os.path.exists(full_image_path):
            try:
                os.remove(full_image_path)
                return jsonify({"message": "Gambar berhasil dihapus."}), 200
            except Exception as e:
                return jsonify({"message": f"Error deleting file: {str(e)}"}), 500
        else:
            return jsonify({"message": "File not found."}), 404
    return jsonify({"message": "Invalid request."}), 400

@app.route('/change_background', methods=['POST'])
def change_background():
    global background_image
    file = request.files['file']
    if file:
        with lock:
            background_image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        return "Background changed", 200
    return "Failed to change background", 500

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
