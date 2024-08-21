from flask import Flask, render_template, request
import os
import numpy as np
from PIL import Image
import torch
import mobileclip

app = Flask(__name__)

# 画像フォルダのパス
image_folder = 'static/images/'

model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained='checkpoints/mobileclip_s0.pt')
tokenizer = mobileclip.get_tokenizer('mobileclip_s0')

target_list = ["a diagram", "a dog", "a cat", "Pikachu", "Squirtle", "Venusaur"]

# 画像の一覧を取得する関数
def get_image_list():
    image_list = os.listdir(image_folder)
    image_list = [f for f in image_list if f.endswith('.png')]
    return image_list

# indexページを表示する関数
@app.route('/')
def index():
    image_list = get_image_list()
    return render_template('index.html', image_list=image_list)

# 選択された画像に対して処理を行う関数
@app.route('/submit', methods=['POST'])
def submit():
    selected_image = request.form['image']
    # 選択された画像に対する処理 (例: 画像を表示する)
    # img = Image.open(os.path.join(image_folder, selected_image))
    # ここに任意の処理を追加
    image = preprocess(Image.open(os.path.join(image_folder, selected_image)).convert('RGB')).unsqueeze(0)
    text = tokenizer(target_list)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).tolist()[0]
        print(type(text_probs))
        print(max(text_probs))

    return 'You selected: ' + target_list[text_probs.index(max(text_probs))]

if __name__ == '__main__':
    app.run(debug=True)
