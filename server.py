from PIL import Image
from chromadb import chromadb, QueryResult
from flask import Flask, render_template, request, send_from_directory
import os
import random

client = chromadb.PersistentClient(path="./store.db")
collection = client.get_or_create_collection("images")

from image_processor import CLIPImageProcessor
from model import CLIPModel
from tokenizer import CLIPTokenizer
from typing import Tuple
def load(model_dir: str) -> Tuple[CLIPModel, CLIPTokenizer, CLIPImageProcessor]:
    model = CLIPModel.from_pretrained(model_dir)
    tokenizer = CLIPTokenizer.from_pretrained(model_dir)
    img_processor = CLIPImageProcessor.from_pretrained(model_dir)
    return model, tokenizer, img_processor
model, tokenizer, img_processor = load("mlx_model")

def search(name: str, image: Image) -> QueryResult:
    inputs = {
        "input_ids": tokenizer([name]),
        "pixel_values": img_processor(
            [image]
        ),
    }
    output = model(**inputs)
    results = collection.query(
        query_embeddings=[output.image_embeds[0].tolist()],
        n_results=30,
    )
    return results

assets_dir = os.path.join(os.path.dirname(__file__), 'images/')

app = Flask(__name__, static_folder=assets_dir)
@app.route("/search/<path:filename>", methods=['GET'])
def get_file(filename):
    image = Image.open(assets_dir+filename)
    results = search(filename, image)
    images = [f for key, f in enumerate(results["ids"][0]) if results['distances'][0][key] < 0.5]
    return render_template('index.html', images=images)
@app.route("/")
def index():
    images = [f for f in os.listdir(assets_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.heic'))]
    images = random.sample(images, 20)
    return render_template('index.html', images=images)
app.run(port=8087)