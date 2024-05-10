from typing import Tuple

from image_processor import CLIPImageProcessor
from model import CLIPModel
from tokenizer import CLIPTokenizer


def load(model_dir: str) -> Tuple[CLIPModel, CLIPTokenizer, CLIPImageProcessor]:
    model = CLIPModel.from_pretrained(model_dir)
    tokenizer = CLIPTokenizer.from_pretrained(model_dir)
    img_processor = CLIPImageProcessor.from_pretrained(model_dir)
    return model, tokenizer, img_processor


if __name__ == "__main__":
    from PIL import Image

    model, tokenizer, img_processor = load("mlx_model")

    import chromadb
    from pathlib import Path
    import logging

    PATH = "./images/"

    mlx_path = Path(PATH)
    logger = logging.getLogger(__name__)

    client = chromadb.PersistentClient(path="./store.db")
    collection = client.get_or_create_collection("images")

    def create():
        if not mlx_path.exists():
            logger.error("path not exists.")
        if not mlx_path.is_dir():
            logger.error("path is not directory.")
        for entry in mlx_path.iterdir():
            try:
                inputs = {
                    "input_ids": tokenizer([entry.name]),
                    "pixel_values": img_processor(
                        [Image.open(entry)]
                    ),
                }
                output = model(**inputs)
                collection.add(
                    ids=[entry.name],
                    embeddings=[output.image_embeds[0].tolist()]
                )
            except OSError as e:
                print(e)
                continue

    def query():
        inputs = {
            "input_ids": tokenizer(["dog"]),
            "pixel_values": img_processor(
                [Image.open("./images.jpeg")]
            ),
        }
        output = model(**inputs)
        result = collection.query(
            query_embeddings=[output.image_embeds[0].tolist()],
            n_results=2
            
        )
        for key, name in enumerate(result["ids"][0]):
            print(key, name)

    create()
    # query()