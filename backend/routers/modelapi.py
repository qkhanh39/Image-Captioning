from models.model import generate_caption, load_model
from PIL import Image
import io



model = load_model()

async def image_captioning(file):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    caption = generate_caption(model, image)
    return caption
