from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from routers.modelapi import image_captioning
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/image-captioning")
async def generate_caption_for_image(file: UploadFile = File(...)):
    caption = await image_captioning(file)
    return JSONResponse(content={"caption": caption})