import os
from typing import List
from urllib.parse import urlparse

import requests
import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from transformer.image import corners, fix_colors, prepare, process
from transformer.utils import save_file, upload_file

URL = os.getenv("URL", default="http://localhost:8000")
UPLOAD_URL = os.getenv("UPLOAD_URL")
EDITOR_URL = os.getenv("EDITOR_URL", "*")
TOKEN = os.getenv("TOKEN")


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

origins = [
    EDITOR_URL,
    "http://localhost",
    "http://127.0.0.1:1234",
    "http://localhost:1234",
    "http://localhost:3333",
]


class Img(BaseModel):
    url: str
    record: str


class Image(BaseModel):
    url: str
    corners: List[int]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload(file: UploadFile):
    contents = await file.read()
    filename = f"./static/images/{file.filename}"
    save_file(filename, contents)
    prep = prepare(filename)
    height, width = prep.shape[:2]
    coordinates = []
    for c in corners(prep):
        coordinates.append(int(c[0]))
        coordinates.append(int(c[1]))
    return {
        "url": f"{URL}/static/images/{file.filename}",
        "coordinates": coordinates,
        "width": width,
        "height": height,
    }


@app.post("/process")
async def process_img(file: UploadFile):
    contents = await file.read()
    filename = f"./static/images/{file.filename}"
    save_file(filename, contents)
    result = process(filename, "static/images")
    return {
        "url": f"{URL}/{result}",
    }


@app.post("/process_url")
async def process_url(img: Img):
    a = urlparse(img.url)
    contents = requests.get(img.url).content
    filename = f"./static/images/{os.path.basename(a.path)}"
    save_file(filename, contents)
    result = process(filename, "static/images")
    upload_file(result, img.record, TOKEN, UPLOAD_URL)
    return {
        "url": f"{URL}/{result}",
    }


@app.post("/fix_colors")
async def fix_colors_service(img: Img):
    a = urlparse(img.url)
    contents = requests.get(img.url).content
    filename = f"./static/images/{os.path.basename(a.path)}"
    save_file(filename, contents)
    result = fix_colors(filename, "static/images")
    upload_file(result, img.record, TOKEN, UPLOAD_URL)
    return {"url": f"{URL}/{result}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, debug=True)
