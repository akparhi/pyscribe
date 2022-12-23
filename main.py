import os
import time
from fastapi import FastAPI
import pytube
import whisper

if not os.path.isdir("files"):
    os.mkdir("files")


app = FastAPI()
# tiny | base | small | medium | large
tiny_model = whisper.load_model("tiny.en")
base_model = whisper.load_model("base.en")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/text")
async def root():

    # inputs
    filename = 'test'
    video = 'https://www.youtube.com/watch?v=MLU4E62c6Vs'

    # API
    tic_download = time.perf_counter()
    data = pytube.YouTube(video)
    audio = data.streams.get_audio_only()
    audio.download(output_path="files", filename=filename+'.wav')
    toc_download = time.perf_counter()
    tic_transcription = time.perf_counter()
    text = tiny_model.transcribe('files/'+filename+'.wav')
    toc_transcription = time.perf_counter()

    task_durations = {
        "0.download": round(toc_download - tic_download, 1),
        "1.transcription": round(toc_transcription - tic_transcription, 1),
        "t.total": round(toc_transcription - tic_download, 1)
    }
    return {"success": True, "taskDurations": task_durations, "text": text['text']}
