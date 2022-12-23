import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import whisperx
import ffmpeg
from utils import download_file

if not os.path.isdir("files"):
    os.mkdir("files")

if not os.path.isdir("samples"):
    os.mkdir("samples")

app = FastAPI()

language = "en"
device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
# whisper models
# tiny | base | small | medium | large
models = {
    "tiny": whisperx.load_model("tiny.en"),
    "base": whisperx.load_model("base.en")
}
# aligner models
model_a, metadata = whisperx.load_align_model(
    language_code=language, device=device)


@app.get("/")
async def root():
    return {"success": True, "data": "Alive"}


class AccuracyEnum(str, Enum):
    phrase = "phrase"
    word = "word"


class ModelEnum(str, Enum):
    tiny = "tiny"
    base = "base"


class TranscribeInput(BaseModel):
    url: str
    model: ModelEnum = ModelEnum.tiny
    accuracy: AccuracyEnum = AccuracyEnum.phrase
    debug: bool = False


@app.post("/speech-to-text")
async def root(input: TranscribeInput):
    # inputs
    accuracy = input.accuracy
    url = input.url
    model = models[input.model]
    debug = input.debug

    # response data
    data = {
        "model": input.model,
        "accuracy": accuracy,
        "task_durations": {},
        "duration": "",
        "phrases": [],
        "words": []
    }

    # API
    # 1.download
    tic_download = time.perf_counter()
    src_filename = download_file(url)
    toc_download = time.perf_counter()

    # 2.metadata
    tic_metadata = time.perf_counter()
    data["duration"] = ffmpeg.probe(src_filename)["format"]["duration"]
    toc_metadata = time.perf_counter()

    # 3.transcription
    tic_transcription = time.perf_counter()
    result = model.transcribe(src_filename, fp16=False)
    if debug:
        data["text"] = result["text"]
    phrases = [{"b": round(phrase["start"], 1), "e": round(phrase["end"], 1), "p": phrase["text"]}
               for i, phrase in enumerate(result["segments"])]
    data["phrases"] = phrases
    toc_transcription = time.perf_counter()

    # 4.alignment
    tic_alignment = time.perf_counter()
    if accuracy == "word":
        result_aligned = whisperx.align(
            result["segments"], model_a, metadata, src_filename, device)
        data["phrases"] = [{"b": round(phrase["start"], 1), "e": round(phrase["end"], 1), "p": phrase["text"]}
                           for i, phrase in enumerate(result_aligned["segments"])]
        data["words"] = [{"b": round(phrase["start"], 1), "e": round(phrase["end"], 1), "w": phrase["text"]}
                         for i, phrase in enumerate(result_aligned["word_segments"])]
        if debug:
            data["nonaligned_phrases"] = phrases
    toc_alignment = time.perf_counter()

    # 5.cleanups
    os.remove(src_filename)
    task_durations = {
        "1.download": round(toc_download - tic_download, 1),
        "2.metadata": round(toc_metadata - tic_metadata, 1),
        "3.transcription": round(toc_transcription - tic_transcription, 1),
        "4.alignment": round(toc_alignment - tic_alignment, 1),
        "t.total": round(toc_alignment - tic_download, 1)
    }
    task_durations["t.total"] = round(toc_alignment - tic_download, 1)
    data["task_durations"] = task_durations

    return {"success": True, "data": data}


@app.post("/text-to-speech")
async def root():
    return {"success": True, "data": "Alive"}
