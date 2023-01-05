import os
import time
from enum import Enum

import ffmpeg
import nltk
from fastapi import FastAPI
from pydantic import BaseModel
from stable_whisper import load_model
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.utils import get_stop_words
from torch import cuda

from utils import download_file

if not os.path.isdir("files"):
    os.mkdir("files")

app = FastAPI()

device = "cuda" if cuda.is_available() else "cpu"
print(device)
# whisper models
# tiny | base | small | medium | large
models = {
    "tiny": load_model("tiny")
}
# summarizer model
nltk.download('punkt')
LANGUAGE = "english"

tokenizer = Tokenizer(LANGUAGE)
stemmer = Stemmer(LANGUAGE)
summarizer = Summarizer(stemmer)
summarizer.stop_words = get_stop_words(LANGUAGE)


@app.get("/")
async def root():
    return {"success": True, "data": "Alive"}


class AccuracyEnum(str, Enum):
    NORMAL = "NORMAL"
    HIGH = "HIGH"


class ModelEnum(str, Enum):
    tiny = "tiny"


class TranscribeInput(BaseModel):
    url: str
    model: ModelEnum = ModelEnum.tiny
    accuracy: AccuracyEnum = AccuracyEnum.NORMAL
    summarize: bool = False
    max_summary_length: int = 5


@app.post("/speech-to-text")
async def root(input: TranscribeInput):
    # inputs
    accuracy = input.accuracy
    url = input.url
    model = models[input.model]
    summarize = input.summarize
    max_summary_length = input.max_summary_length

    # response data
    data = {
        "model": input.model,
        "accuracy": accuracy,
        "task_durations": {},
        "duration": "",
        "text": "",
        "summary": "",
        "phrases": [],
        "words": []
    }

    # API
    # 1.download
    tic_1 = time.perf_counter()
    src_filename = download_file(url)

    # 2.metadata
    tic_2 = time.perf_counter()
    data["duration"] = round(
        float(ffmpeg.probe(src_filename)["format"]["duration"]), 2)

    # 3.transcription
    tic_3 = time.perf_counter()
    result = model.transcribe(src_filename, fp16=False, suppress_silence=True,
                              ts_num=16, lower_quantile=0.05, lower_threshold=0.1)
    data["text"] = result["text"].strip()
    data["phrases"] = [{"b": round(phrase["start"], 1), "e": round(phrase["end"], 1), "t": phrase["text"].strip()}
                       for phrase in result["segments"]]

    # 4.alignment
    tic_4 = time.perf_counter()
    if accuracy == "HIGH":
        stab_segments = result["segments"]
        data["phrases"] = [{"b": round(phrase["start"], 1), "e": round(phrase["end"], 1), "t": phrase["text"].strip(), "c": [
            {"t": word["word"].strip(), "c": round(word["confidence"], 2)} for word in phrase['whole_word_timestamps']]} for phrase in result["segments"]]
        words = []
        for segment in stab_segments:
            words += segment['whole_word_timestamps']

        words_len = len(words)
        data['words'] = [{"b": round(word["timestamp"], 1), "e": round(words[i+1]["timestamp"], 1) if (i < (words_len - 1)) else (data["duration"] if ((data["duration"] - round(
            word["timestamp"], 1)) <= 1)else round(word["timestamp"], 1) + 1), "t": word["word"].strip(), "c": round(word["confidence"], 2)} for i, word in enumerate(words)]
        os.remove("filename.pickle")

    # 5.summarization
    tic_5 = time.perf_counter()
    if summarize:
        parser = PlaintextParser.from_string(result["text"].strip(), tokenizer)
        no_of_sentences = len(parser.document.sentences)
        summary = summarizer(parser.document, min(
            max(no_of_sentences//10, 1), max_summary_length))
        data["summary"] = " ".join([str(sentence).strip()
                                   for sentence in summary])

    # 6.cleanups
    tic_6 = time.perf_counter()
    os.remove(src_filename)
    task_durations = {
        "1.download": round(tic_2 - tic_1, 1),
        "2.metadata": round(tic_3 - tic_2, 1),
        "3.transcription": round(tic_4 - tic_3, 1),
        "4.alignment": round(tic_5 - tic_4, 1),
        "5.summarization": round(tic_6 - tic_5, 1),
        "t.total": round(tic_6 - tic_1, 1)
    }
    data["task_durations"] = task_durations

    return {"success": True, "data": data}


@app.post("/text-to-speech")
async def root():
    return {"success": True, "data": "Alive"}
