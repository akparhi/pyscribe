import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import whisperx
from pydub import AudioSegment
from utils import download_file
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.utils import get_stop_words
import nltk
import torch

if not os.path.isdir("files"):
    os.mkdir("files")

app = FastAPI()

sample_rate = 16000
language = "en"
device = "cuda" if torch.cuda.is_available() else "cpu"
# whisper models
# tiny | base | small | medium | large
models = {
    "tiny": whisperx.load_model("tiny.en", device),
    "base": whisperx.load_model("base.en", device)
}
# aligner models
model_a, metadata = whisperx.load_align_model(
    language_code=language, device=device)
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
    phrase = "phrase"
    word = "word"


class ModelEnum(str, Enum):
    tiny = "tiny"
    base = "base"


class TranscribeInput(BaseModel):
    url: str
    model: ModelEnum = ModelEnum.tiny
    accuracy: AccuracyEnum = AccuracyEnum.phrase
    round_accuracy: int = 1
    summarize: bool = False
    max_summary_length: int = 5


@app.post("/speech-to-text")
async def root(input: TranscribeInput):
    # inputs
    url = input.url
    model = models[input.model]
    accuracy = input.accuracy
    round_accuracy = input.round_accuracy
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
    filename = src_filename
    # 2.metadata
    tic_2 = time.perf_counter()
    src_filename_arr = src_filename.split(".")
    src_filename_arr.pop()
    dst_filename = ".".join(src_filename_arr) + '_dst' + '.wav'
    src = AudioSegment.from_file(src_filename)
    data["duration"] = round(src.duration_seconds, round_accuracy)
    src = src.set_frame_rate(sample_rate)
    src = src.set_channels(1)
    src.export(dst_filename, format="wav")
    os.remove(src_filename)
    filename = dst_filename

    # 3.transcription
    tic_3 = time.perf_counter()
    result = model.transcribe(filename, fp16=False)
    data["text"] = result["text"].strip()
    data["phrases"] = [{"b": round(phrase["start"], round_accuracy), "e": round(phrase["end"], round_accuracy), "p": phrase["text"].strip()}
                       for i, phrase in enumerate(result["segments"])]

    # 4.alignment
    tic_4 = time.perf_counter()
    if accuracy == "word":
        result_aligned = whisperx.align(
            result["segments"], model_a, metadata, filename, device)
        data["phrases"] = [{"b": round(phrase["start"], round_accuracy), "e": round(phrase["end"], round_accuracy), "p": phrase["text"].strip()}
                           for i, phrase in enumerate(result_aligned["segments"])]
        data["words"] = [{"b": round(phrase["start"], round_accuracy), "e": round(phrase["end"], round_accuracy), "w": phrase["text"].strip()}
                         for i, phrase in enumerate(result_aligned["word_segments"])]

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
    os.remove(filename)
    task_durations = {
        "1.download": round(tic_2 - tic_1, round_accuracy),
        "2.metadata": round(tic_3 - tic_2, round_accuracy),
        "3.transcription": round(tic_4 - tic_3, round_accuracy),
        "4.alignment": round(tic_5 - tic_4, round_accuracy),
        "5.summarization": round(tic_6 - tic_5, round_accuracy),
        "t.total": round(tic_6 - tic_1, round_accuracy)
    }
    data["task_durations"] = task_durations

    return {"success": True, "data": data}


@app.post("/text-to-speech")
async def root():
    return {"success": True, "data": "Alive"}
