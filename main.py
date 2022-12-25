import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import whisper
import ffmpeg
from utils import download_file
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.utils import get_stop_words
import nltk
from stable_whisper import load_model, stabilize_timestamps

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
    "tiny": load_model("tiny"),
    "base": load_model("base")
}
# aligner models
# model_a, metadata = whisperx.load_align_model(
#     language_code=language, device=device)
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
        "aligned_phrases": [],
        "aligned_words": []
    }

    # API
    # 1.download
    tic_1 = time.perf_counter()
    src_filename = download_file(url)

    # 2.metadata
    tic_2 = time.perf_counter()
    data["duration"] = ffmpeg.probe(src_filename)["format"]["duration"]

    # 3.transcription
    tic_3 = time.perf_counter()
    result = model.transcribe(src_filename, fp16=False, suppress_silence=True,
                              ts_num=16, lower_quantile=0.05, lower_threshold=0.1)
    data["text"] = result["text"].strip()
    phrases = [{"b": round(phrase["start"], 1), "e": round(phrase["end"], 1), "p": phrase["text"].strip()}
               for i, phrase in enumerate(result["segments"])]
    data["phrases"] = phrases

    # 4.alignment
    tic_4 = time.perf_counter()
    if accuracy == "word":
        stab_segments = result["segments"]
        aligned_words = []
        for segment in stab_segments:
            aligned_words += segment['whole_word_timestamps']

        data['aligned_words'] = [{"t": round(word["timestamp"], 2), "w": word["word"].strip()}
                                 for i, word in enumerate(aligned_words)]

        # result_aligned = whisperx.align(
        #     result["segments"], model_a, metadata, src_filename, device)
        # data["aligned_phrases"] = [{"b": round(phrase["start"], 1), "e": round(phrase["end"], 1), "p": phrase["text"].strip()}
        #                            for i, phrase in enumerate(result_aligned["segments"])]
        # data["aligned_words"] = [{"b": round(phrase["start"], 1), "e": round(phrase["end"], 1), "w": phrase["text"].strip()}
        #                          for i, phrase in enumerate(result_aligned["word_segments"])]

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
