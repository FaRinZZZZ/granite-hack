import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import time
import subprocess  # Import subprocess to call main.py


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# sample = dataset[0]["audio"]

def detect_silence_and_process():
    silence_duration = 5  # seconds
    start_time = time.time()

    while True:
        result = pipe("mud.mp3")  # Replace with actual audio input handling
        if result["text"].strip() == "":
            if time.time() - start_time >= silence_duration:
                print("Silence detected for 5 seconds. Calling main.py...")
                subprocess.run(["python", "LLM/main.py"])  # Call main.py
                break
        else:
            start_time = time.time()  # Reset the timer if speech is detected

# Call the function to start monitoring
detect_silence_and_process()
