from RealtimeSTT import AudioToTextRecorder
import time
from main import ResponseGenerator

# Threshold for silence in seconds
SILENCE_THRESHOLD = 3

def process_text(text):
    if text.strip():  # Ensure text is not empty
        generator = ResponseGenerator()
        result = generator.run(text)
        print(f"Generated response: {result}")

if __name__ == '__main__':
    recorder = AudioToTextRecorder(language='en')

    last_speech_time = time.time()

    while True:
        text = recorder.text()  # Capture the speech-to-text output

        if text.strip():
            last_speech_time = time.time()  # Reset silence timer
            process_text(text)

        elif time.time() - last_speech_time >= SILENCE_THRESHOLD:
            print("Silence detected. Generating response...")
            generator = ResponseGenerator()
            result = generator.run("No input detected, responding to silence.")
            print(f"Generated response: {result}")
            last_speech_time = time.time()  # Reset timer after generating response