from RealtimeSTT import AudioToTextRecorder
import time
from api import ResponseGenerator

# Threshold for silence in seconds
SILENCE_THRESHOLD = 4  # Can set to 3.5 or adjust as needed

def process_text(text):
    if text.strip():  # Ensure text is not empty
        generator = ResponseGenerator()
        result, destination = generator.run(text)
        # Print the format as requested: "User: xxx", "Granite-Chan: xxx", "Destination: xxx"
        print(f"User: {text}")
        print(f"Granite-chan: {result}")
        print(f"Destination: {destination if destination else 'None'}")

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
            result, destination = generator.run("No input detected, responding to silence.")
            # Print the format for the silence response
            print("User: No input detected.")
            print(f"Granite-chan: {result}")
            print(f"Destination: {destination if destination else 'None'}")
            last_speech_time = time.time()  # Reset timer after generating response
