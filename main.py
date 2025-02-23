from RealtimeSTT import AudioToTextRecorder
import time
from advanceRAG_api import GraniteAssistant
from RealtimeTTS import TextToAudioStream, PiperEngine, PiperVoice

# Threshold for silence in seconds
SILENCE_THRESHOLD = 4  # Can set to 3.5 or adjust as needed

file_path = "data/Granite Supercenter Detailed Guide.pdf"
assistant = GraniteAssistant(file_path)

# Configure Piper TTS Engine
voice = PiperVoice(
    model_file="piper\en_GB-southern_english_female-low.onnx",
    config_file="piper\en_GB-southern_english_female-low.onnx.json",
)

engine = PiperEngine(
    piper_path="piper\piper.exe",  # Direct path to piper.exe
    voice=voice,
)

stream = TextToAudioStream(engine)  # Initialize the TTS stream


def speak_text(text):
    """Convert text to speech using Piper TTS."""
    stream.feed([text])  # Send text to TTS engine
    stream.play()  # Play the generated audio


def process_text(text):
    """Process user input, get Granite-chan's response, and play only her reply."""
    if text.strip():  # Ensure text is not empty
        # generator = ResponseGenerator()
        result, destination = assistant.run(text)

        # Print the format as requested
        print(f"User: {text}")
        print(f"Granite-chan: {result}")
        print(f"Destination: {destination if destination else 'None'}")

        # Speak only Granite-chan's response (do not speak destination or user input)
        speak_text(result)


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
            # generator = ResponseGenerator()
            result, destination = assistant.run("No input detected, responding to silence.")

            # Print and speak Granite-chan's response
            print("User: No input detected.")
            print(f"Granite-chan: {result}")
            print(f"Destination: {destination if destination else 'None'}")
            speak_text(result)

            last_speech_time = time.time()  # Reset timer after generating response