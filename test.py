if __name__ == "__main__":
    from RealtimeTTS import TextToAudioStream, PiperEngine, PiperVoice

    def dummy_generator():
        yield "sup ma nigga bitch as stupid piece of shit, baka baka ,sussy baka"

    voice = PiperVoice(
        model_file="D:/WORK/Compettition/GRANITE HACKATHON/granite/piper/en_GB-southern_english_female-low.onnx",
        config_file="D:/WORK/Compettition/GRANITE HACKATHON/granite/piper/en_GB-southern_english_female-low.onnx.json",
    )

    engine = PiperEngine(
        piper_path="D:\WORK\Compettition\GRANITE HACKATHON\granite\piper\piper.exe",
        voice=voice,
    )

    stream = TextToAudioStream(engine)
    stream.feed(dummy_generator())
    stream.play()