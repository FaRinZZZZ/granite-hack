"""Terminal client for testing the pipeline without any UI.

    python -m scripts.ask                          # text chat loop
    python -m scripts.ask --speak                  # also play the answer aloud
    python -m scripts.ask --audio clip.wav         # transcribe a file, then answer
"""
import argparse
import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

from app import llm, tts


def play(mp3: bytes) -> None:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
        handle.write(mp3)
        path = handle.name
    player = "afplay" if sys.platform == "darwin" else "ffplay"
    args = [player, path] if player == "afplay" else [player, "-nodisp", "-autoexit", path]
    subprocess.run(args, check=False, capture_output=True)
    Path(path).unlink(missing_ok=True)


async def respond(question: str, session: str, speak: bool) -> None:
    print(f"\n\033[36mคุณ:\033[0m {question}")
    print("\033[32mผู้ช่วย:\033[0m ", end="", flush=True)

    parts = []
    async for delta in llm.stream_answer(session, question):
        parts.append(delta)
        print(delta, end="", flush=True)
    print()

    answer = "".join(parts).strip()
    if speak and answer:
        play(await tts.synthesize(answer))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, help="transcribe this file and ask it")
    parser.add_argument("--speak", action="store_true", help="play answers aloud")
    parser.add_argument("--session", default="cli")
    args = parser.parse_args()

    if args.audio:
        from app import audio as audio_utils
        from app import stt

        samples = audio_utils.decode(args.audio.read_bytes())
        transcript = stt.get_stt().transcribe(samples)
        await respond(transcript.text, args.session, args.speak)
        return

    print("พิมพ์คำถามภาษาไทย (Ctrl+C หรือ 'exit' เพื่อออก)")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.lower() in {"exit", "quit", "q"}:
            return
        if question:
            await respond(question, args.session, args.speak)


if __name__ == "__main__":
    asyncio.run(main())
