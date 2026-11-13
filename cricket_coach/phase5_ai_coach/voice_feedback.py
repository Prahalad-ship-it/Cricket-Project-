"""Optional text-to-speech feedback generation."""

from pathlib import Path


class VoiceFeedback:
    """Generate a local speech file when an optional TTS backend is installed."""

    def generate(self, text: str, output_path: str | Path) -> Path:
        """Synthesize speech via pyttsx3 or raise a clear dependency error."""
        if not text.strip():
            raise ValueError("Feedback text cannot be empty.")
        try:
            import pyttsx3
        except ImportError as error:
            raise ImportError("Install pyttsx3 to generate offline voice feedback.") from error
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        engine = pyttsx3.init()
        try:
            engine.save_to_file(text, str(destination))
            engine.runAndWait()
        finally:
            engine.stop()
        if not destination.is_file():
            raise OSError(f"TTS did not create an audio file: {destination}")
        return destination


if __name__ == "__main__":
    assert VoiceFeedback
    print("Voice module checks passed; install pyttsx3 to synthesize audio.")
