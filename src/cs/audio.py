from pydub import AudioSegment
from pathlib import Path


class Audio:
    @staticmethod
    def load(file_path: Path):
        format = file_path.suffix.strip(".")
        return AudioSegment.from_file(file_path, format=format)

    @staticmethod
    def save(audio: AudioSegment, file_path, format):
        # 导出为 WAV 文件
        audio.export(file_path, format=format)
