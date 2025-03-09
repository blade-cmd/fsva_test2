from src.scripts.utils import create_video


def process_video(input_path: str):
    """Обрабатывает видео."""
    output_video = "output/processed_video.mp4"

    create_video(input_path, output_video)
    print("✅ Обработка завершена.")
