import os
from PIL import Image

from ultralytics import YOLO
import cv2
#from src.scripts.utils import create_video


def process_video(name: str, margin=50):
    """Обрабатывает видео."""
    #output_video = "output/processed_video.mp4"
    model = YOLO('yolo11s.pt')  # Инициализация модели

    # Обрезка фото по координатам с добовлением отступа
    results = model(f'input/{name}.mp4',
                    classes=0,
                    save=True, conf=0.25, iou=0.7, max_det=1, show_labels=False, save_frames=True, line_width=1,
                    project='output', name='frames'
                    )
    out = cv2.VideoWriter(f'output/{name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (1920, 1080))

    # Добавляем отступ
    for i in range(1, len(os.listdir(f'output/frames/{name}_frames')) + 1):
        size = (results[i - 1].boxes.xyxy).tolist()
        x1 = int(size[0][0]) - margin
        x2 = int(size[0][2]) + margin
        y1 = int(size[0][1]) - margin
        y2 = int(size[0][3]) + margin

        # Обрезание кадра по значениям boxes + отступ(margin=50)
        img = cv2.imread(f'output/frames/{name}_frames/{i}.jpg')
        cropped = img[y1:y2, x1:x2]
        cv2.imwrite(f"output/frames/{name}_frames/{i}.jpg", cropped)

        # Полученные фото добовляю в пустое фото по координатам (700, 200) + сохраняю в отдельную папку
        image = Image.new("RGB", (1920, 1080))
        s = Image.open(f'output/frames/{name}_frames/{i}.jpg')
        image.paste(s, (700, 200))
        image.save(f'frames_for_video/Image{i}.jpg')
        img = cv2.imread(f'frames_for_video/Image{i}.jpg')
        out.write(img)

    out.release()

    #create_video(input_path, output_video)
    print("✅ Обработка завершена.")
