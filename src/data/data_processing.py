import math
import os

from PIL import Image

from src.config import DATA_DIR, DATA_PROCESSED_DIR, MARGIN_RATIO, TARGET_SIZE


def process_images():
    """Process cat images by centering them with given points and cropping.
    Save results to DATA_PROCESSED_DIR"""

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    for i in range(7):
        data_dir = DATA_DIR + str(i)
        print(f"Processing images in {data_dir}")
        for filename in os.listdir(data_dir):
            if not filename.endswith(".jpg"):
                continue
            img_path = os.path.join(data_dir, filename)
            cat_info_path = img_path + ".cat"

            if not os.path.exists(cat_info_path):
                print(f"File {cat_info_path} does not exist.")
                continue

            with open(cat_info_path, "r") as f:
                cat_info = list(map(int, f.read().split()))

            points = cat_info[1:]
            left_eye_x, left_eye_y = points[0], points[1]
            right_eye_x, right_eye_y = points[2], points[3]

            # rotation
            dy = right_eye_y - left_eye_y
            dx = right_eye_x - left_eye_x

            angle = math.degrees(math.atan2(dy, dx))
            xs = points[0::2]
            ys = points[1::2]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            face_center_x = (min_x + max_x) // 2
            face_center_y = (min_y + max_y) // 2

            face_width = max_x - min_x
            face_height = max_y - min_y
            box_size = int(max(face_width, face_height) * (1 + MARGIN_RATIO))

            with Image.open(img_path) as img:
                img = img.convert("RGB")

                rotated_img = img.rotate(
                    angle,
                    center=(face_center_x, face_center_y),
                    resample=Image.Resampling.BICUBIC,
                )
                left = face_center_x - (box_size // 2)
                top = face_center_y - (box_size // 2)
                right = face_center_x + (box_size // 2)
                bottom = face_center_y + (box_size // 2)

                cropped_img = rotated_img.crop((left, top, right, bottom))
                final_img = cropped_img.resize(
                    size=(TARGET_SIZE, TARGET_SIZE), resample=Image.Resampling.BILINEAR
                )

                save_path = os.path.join(DATA_PROCESSED_DIR, filename)
                final_img.save(save_path, quality=95)
