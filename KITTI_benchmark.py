import multiprocessing
import cv2
import time
from ultralytics import YOLO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import shutil
from PIL import Image

def main():
    # Get dataset
    base_dir = Path("C:/Users/nouveau/Downloads")
    img_path = base_dir / 'data_object_image_2' / 'training' / 'image_2'
    label_path = Path("C:/Users/nouveau/Downloads/kitti-yolo-labels/labels")
    with open('C:/Users/nouveau/Downloads/kitti-yolo-labels/classes.json','r') as f:
        classes = json.load(f)

    # Set pairs for images and labels
    ims = sorted(list(img_path.glob('*')))
    labels = sorted(list(label_path.glob('*')))
    pairs = list(zip(ims,labels))

    # Split train and test sets
    train, test = train_test_split(pairs,test_size=0.2,shuffle=True)
    print(len(train), len(test))

    # Store train and test sets
    train_path = Path('C:/Users/nouveau/Downloads/kitti-yolo/train').resolve()
    train_path.mkdir(exist_ok=True)
    valid_path = Path('C:/Users/nouveau/Downloads/kitti-yolo/valid').resolve()
    valid_path.mkdir(exist_ok=True)

    for t_img, t_lb in tqdm(train, desc='Copy train images to directory'):
        im_path = train_path / t_img.name
        lb_path = train_path / t_lb.name
        shutil.copy(t_img,im_path)
        shutil.copy(t_lb,lb_path)

    for t_img, t_lb in tqdm(test, desc='Copy valid images to directory '):
        im_path = valid_path / t_img.name
        lb_path = valid_path / t_lb.name
        shutil.copy(t_img,im_path)
        shutil.copy(t_lb,lb_path)

    # YAML file for the data
    yaml_file = 'names:\n'
    yaml_file += '\n'.join(f'- {c}' for c in classes)
    yaml_file += f'\nnc: {len(classes)}'
    yaml_file += f'\ntrain: {str(train_path)}\nval: {str(valid_path)}'
    with open('C:/Users/nouveau/Downloads/kitti-yolo/kitti.yaml','w') as f:
        f.write(yaml_file)

    # model = YOLO('yolov8n.yaml')
    model = YOLO('yolov8n.pt')

    # Fine-tuning the model
    train_results = model.train(
        data='C:/Users/nouveau/Downloads/kitti-yolo/kitti.yaml',
        epochs=10,
        patience=3,
        mixup=0.1,
        project='yolov8n-kitti',
        device=0
    )

    # Validate
    valid_results = model.val()

def show_result():
    plt.figure(figsize=(10,20))
    plt.imshow(Image.open('yolov8n-kitti/train3/results.png'))
    plt.axis('off')
    plt.show()

def estimate_FPS():
    print("\nRunning inference speed test...")

    # Model
    model = YOLO('yolov8m.pt')

    # Load Images
    valid_path = Path('C:/Users/nouveau/Downloads/kitti-yolo/valid').resolve()
    valid_path.mkdir(exist_ok=True)
    val_img_paths = sorted(list(valid_path.glob("*.png")))[:100]  # 可改数量
    total_time = 0.0

    for img_path in tqdm(val_img_paths, desc="Running inference"):
        img = cv2.imread(str(img_path))
        start = time.time()
        _ = model.predict(source=img, save=False, verbose=False, device=0)
        end = time.time()
        total_time += (end - start)

    avg_time = total_time / len(val_img_paths)
    fps = 1 / avg_time

    print(f"\nInference Images: {len(val_img_paths)}")
    print(f"Average Inference Time: {avg_time * 1000:.2f} ms")
    print(f"FPS: {fps:.2f} frames/sec")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # main()
    estimate_FPS()
    show_result()