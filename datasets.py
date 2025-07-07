import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import extra_augs
import pandas as pd
import shutil

# Глобальные директории для результатов
RESULTS_DIR = 'results'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
TESTS_DIR = os.path.join(RESULTS_DIR, 'tests')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TESTS_DIR, exist_ok=True)

class CustomImageDataset(Dataset):
    """Кастомный датасет для работы с папками классов"""
    
    def __init__(self, root_dir, transform=None, target_size=(224, 224)):
        """
        Args:
            root_dir (str): Путь к папке с классами
            transform: Аугментации для изображений
            target_size (tuple): Размер для ресайза изображений
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        
        # Получаем список классов (папок)
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Собираем все пути к изображениям
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Загружаем изображение
        image = Image.open(img_path).convert('RGB')
        
        # Ресайзим изображение
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Применяем аугментации
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_names(self):
        """Возвращает список имен классов"""
        return self.classes 

# Задание 1: Визуализация аугментаций
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision import transforms
    import random

    # Пути к 5 изображениям из разных классов
    sample_paths = [
        'data/train/Гароу/02b8c85727aa4d9978ae2cb518affb2a.jpg',
        'data/train/Генос/08a7ce2efd7b8b95a254833da966265f.jpg',
        'data/train/Сайтама/05eaf704e653c1cedad80dedc0e30824.jpg',
        'data/train/Соник/00badad3b4e440bc1264bf25f0ea65f2.jpg',
        'data/train/Татсумаки/06c27614479e71583feac6d0e98f3073.jpg',
    ]

    # Оригинальные изображения
    orig_images = [Image.open(p).convert('RGB') for p in sample_paths]

    # Аугментации
    aug_transforms = {
        'RandomHorizontalFlip': transforms.RandomHorizontalFlip(p=1.0),
        'RandomCrop': transforms.RandomCrop(size=180),
        'ColorJitter': transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        'RandomRotation': transforms.RandomRotation(degrees=45),
        'RandomGrayscale': transforms.RandomGrayscale(p=1.0),
    }
    all_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomCrop(size=180),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        transforms.RandomRotation(degrees=45),
        transforms.RandomGrayscale(p=1.0),
    ])

    # Визуализация
    for idx, img in enumerate(orig_images):
        fig, axes = plt.subplots(1, 7, figsize=(20, 4))
        axes[0].imshow(img)
        axes[0].set_title('Оригинал')
        axes[0].axis('off')
        for i, (name, aug) in enumerate(aug_transforms.items()):
            aug_img = aug(img)
            axes[i+1].imshow(aug_img, cmap=None if name != 'RandomGrayscale' else 'gray')
            axes[i+1].set_title(name)
            axes[i+1].axis('off')
        # Все аугментации вместе
        all_img = all_aug(img)
        axes[-1].imshow(all_img, cmap='gray' if all_img.mode == 'L' else None)
        axes[-1].set_title('Все вместе')
        axes[-1].axis('off')
        plt.suptitle(f'Класс: {sample_paths[idx].split("/")[-2]}')
        plt.tight_layout()
        plt.show() 

# Задание 2: Кастомные аугментации и сравнение
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from PIL import ImageFilter, ImageEnhance
    import extra_augs
    import torch

    # Кастомные аугментации
    class RandomBlur:
        def __init__(self, p=0.7, radius_range=(1, 3)):
            self.p = p
            self.radius_range = radius_range
        def __call__(self, img):
            if random.random() > self.p:
                return img
            radius = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))

    class RandomPerspective:
        def __init__(self, p=0.7, distortion_scale=0.5):
            self.p = p
            self.distortion_scale = distortion_scale
        def __call__(self, img):
            if random.random() > self.p:
                return img
            return transforms.functional.perspective(
                img,
                startpoints=[(0,0),(img.width,0),(img.width,img.height),(0,img.height)],
                endpoints=[
                    (random.randint(0, int(img.width*self.distortion_scale)), random.randint(0, int(img.height*self.distortion_scale))),
                    (img.width - random.randint(0, int(img.width*self.distortion_scale)), random.randint(0, int(img.height*self.distortion_scale))),
                    (img.width - random.randint(0, int(img.width*self.distortion_scale)), img.height - random.randint(0, int(img.height*self.distortion_scale))),
                    (random.randint(0, int(img.width*self.distortion_scale)), img.height - random.randint(0, int(img.height*self.distortion_scale)))
                ]
            )

    class RandomBrightnessContrast:
        def __init__(self, p=0.7, brightness_range=(0.5, 1.5), contrast_range=(0.5, 1.5)):
            self.p = p
            self.brightness_range = brightness_range
            self.contrast_range = contrast_range
        def __call__(self, img):
            if random.random() > self.p:
                return img
            enhancer_b = ImageEnhance.Brightness(img)
            img = enhancer_b.enhance(random.uniform(*self.brightness_range))
            enhancer_c = ImageEnhance.Contrast(img)
            img = enhancer_c.enhance(random.uniform(*self.contrast_range))
            return img

    # Пути к 3 изображениям из разных классов
    sample_paths = [
        'data/train/Гароу/02b8c85727aa4d9978ae2cb518affb2a.jpg',
        'data/train/Генос/08a7ce2efd7b8b95a254833da966265f.jpg',
        'data/train/Сайтама/05eaf704e653c1cedad80dedc0e30824.jpg',
    ]
    orig_images = [Image.open(p).convert('RGB') for p in sample_paths]

    # Кастомные аугментации
    custom_augs = {
        'RandomBlur': RandomBlur(p=1.0),
        'RandomPerspective': RandomPerspective(p=1.0),
        'RandomBrightnessContrast': RandomBrightnessContrast(p=1.0)
    }

    # Готовые аугментации из extra_augs.py 
    def tensor_to_pil(img_tensor):
        img_tensor = img_tensor.detach().cpu().clamp(0, 1)
        return transforms.ToPILImage()(img_tensor)

    ready_augs = {
        'AddGaussianNoise': lambda img: tensor_to_pil(extra_augs.AddGaussianNoise(std=0.2)(transforms.ToTensor()(img))),
        'RandomErasingCustom': lambda img: tensor_to_pil(extra_augs.RandomErasingCustom(p=1.0)(transforms.ToTensor()(img))),
        'AutoContrast': lambda img: tensor_to_pil(extra_augs.AutoContrast(p=1.0)(transforms.ToTensor()(img)))
    }

    for idx, img in enumerate(orig_images):
        fig, axes = plt.subplots(1, 1+len(custom_augs)+len(ready_augs), figsize=(18, 4))
        axes[0].imshow(img)
        axes[0].set_title('Оригинал')
        axes[0].axis('off')
        # Кастомные аугментации
        for i, (name, aug) in enumerate(custom_augs.items()):
            aug_img = aug(img)
            axes[i+1].imshow(aug_img)
            axes[i+1].set_title(f'Кастом: {name}')
            axes[i+1].axis('off')
        # Готовые аугментации
        for j, (name, aug) in enumerate(ready_augs.items()):
            aug_img = aug(img)
            axes[len(custom_augs)+1+j].imshow(aug_img)
            axes[len(custom_augs)+1+j].set_title(f'Готовая: {name}')
            axes[len(custom_augs)+1+j].axis('off')
        plt.suptitle(f'Класс: {sample_paths[idx].split("/")[-2]}')
        plt.tight_layout()
        plt.show() 

# Задание 3: Анализ датасета
if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from PIL import Image
    import numpy as np

    data_dir = 'data/train'
    class_counts = {}
    img_shapes = []
    class_names = []

    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        count = 0
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(class_path, img_name)
                try:
                    with Image.open(img_path) as img:
                        img_shapes.append(img.size)  
                except Exception as e:
                    continue
                count += 1
        class_counts[class_name] = count
        class_names.append(class_name)

    # Количество изображений по классам
    print('Количество изображений по классам:')
    for k, v in class_counts.items():
        print(f'{k}: {v}')

    # Размеры изображений
    widths, heights = zip(*img_shapes)
    min_size = (min(widths), min(heights))
    max_size = (max(widths), max(heights))
    mean_size = (int(np.mean(widths)), int(np.mean(heights)))
    print(f'Минимальный размер: {min_size}')
    print(f'Максимальный размер: {max_size}')
    print(f'Средний размер: {mean_size}')

    # Визуализация распределения размеров
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(widths, heights, alpha=0.5)
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Распределение размеров изображений')

    # Гистограмма по классам
    plt.subplot(1, 2, 2)
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xticks(rotation=45)
    plt.ylabel('Количество изображений')
    plt.title('Гистограмма по классам')
    plt.tight_layout()
    plt.show() 

# Задание 4: Pipeline аугментаций
class AugmentationPipeline:
    """Pipeline для последовательного применения аугментаций к изображению."""
    def __init__(self):
        self.augmentations = []  # Список (name, aug)
    def add_augmentation(self, name, aug):
        """Добавить аугментацию по имени."""
        self.augmentations.append((name, aug))
    def remove_augmentation(self, name):
        """Удалить аугментацию по имени."""
        self.augmentations = [(n, a) for n, a in self.augmentations if n != name]
    def apply(self, image):
        """Применить pipeline к изображению."""
        img = image
        for _, aug in self.augmentations:
            img = aug(img)
        return img
    def get_augmentations(self):
        """Получить список имён аугментаций."""
        return [n for n, _ in self.augmentations]

if __name__ == "__main__":
    import os
    from torchvision import transforms
    from PIL import Image, ImageFilter, ImageEnhance
    import shutil

    # Конфигурации 
    def get_light():
        pipe = AugmentationPipeline()
        pipe.add_augmentation('hflip', transforms.RandomHorizontalFlip(p=1.0))
        return pipe
    def get_medium():
        pipe = AugmentationPipeline()
        pipe.add_augmentation('hflip', transforms.RandomHorizontalFlip(p=1.0))
        pipe.add_augmentation('color', transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3))
        pipe.add_augmentation('rotate', transforms.RandomRotation(degrees=20))
        return pipe
    def get_heavy():
        pipe = AugmentationPipeline()
        pipe.add_augmentation('hflip', transforms.RandomHorizontalFlip(p=1.0))
        pipe.add_augmentation('color', transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2))
        pipe.add_augmentation('rotate', transforms.RandomRotation(degrees=45))
        pipe.add_augmentation('blur', lambda img: img.filter(ImageFilter.GaussianBlur(radius=2)))
        pipe.add_augmentation('perspective', transforms.RandomPerspective(distortion_scale=0.5, p=1.0))
        return pipe

    configs = {
        'light': get_light(),
        'medium': get_medium(),
        'heavy': get_heavy(),
    }

    # Применение к train и сохранение 
    src_dir = 'data/train'
    out_base = 'augmented_train'
    os.makedirs(out_base, exist_ok=True)

    for cfg_name, pipeline in configs.items():
        out_dir = os.path.join(out_base, cfg_name)
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        for class_name in os.listdir(src_dir):
            class_path = os.path.join(src_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            out_class = os.path.join(out_dir, class_name)
            os.makedirs(out_class, exist_ok=True)
            for img_name in os.listdir(class_path):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    continue
                img_path = os.path.join(class_path, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')
                    aug_img = pipeline.apply(img)
                    aug_img.save(os.path.join(out_class, img_name))
                except Exception as e:
                    print(f'Ошибка с {img_path}: {e}')
    print('Аугментированные датасеты сохранены в папке augmented_train/')

# Задание 5: Эксперимент с размерами изображений
if __name__ == "__main__":
    import time
    import psutil
    import matplotlib.pyplot as plt
    from torchvision import transforms
    import gc

    sizes = [64, 128, 224, 512]
    n_images = 100
    results = {'size': [], 'time': [], 'memory': []}

    # Простая аугментация для эксперимента
    aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

    # Собираем пути к первым 100 изображениям train
    img_paths = []
    train_dir = 'data/train'
    for class_name in sorted(os.listdir(train_dir)):
        class_path = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_paths.append(os.path.join(class_path, img_name))
            if len(img_paths) >= n_images:
                break
        if len(img_paths) >= n_images:
            break

    for size in sizes:
        gc.collect()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024**2  # MB
        t0 = time.time()
        images = []
        for path in img_paths:
            img = Image.open(path).convert('RGB').resize((size, size))
            img = aug(img)
            images.append(img)
        t1 = time.time()
        mem_after = process.memory_info().rss / 1024**2  # MB
        results['size'].append(size)
        results['time'].append(t1 - t0)
        results['memory'].append(mem_after - mem_before)
        print(f'Size: {size}x{size} | Time: {t1-t0:.2f}s | Mem: {mem_after-mem_before:.2f} MB')
        del images
        gc.collect()

    # Графики
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(results['size'], results['time'], marker='o')
    plt.xlabel('Размер (px)')
    plt.ylabel('Время (сек)')
    plt.title('Время обработки 100 изображений')
    plt.grid()
    plt.subplot(1,2,2)
    plt.plot(results['size'], results['memory'], marker='o', color='orange')
    plt.xlabel('Размер (px)')
    plt.ylabel('Память (MB)')
    plt.title('Память для 100 изображений')
    plt.grid()
    plt.tight_layout()
    plt.show() 

# Задание 6: Дообучение предобученной модели
if __name__ == "__main__":
    import torch
    import torchvision
    from torchvision import transforms, models
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Подготовка датасета
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = CustomImageDataset('data/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    val_dataset = None
    val_loader = None
    if os.path.exists('data/val') and len(os.listdir('data/val')) > 0:
        val_dataset = CustomImageDataset('data/val', transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)

    # Загрузка предобученной модели
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = torch.nn.Linear(model.fc.in_features, len(train_dataset.get_class_names()))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    n_epochs = 5
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            _, preds = torch.max(out, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation 
        if val_loader is not None:
            model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    loss = loss_fn(out, y)
                    running_loss += loss.item() * x.size(0)
                    _, preds = torch.max(out, 1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            val_loss = running_loss / total
            val_acc = correct / total
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            print(f'Epoch {epoch+1}/{n_epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f}')
        else:
            print(f'Epoch {epoch+1}/{n_epochs} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}')

    # Визуализация процесса обучения
    epochs = np.arange(1, n_epochs+1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Train loss')
    if val_losses:
        plt.plot(epochs[:len(val_losses)], val_losses, label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(epochs, train_accs, label='Train acc')
    if val_accs:
        plt.plot(epochs[:len(val_accs)], val_accs, label='Val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'finetune_training.png'))
    plt.close()
    # Сохранение финальных метрик
    with open(os.path.join(TESTS_DIR, 'finetune_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Final train loss: {train_losses[-1]:.4f}\n')
        if val_losses:
            f.write(f'Final val loss: {val_losses[-1]:.4f}\n')
        f.write(f'Final train acc: {train_accs[-1]:.4f}\n')
        if val_accs:
            f.write(f'Final val acc: {val_accs[-1]:.4f}\n')

    # После обучения: предсказания на val и сохранение результатов
    if val_loader is not None:
        val_results = []
        class_names = train_dataset.get_class_names()
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                outputs = model(x)
                _, preds = torch.max(outputs, 1)
                for i in range(x.size(0)):
                    # Получаем имя файла (через val_dataset.images)
                    idx = len(val_results) + i
                    if idx < len(val_dataset.images):
                        img_path = val_dataset.images[idx]
                        img_name = os.path.basename(img_path)
                        val_results.append({
                            'filename': img_name,
                            'true_label': class_names[y[i].item()],
                            'predicted_label': class_names[preds[i].item()]
                        })
        # Сохраняем в CSV
        df = pd.DataFrame(val_results)
        df.to_csv(os.path.join('data', 'val', 'predictions.csv'), index=False, encoding='utf-8')
        print('Валидационные предсказания сохранены в data/val/predictions.csv')

# Анализ датасета: сохраняем графики и метрики
if __name__ == "__main__":
    # Визуализация распределения размеров
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(widths, heights, alpha=0.5)
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Распределение размеров изображений')
    plt.subplot(1, 2, 2)
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xticks(rotation=45)
    plt.ylabel('Количество изображений')
    plt.title('Гистограмма по классам')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'dataset_analysis.png'))
    plt.close()
    # Сохраняем метрики
    with open(os.path.join(TESTS_DIR, 'dataset_stats.txt'), 'w', encoding='utf-8') as f:
        f.write('Количество изображений по классам:\n')
        for k, v in class_counts.items():
            f.write(f'{k}: {v}\n')
        f.write(f'\nМинимальный размер: {min_size}\n')
        f.write(f'Максимальный размер: {max_size}\n')
        f.write(f'Средний размер: {mean_size}\n')

# Эксперимент с размерами
if __name__ == "__main__":
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(results['size'], results['time'], marker='o')
    plt.xlabel('Размер (px)')
    plt.ylabel('Время (сек)')
    plt.title('Время обработки 100 изображений')
    plt.grid()
    plt.subplot(1,2,2)
    plt.plot(results['size'], results['memory'], marker='o', color='orange')
    plt.xlabel('Размер (px)')
    plt.ylabel('Память (MB)')
    plt.title('Память для 100 изображений')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'resize_experiment.png'))
    plt.close()
    with open(os.path.join(TESTS_DIR, 'resize_experiment.txt'), 'w', encoding='utf-8') as f:
        for i, s in enumerate(results['size']):
            f.write(f'Size: {s} | Time: {results["time"][i]:.2f}s | Mem: {results["memory"][i]:.2f} MB\n')

# Дообучение модели
if __name__ == "__main__":
    # Визуализация процесса обучения
    epochs = np.arange(1, n_epochs+1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Train loss')
    if val_losses:
        plt.plot(epochs[:len(val_losses)], val_losses, label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(epochs, train_accs, label='Train acc')
    if val_accs:
        plt.plot(epochs[:len(val_accs)], val_accs, label='Val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'finetune_training.png'))
    plt.close()
    with open(os.path.join(TESTS_DIR, 'finetune_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Final train loss: {train_losses[-1]:.4f}\n')
        if val_losses:
            f.write(f'Final val loss: {val_losses[-1]:.4f}\n')
        f.write(f'Final train acc: {train_accs[-1]:.4f}\n')
        if val_accs:
            f.write(f'Final val acc: {val_accs[-1]:.4f}\n')