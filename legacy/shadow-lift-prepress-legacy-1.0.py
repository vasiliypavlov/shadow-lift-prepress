import os
import argparse
import sys
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

def parse_size(size_str):
    try:
        if 'x' in size_str.lower():
            w, h = size_str.lower().split('x')
        elif ',' in size_str:
            w, h = size_str.split(',')
        else:
            raise ValueError
        return (int(w.strip()), int(h.strip()))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Неверный формат размера: '{size_str}'. Используйте '900x1500' или '900,1500'")

def detect_and_create_protection_mask(arr, luminance, protection_zone, height):
    h, w = luminance.shape
    y_indices = np.arange(height, dtype=np.float32)[:, np.newaxis]
    
    if protection_zone != 'auto':
        print(f"  -> [MANUAL] Принудительная защита: {protection_zone.upper()}")
        if protection_zone == 'none':
            return np.ones((height, 1), dtype=np.float32)
        elif protection_zone == 'bottom':
            return np.clip(((height * 0.60) - y_indices) / (height * 0.25), 0, 1)
        else: # 'top'
            return np.clip((y_indices - (height * 0.40)) / (height * 0.25), 0, 1)

    top_zone = luminance[:int(h * 0.20), :]
    bottom_zone = luminance[-int(h * 0.20):, :]
    center_zone = luminance[int(h * 0.30):int(h * 0.70), :]
    
    top_mean = np.mean(top_zone)
    bottom_mean = np.mean(bottom_zone)
    center_mean = np.mean(center_zone)
    
    diff = bottom_mean - top_mean
    is_bright_center = (center_mean > top_mean + 30) and (center_mean > bottom_mean + 30)
    
    if is_bright_center:
        print(f"  -> [AUTO] Обнаружен яркий центральный объект (Центр: {center_mean:.1f}). Анализ фона...")

    if diff > 15:
        print(f"  -> [AUTO] Низ светлее верха (Верх: {top_mean:.1f}, Низ: {bottom_mean:.1f}, Разница: +{diff:.1f}). Защита: BOTTOM")
        center_line = height * 0.60
        transition_zone = height * 0.20
        return np.clip((center_line - y_indices) / transition_zone, 0, 1)
        
    if diff < -15:
        print(f"  -> [AUTO] Верх светлее низа (Верх: {top_mean:.1f}, Низ: {bottom_mean:.1f}, Разница: {diff:.1f}). Защита: TOP")
        center_line = height * 0.40
        transition_zone = height * 0.20
        return np.clip((y_indices - center_line) / transition_zone, 0, 1)
        
    top_rgb = arr[:int(h * 0.20), :, :]
    r_mean = np.mean(top_rgb[:,:,0])
    b_mean = np.mean(top_rgb[:,:,2])
    
    if top_mean < 65 and b_mean > r_mean * 1.15:
        print(f"  -> [AUTO] Обнаружено ночное небо (Верх: {top_mean:.1f}, Синева: {b_mean/r_mean:.2f}x). Защита: TOP")
        center_line = height * (0.35 if is_bright_center else 0.40)
        transition_zone = height * (0.30 if is_bright_center else 0.25)
        return np.clip((y_indices - center_line) / transition_zone, 0, 1)

    print(f"  -> [AUTO] Равномерное освещение (Верх: {top_mean:.1f}, Низ: {bottom_mean:.1f}). Защита: NONE")
    return np.ones((height, 1), dtype=np.float32)

def process_single_image(input_path, output_path, target_min_black, target_size, protection_zone='auto', gamma=0.85):
    try:
        img = Image.open(input_path).convert('RGB')
        
        if target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        arr = np.array(img, dtype=np.float32)
        height, width, _ = arr.shape
        
        luminance = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
        
        # Немного снизили порог отсечения (с 25 до 15), чтобы захватывать только реально темные пиксели
        shadow_threshold = max(np.percentile(luminance, 5), 15.0)
        raw_shadow_mask = np.clip((shadow_threshold - luminance) / shadow_threshold, 0, 1)
        
        protection_mask = detect_and_create_protection_mask(arr, luminance, protection_zone, height)
        
        smooth_shadows = gaussian_filter(raw_shadow_mask, sigma=3.0)
        final_mask = smooth_shadows * protection_mask
        final_mask = np.expand_dims(final_mask, axis=2)
        
        # Вытягивание теней с учетом новой гаммы
        lifted_arr = (arr / 255.0) ** gamma * (255.0 - target_min_black) + target_min_black
        arr = arr * (1 - final_mask) + lifted_arr * final_mask
        
        # Локальный контраст (снизили с 0.30 до 0.20, чтобы не было "пыли")
        blurred = gaussian_filter(arr, sigma=2.0, truncate=2.0)
        detail = arr - blurred
        arr = arr + (detail * 0.20 * final_mask) 
        
        # Компенсация насыщенности (снизили с 0.1 до 0.05)
        mean_color = np.mean(arr, axis=2, keepdims=True)
        arr = arr + (arr - mean_color) * 0.05 * final_mask
        
        current_min = np.min(arr)
        if current_min < target_min_black:
            scale = (255.0 - target_min_black) / 255.0
            arr = arr * scale + target_min_black
            
        arr = np.clip(arr, 0, 255)
        
        processed_img = Image.fromarray(arr.astype(np.uint8))
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
        processed_img.save(output_path, format='PNG', compress_level=6)
        return True, None
        
    except Exception as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(
        description="Shadow-Lift Prepres: Умное вытягивание теней для TGC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-i", "--input", required=True, dest="input_path", help="Путь к файлу ИЛИ к папке")
    parser.add_argument("-o", "--output", required=True, dest="output_path", help="Путь к папке ИЛИ к файлу")
    parser.add_argument("-b", "--min-black", type=int, default=14, dest="target_min_black", help="Точка черного (14-16 для TGC)")
    parser.add_argument("-s", "--size", type=parse_size, default="900x1500", dest="target_size", help="Размер (ШxВ)")
    parser.add_argument("-p", "--protection", choices=['auto', 'top', 'bottom', 'none'], default='auto', dest="protection_zone", help="Зона пространственной защиты.")
    parser.add_argument("-g", "--gamma", type=float, default=0.85, dest="gamma", 
                        help="Сила вытягивания теней. 0.85 (по умолч.), 0.95 (мягко, сохраняет глубину), 0.70 (агрессивно).")
    
    args = parser.parse_args()
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

    if os.path.isfile(args.input_path):
        if not args.input_path.lower().endswith(valid_extensions):
            print(f"Ошибка: Файл {args.input_path} не является изображением.")
            sys.exit(1)
            
        if args.output_path.lower().endswith('.png'):
            out_file = args.output_path
        else:
            base_name = os.path.splitext(os.path.basename(args.input_path))[0]
            out_file = os.path.join(args.output_path, base_name + '.png')
            
        print(f"Обработка файла: {args.input_path}")
        success, err = process_single_image(args.input_path, out_file, args.target_min_black, args.target_size, args.protection_zone, args.gamma)
        if success:
            print(f"[OK] Сохранено: {out_file}")
        else:
            print(f"[ОШИБКА] {err}")

    elif os.path.isdir(args.input_path):
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path, exist_ok=True)
            
        files = [f for f in os.listdir(args.input_path) if f.lower().endswith(valid_extensions)]
        
        if not files:
            print(f"В папке {args.input_path} не найдено изображений.")
            sys.exit(0)

        print(f"Найдено изображений: {len(files)}")
        print(f"Параметры: Размер={args.target_size}, Мин. черный={args.target_min_black}, Гамма={args.gamma}, Защита={args.protection_zone}")
        print("-" * 50)
        
        success_count = 0
        failed_files = []
        
        for filename in files:
            in_file = os.path.join(args.input_path, filename)
            base_name = os.path.splitext(filename)[0]
            out_file = os.path.join(args.output_path, base_name + '.png')
            
            print(f"Обработка: {filename}")
            success, err = process_single_image(in_file, out_file, args.target_min_black, args.target_size, args.protection_zone, args.gamma)
            
            if success:
                print(f"[OK] {filename} -> {out_file}\n")
                success_count += 1
            else:
                print(f"[ОШИБКА] {filename}: {err}\n")
                failed_files.append((filename, err))
            
        print("-" * 50)
        print(f"ИТОГО: Успешно обработано {success_count} из {len(files)}.")
        
        if failed_files:
            print("\nВНИМАНИЕ! Следующие файлы не были обработаны:")
            for fname, error in failed_files:
                print(f" - {fname}: {error}")
        else:
            print("Все файлы обработаны без ошибок. Отлично!")
        
    else:
        print(f"Ошибка: Путь {args.input_path} не найден.")
        sys.exit(1)

if __name__ == "__main__":
    main()