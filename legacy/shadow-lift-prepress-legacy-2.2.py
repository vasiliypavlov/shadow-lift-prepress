import os
import argparse
import sys
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

"""
Shadow-Lift Prepress (Legacy v2.2)
==================================
Гамма-версия с усиленным локальным контрастом и жестким клэмпом точки черного.
Убрана "серая вуаль" от глобального remap.
"""

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

def auto_detect_gamma(luminance):
    """
    Анализирует яркость сцены и автоматически подбирает гамму.
    """
    mean_lum = np.mean(luminance)
    p10_lum = np.percentile(luminance, 10) 

    if mean_lum < 45:
        scene = "Ночная / Очень темная"
        gamma = 0.65
    elif mean_lum < 85:
        scene = "Вечерняя / Сумрачная / Мрачная (Индустриальная)"
        gamma = 0.70
    elif mean_lum < 125:
        scene = "Пасмурная / Нейтральная"
        gamma = 0.82
    else:
        scene = "Дневная / Солнечная / Светлая"
        gamma = 0.92
        
    print(f"  -> [AUTO-GAMMA] Сцена: {scene} (Ср.яркость: {mean_lum:.1f}, Глубина теней: {p10_lum:.1f}). Установлена гамма: {gamma}")
    return gamma

def detect_and_create_protection_mask(arr, luminance, protection_zone, height, width):
    """Возвращает 2D-маску (height, width) для удобства будущих правок."""
    h, w = luminance.shape
    y_indices = np.arange(height, dtype=np.float32)[:, np.newaxis]
    x_indices = np.arange(width, dtype=np.float32)[np.newaxis, :]

    def make_v_mask(center, trans, invert=False):
        mask = np.clip((y_indices - center) / trans, 0, 1)
        return 1.0 - mask if invert else mask

    def make_h_mask(center, trans, invert=False):
        mask = np.clip((x_indices - center) / trans, 0, 1)
        return 1.0 - mask if invert else mask

    if protection_zone == 'none':
        return np.ones((height, width), dtype=np.float32)
    
    if protection_zone == 'top':
        return np.tile(make_v_mask(height * 0.40, height * 0.20), (1, width))
    if protection_zone == 'bottom':
        return np.tile(make_v_mask(height * 0.60, height * 0.25, invert=True), (1, width))
    if protection_zone == 'left':
        return np.tile(make_h_mask(width * 0.40, width * 0.20), (height, 1))
    if protection_zone == 'right':
        return np.tile(make_h_mask(width * 0.60, width * 0.20, invert=True), (height, 1))
    if protection_zone == 'sides':
        left_mask = make_h_mask(width * 0.35, width * 0.15)
        right_mask = make_h_mask(width * 0.65, width * 0.15, invert=True)
        sides_1d = np.maximum(left_mask, right_mask)
        return np.tile(sides_1d, (height, 1))
    if protection_zone == 'center':
        left_mask = make_h_mask(width * 0.35, width * 0.15)
        right_mask = make_h_mask(width * 0.65, width * 0.15, invert=True)
        sides_1d = np.maximum(left_mask, right_mask)
        return np.tile(1.0 - sides_1d, (height, 1))

    # AUTO logic
    if protection_zone == 'auto':
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
            print(f"  -> [AUTO] Низ светлее верха (Разница: +{diff:.1f}). Защита: BOTTOM")
            center_line = height * 0.60
            transition_zone = height * 0.20
            base_mask = np.clip((center_line - y_indices) / transition_zone, 0, 1)
            return np.tile(base_mask, (1, width))
            
        if diff < -15:
            print(f"  -> [AUTO] Верх светлее низа (Разница: {diff:.1f}). Защита: TOP")
            center_line = height * 0.40
            transition_zone = height * 0.20
            base_mask = np.clip((y_indices - center_line) / transition_zone, 0, 1)
            return np.tile(base_mask, (1, width))
            
        top_rgb = arr[:int(h * 0.20), :, :]
        r_mean = np.mean(top_rgb[:,:,0])
        b_mean = np.mean(top_rgb[:,:,2])
        
        if top_mean < 65 and b_mean > r_mean * 1.15:
            print(f"  -> [AUTO] Обнаружено ночное небо (Синева: {b_mean/r_mean:.2f}x). Защита: TOP")
            center_line = height * (0.35 if is_bright_center else 0.40)
            transition_zone = height * (0.30 if is_bright_center else 0.25)
            base_mask = np.clip((y_indices - center_line) / transition_zone, 0, 1)
            return np.tile(base_mask, (1, width))

        print(f"  -> [AUTO] Равномерное освещение. Защита: NONE")
        return np.ones((height, width), dtype=np.float32)

    return np.ones((height, width), dtype=np.float32)

def process_single_image(input_path, output_path, target_min_black, target_size, protection_zone='auto', gamma='auto'):
    try:
        img = Image.open(input_path).convert('RGB')
        
        if target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        arr = np.array(img, dtype=np.float32)
        height, width, _ = arr.shape
        
        luminance = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
        
        # 1. АВТОМАТИЧЕСКИЙ ПОДБОР ГАММЫ
        if str(gamma).lower() == 'auto':
            actual_gamma = auto_detect_gamma(luminance)
        else:
            try:
                actual_gamma = float(gamma)
                print(f"  -> [MANUAL-GAMMA] Использована заданная гамма: {actual_gamma}")
            except ValueError:
                print(f"  -> [ОШИБКА] Неверное значение гаммы '{gamma}'. Используется 0.85.")
                actual_gamma = 0.85

        # 2. Адаптивный порог теней
        shadow_threshold = max(np.percentile(luminance, 5), 15.0)
        raw_shadow_mask = np.clip((shadow_threshold - luminance) / shadow_threshold, 0, 1)
        
        # 3. Пространственная защита (2D)
        protection_mask = detect_and_create_protection_mask(arr, luminance, protection_zone, height, width)
        
        smooth_shadows = gaussian_filter(raw_shadow_mask, sigma=3.0)
        final_mask = smooth_shadows * protection_mask
        final_mask = np.expand_dims(final_mask, axis=2)
        
        # 4. Вытягивание теней (Глобальная гамма)
        lifted_arr = (arr / 255.0) ** actual_gamma * (255.0 - target_min_black) + target_min_black
        arr = arr * (1 - final_mask) + lifted_arr * final_mask
        
        # 5. УСИЛЕННЫЙ ЛОКАЛЬНЫЙ КОНТРАСТ (0.30)
        blurred = gaussian_filter(arr, sigma=2.0, truncate=2.0)
        detail = arr - blurred
        arr = arr + (detail * 0.30 * final_mask) 
        
        # 6. Компенсация насыщенности
        mean_color = np.mean(arr, axis=2, keepdims=True)
        arr = arr + (arr - mean_color) * 0.05 * final_mask
        
        # 7. ГАРАНТИРОВАННАЯ ТОЧКА ЧЕРНОГО (Жесткий клэмп вместо Remap)
        arr = np.maximum(arr, target_min_black)
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
        description="Shadow-Lift Prepres (legacy-v2.2): Гамма-версия с жестким клэмпом черного.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-i", "--input", required=True, dest="input_path", help="Путь к файлу ИЛИ к папке")
    parser.add_argument("-o", "--output", required=True, dest="output_path", help="Путь к папке ИЛИ к файлу")
    parser.add_argument("-b", "--min-black", type=int, default=14, dest="target_min_black", help="Точка черного (14-16 для TGC)")
    parser.add_argument("-s", "--size", type=parse_size, default="900x1500", dest="target_size", help="Размер (ШxВ)")
    parser.add_argument("-p", "--protection", choices=['auto', 'top', 'bottom', 'left', 'right', 'sides', 'center', 'none'], 
                        default='auto', dest="protection_zone", help="Зона пространственной защиты.")
    
    parser.add_argument("-g", "--gamma", type=str, default="auto", dest="gamma", 
                        help="Сила вытягивания теней. 'auto' (по умолч.) - авто-определение сцены. Или число от 0.6 до 0.95.")
    
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