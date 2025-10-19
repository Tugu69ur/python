# augment_ma_yolo.py
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import os, random

# -------------------
# CONFIG
# -------------------
src_images = {
    "namor": r"source/namor.png",
    "mongol": r"source/mongol.png",
}

output_dir = "synth_dataset"
backgrounds_dir = "backgrounds"  # optional: paper textures
img_size = 128                    # final square size
images_per_char = 50              # how many augmented images per source

# map class names to numeric class ids
class_map = {name: i for i, name in enumerate(src_images.keys())}

# -------------------
# UTILITIES
# -------------------
def load_and_mask_glyph(path, target_size):
    """Load source image, tight crop, scale to fit target_size."""
    img = Image.open(path).convert("RGBA")
    gray = img.convert("L")
    mask = (np.array(gray) < 250).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, mode="L")
    bbox = mask_img.getbbox()
    if not bbox:
        raise ValueError(f"No ink detected in {path}")
    glyph = img.crop(bbox)
    max_dim = int(target_size * 0.7)
    w, h = glyph.size
    scale = min(max_dim / w, max_dim / h, 1.0)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return glyph.resize((new_w, new_h), Image.LANCZOS)

def get_random_background(size, backgrounds_dir=None):
    """Return RGB background image of given size."""
    if backgrounds_dir and os.path.isdir(backgrounds_dir):
        files = [f for f in os.listdir(backgrounds_dir) if os.path.isfile(os.path.join(backgrounds_dir,f))]
        if files:
            choice = random.choice(files)
            bg = Image.open(os.path.join(backgrounds_dir, choice)).convert("RGB")
            return bg.resize((size, size), Image.LANCZOS)
    # synthetic paper
    arr = np.random.normal(240, 6, (size, size)).clip(0,255).astype(np.uint8)
    lowfreq = np.clip(np.random.normal(0, 8, (size//8, size//8)), -30, 30)
    lowfreq = np.kron(lowfreq, np.ones((8,8)))[:size,:size]
    bg = Image.fromarray(np.clip(arr + lowfreq, 0, 255).astype(np.uint8), "L").convert("RGB")
    return bg

def paste_glyph_on_background(bg, glyph):
    """Paste glyph onto background at random position."""
    out = bg.copy()
    gw, gh = glyph.size
    max_x = max(0, bg.size[0] - gw)
    max_y = max(0, bg.size[1] - gh)
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    out.paste(glyph, (x, y), glyph)
    # return image and bounding box
    return out, (x, y, x + gw, y + gh)

def random_augment_and_save(glyph, out_folder, count, img_size, backgrounds_dir, class_id):
    os.makedirs(os.path.join(out_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_folder, "labels"), exist_ok=True)
    for i in range(count):
        bg = get_random_background(img_size, backgrounds_dir)

        # random scale jitter
        scale = random.uniform(0.85, 1.15)
        gw, gh = glyph.size
        new_g = glyph.resize((max(1, int(gw*scale)), max(1, int(gh*scale))), Image.LANCZOS)

        # random rotate
        angle = random.uniform(-12, 12)
        new_g = new_g.rotate(angle, expand=True)

        # paste glyph
        out_img, bbox = paste_glyph_on_background(bg, new_g)

        # brightness & contrast
        if random.random() < 0.9:
            out_img = ImageEnhance.Brightness(out_img).enhance(random.uniform(0.85,1.15))
        if random.random() < 0.7:
            out_img = ImageEnhance.Contrast(out_img).enhance(random.uniform(0.85,1.25))

        # gaussian noise
        if random.random() < 0.5:
            arr = np.array(out_img).astype(np.int16)
            noise = np.random.normal(0,6,arr.shape).astype(np.int16)
            arr = np.clip(arr + noise,0,255).astype(np.uint8)
            out_img = Image.fromarray(arr)

        # optional blur
        if random.random() < 0.15:
            out_img = out_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3,1.2)))

        # convert to grayscale (optional)
        out_img = out_img.convert("L")

        # save image
        img_name = f"{i:04d}.png"
        img_path = os.path.join(out_folder, "images", img_name)
        out_img.save(img_path)

        # save YOLO label
        x0, y0, x1, y1 = bbox
        cx = (x0 + x1) / 2 / img_size
        cy = (y0 + y1) / 2 / img_size
        w = (x1 - x0) / img_size
        h = (y1 - y0) / img_size
        label_path = os.path.join(out_folder, "labels", f"{i:04d}.txt")
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

# -------------------
# MAIN
# -------------------
def main():
    for form, path in src_images.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Source image not found: {path}")
    for form, path in src_images.items():
        print(f"> Processing {form} ...")
        glyph = load_and_mask_glyph(path, img_size)
        out_folder = os.path.join(output_dir, form)
        class_id = class_map[form]
        random_augment_and_save(glyph, out_folder, images_per_char, img_size, backgrounds_dir, class_id)
        print(f"  -> saved {images_per_char} images and labels to {out_folder}")

    print("✅ Done. Dataset ready in", output_dir)

if __name__ == "__main__":
    main()
