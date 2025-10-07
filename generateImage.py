# augment_ma.py
from PIL import Image, ImageEnhance
import numpy as np
import os, random

# ----------------------------
# CONFIG
# ----------------------------
# Put your cropped source images (tight crop around the glyph) here:
# They must be images with the glyph visible (black ink on white or transparent OK).
src_images = {
    "b_final": r"source/final_b.png",
    "d_final": r"source/final_d.png",
    "f_final": r"source/final_f.png",
    "ga_final": r"source/final_ga.png",
    "ge_final": r"source/final_ge.png",
    "h_final": r"source/final_h.png",
    "k_final":   r"source/final_k.png",
    "l_final":   r"source/final_l.png",
    "p_final":   r"source/final_p.png",
    "r_final":   r"source/final_r.png",
    "s_final":   r"source/final_s.png",
    "ts_final":   r"source/final_ts.png",
    "v_final":   r"source/final_v.png",
    "z_final":   r"source/final_z.png",
}

output_dir = "dataset/train"
backgrounds_dir = "backgrounds"   # optional: add some paper textures here; leave if you don't have any
img_size = 64                     # final image size (square)
images_per_char = 50             # how many augmented images to generate per form

# ----------------------------
# UTILITIES
# ----------------------------
def load_and_mask_glyph(path, target_size):
    """Load source image, create a tight RGBA glyph with alpha mask, and scale down if too big."""
    img = Image.open(path).convert("RGBA")
    # create mask from luminance - treat anything dark as ink
    gray = img.convert("L")
    arr = np.array(gray)
    mask = (arr < 250).astype(np.uint8) * 255            # threshold; adjust 250 if needed
    mask_img = Image.fromarray(mask, mode="L")
    bbox = mask_img.getbbox()
    if not bbox:
        raise ValueError(f"No ink detected in {path} (check threshold/background).")
    glyph = img.crop(bbox)            # tight crop
    # ensure RGBA
    if glyph.mode != "RGBA":
        glyph = glyph.convert("RGBA")
    # optionally scale glyph so it fits nicely in target_size (max ~70% of img)
    max_dim = int(target_size * 0.7)
    w,h = glyph.size
    scale = min(max_dim/w, max_dim/h, 1.0)
    new_w, new_h = max(1, int(w*scale)), max(1, int(h*scale))
    glyph = glyph.resize((new_w, new_h), Image.LANCZOS)
    return glyph

def get_random_background(size, backgrounds_dir=None):
    """Return an RGB background image of given size. Prefer real backgrounds if provided, else generate noise/paper."""
    if backgrounds_dir and os.path.isdir(backgrounds_dir):
        files = [f for f in os.listdir(backgrounds_dir) if os.path.isfile(os.path.join(backgrounds_dir,f))]
        if files:
            choice = random.choice(files)
            bg = Image.open(os.path.join(backgrounds_dir, choice)).convert("RGB")
            return bg.resize((size, size), Image.LANCZOS)

    # create synthetic paper/noise background
    arr = np.random.normal(loc=240, scale=6, size=(size, size)).clip(0,255).astype(np.uint8)
    # add low-frequency variation for paper effect
    lowfreq = np.clip(np.random.normal(loc=0, scale=8, size=(size//8, size//8)), -30, 30)
    lowfreq = np.kron(lowfreq, np.ones((8,8)))[:size,:size]
    arr2 = np.clip(arr + lowfreq, 0, 255).astype(np.uint8)
    bg = Image.fromarray(arr2, mode="L").convert("RGB")
    return bg

def paste_glyph_on_background(bg, glyph):
    """Paste glyph (RGBA) onto background at a random position and return RGB image."""
    out = bg.copy()
    gw, gh = glyph.size
    max_x = max(0, bg.size[0] - gw)
    max_y = max(0, bg.size[1] - gh)
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    out.paste(glyph, (x,y), glyph)   # use glyph alpha as mask
    return out

def random_augment_and_save(glyph, out_folder, count, img_size, backgrounds_dir):
    os.makedirs(out_folder, exist_ok=True)
    for i in range(count):
        # background
        bg = get_random_background(img_size, backgrounds_dir)

        # random scale jitter 0.8-1.15
        scale = random.uniform(0.80, 1.15)
        gw, gh = glyph.size
        new_g = glyph.resize((max(1,int(gw*scale)), max(1,int(gh*scale))), Image.LANCZOS)

        # random rotate
        angle = random.uniform(-12, 12)
        new_g = new_g.rotate(angle, expand=True)

        # paste on BG
        out = paste_glyph_on_background(bg, new_g)

        # brightness & contrast
        if random.random() < 0.9:
            out = ImageEnhance.Brightness(out).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.7:
            out = ImageEnhance.Contrast(out).enhance(random.uniform(0.85, 1.25))

        # small gaussian noise
        if random.random() < 0.6:
            ar = np.array(out).astype(np.int16)
            noise = np.random.normal(0, 6, ar.shape).astype(np.int16)
            ar = np.clip(ar + noise, 0, 255).astype(np.uint8)
            out = Image.fromarray(ar)

        # convert to grayscale final (common for OCR), or keep RGB if you prefer
        out = out.convert("L")

        # optionally add slight blur occasionally (paper capture)
        if random.random() < 0.15:
            out = out.filter(Image.Filter.GaussianBlur(radius=random.uniform(0.3, 1.2))) \
                  if hasattr(Image, 'Filter') else out

        filename = f"{i:04d}.png"
        out.save(os.path.join(out_folder, filename))

# ----------------------------
# MAIN - build dataset
# ----------------------------
def main():
    # validate sources
    for form, path in src_images.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Source image not found: {path}")

    for form, path in src_images.items():
        print(f"> Processing {form} from {path} ...")
        glyph = load_and_mask_glyph(path, img_size)
        out_folder = os.path.join(output_dir, form)
        random_augment_and_save(glyph, out_folder, images_per_char, img_size, backgrounds_dir)
        print(f"  -> saved {images_per_char} images to {out_folder}")

    print("Done. Dataset ready in", output_dir)

if __name__ == "__main__":
    main()
