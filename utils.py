# utils.py
import cv2
import numpy as np
import os
import random
from datetime import datetime


# ============== 1. LOAD & INFO ==============
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load: {path}")
    return img

def get_image_info(img):
    if len(img.shape) == 3:
        h, w, c = img.shape
    else:
        h, w = img.shape
        c = 1
    return w, h, c


# ============== 2. RESIZE ==============
def resize_image(img, size=(256, 256)):
    return cv2.resize(img, size)


# ============== 3. CROP ==============
def center_crop(img, size=(200, 200)):
    h, w = img.shape[:2]
    ch, cw = size
    sy = (h - ch) // 2
    sx = (w - cw) // 2
    return img[sy:sy+ch, sx:sx+cw]

def random_crop(img, size=(200, 200)):
    h, w = img.shape[:2]
    ch, cw = size
    sy = random.randint(0, max(0, h - ch))
    sx = random.randint(0, max(0, w - cw))
    return img[sy:sy+ch, sx:sx+cw]


# ============== 4. ROTATE ==============
def rotate_image(img, angle):
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h))


# ============== 5. FLIP ==============
def flip_image(img, mode='horizontal'):
    modes = {'horizontal': 1, 'vertical': 0, 'both': -1}
    return cv2.flip(img, modes[mode])


# ============== 6. COLOR CONVERSION ==============
def convert_color(img, target='gray'):
    conversions = {
        'rgb':  cv2.COLOR_BGR2RGB,
        'gray': cv2.COLOR_BGR2GRAY,
        'hsv':  cv2.COLOR_BGR2HSV,
        'lab':  cv2.COLOR_BGR2LAB,
    }
    return cv2.cvtColor(img, conversions[target])


# ============== 7. BLUR ==============
def apply_blur(img, kind='gaussian', ksize=5):
    if kind == 'gaussian':
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    elif kind == 'median':
        return cv2.medianBlur(img, ksize)
    elif kind == 'average':
        return cv2.blur(img, (ksize, ksize))
    elif kind == 'bilateral':
        return cv2.bilateralFilter(img, ksize, 75, 75)


# ============== 8. EDGE DETECTION ==============
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

def canny_edge(img, t1=100, t2=200):
    return cv2.Canny(to_gray(img), t1, t2)

def sobel_edge(img, ksize=3):
    g = to_gray(img)
    sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=ksize)
    sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=ksize)
    return cv2.magnitude(sx, sy).astype(np.uint8)

def laplacian_edge(img, ksize=3):
    return cv2.Laplacian(to_gray(img), cv2.CV_64F, ksize=ksize).astype(np.uint8)


# ============== 9. THRESHOLDING ==============
def threshold_image(img, method='binary', thresh=127, maxval=255):
    g = to_gray(img)
    if method == 'binary':
        _, out = cv2.threshold(g, thresh, maxval, cv2.THRESH_BINARY)
    elif method == 'adaptive':
        out = cv2.adaptiveThreshold(g, maxval, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    elif method == 'otsu':
        _, out = cv2.threshold(g, 0, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


# ============== 10. HISTOGRAM ==============
def compute_histogram(img):
    if len(img.shape) == 3:
        return {c: cv2.calcHist([img], [i], None, [256], [0, 256])
                for i, c in enumerate(('b', 'g', 'r'))}
    return cv2.calcHist([img], [0], None, [256], [0, 256])


# ============== 11. DRAW SHAPES ==============
def draw_shapes(img):
    out = img.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (50, 50), (200, 200), (0, 255, 0), 3)
    cv2.circle(out, (w//2, h//2), 80, (0, 0, 255), -1)
    cv2.line(out, (0, 0), (w, h), (255, 0, 0), 3)
    cv2.putText(out, 'OpenCV', (50, h-50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return out


# ============== 12. SAVE ==============
def save_image(img, output_dir='output', prefix='result'):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(output_dir, f"{prefix}_{ts}.jpg")
    cv2.imwrite(path, img)
    return path


# ============== BONUS ==============
def adjust_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v.astype(np.int16) + value, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

def adjust_contrast(img, alpha=1.5, beta=0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def gamma_correction(img, gamma=1.5):
    inv = 1.0 / gamma
    table = np.array([((i/255.0)**inv)*255 for i in range(256)]).astype('uint8')
    return cv2.LUT(img, table)

def sharpen_image(img):
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)

def cartoon_effect(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.medianBlur(g, 5)
    edges = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    return cv2.bitwise_and(color, color, mask=edges)

def pencil_sketch(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(g)
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    return cv2.divide(g, cv2.bitwise_not(blur), scale=256.0)

def add_watermark(img, text='WATERMARK'):
    out = img.copy()
    h, w = out.shape[:2]
    cv2.putText(out, text, (w-300, h-30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return out