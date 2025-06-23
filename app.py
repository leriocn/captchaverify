# -*- coding: utf-8 -*-
"""
- 【终极优化】针对GIF数学验证码，加入了“公式固定为3部分”的业务规则，通过筛选轮廓面积，只保留最大的3个部分进行识别，极大提高了抗干扰能力。
- 保持了基于霍夫变换的符号识别和所有其他成熟逻辑。
- 此版本是为您的验证码量身定制的、经过反复验证的、最稳定可靠的终极解决方案。
"""
import flask
from flask import request, jsonify
import requests
import io
import re
import ddddocr
import numpy as np
from PIL import Image, ImageSequence, ImageFilter, ImageEnhance
import logging
from collections import Counter
import random
import cv2
import os
import uuid

# --- 1. 初始化与配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
app = flask.Flask(__name__)
try:
    det = ddddocr.DdddOcr(det=False, ocr=True, show_ad=False)
    logging.info("ddddocr OCR model loaded successfully.")
except Exception as e:
    logging.error(f"Fatal error loading ddddocr model: {e}")
    det = None

DEBUG_DIR = 'debug_images'
os.makedirs(DEBUG_DIR, exist_ok=True)
logging.info(f"Base debug directory is '{DEBUG_DIR}/'. Subfolders will be created here on-demand.")

# --- 2. 核心图像处理与识别逻辑 ---

def get_most_common(items: list):
    if not items: return None
    return Counter(items).most_common(1)[0][0]

def apply_random_perturbations(image: Image.Image) -> Image.Image:
    processed_image = image.copy()
    enhancer = ImageEnhance.Contrast(processed_image)
    processed_image = enhancer.enhance(random.uniform(0.8, 1.5))
    enhancer = ImageEnhance.Brightness(processed_image)
    processed_image = enhancer.enhance(random.uniform(0.9, 1.2))
    if random.random() > 0.5:
        processed_image = processed_image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.5)))
    else:
        processed_image = processed_image.filter(ImageFilter.SHARPEN)
    return processed_image

def solve_gif_math_from_bytes(image_bytes: bytes, attempts: int, debug_path: str = None) -> tuple[str, str]:
    """【V29 终极版】针对“1位+符+1位”规则优化。"""
    logging.info("Starting GIF math recognition with V29 Optimized Strategy...")
    gif_image = Image.open(io.BytesIO(image_bytes))
    try:
        first_frame = gif_image.convert('L')
    except EOFError:
        raise ValueError("Invalid or corrupted GIF file provided.")
    
    # 步骤 1: 清理 & 强化
    composite_array = np.full(np.array(first_frame).shape, 255, dtype=np.uint8)
    for frame in ImageSequence.Iterator(gif_image):
        composite_array = np.minimum(composite_array, np.array(frame.convert('L')))
    clean_image = Image.fromarray(composite_array)
    if debug_path: clean_image.save(os.path.join(debug_path, 'math_1_clean_composite.png'))

    cv_clean_image = np.array(clean_image)
    kernel = np.ones((2, 2), np.uint8)
    enhanced_image_cv = cv2.dilate(cv_clean_image, kernel, iterations=1)
    if debug_path: Image.fromarray(enhanced_image_cv).save(os.path.join(debug_path, 'math_2_enhanced_image.png'))

    # 步骤 2: 准备蒙版
    _, segmentation_mask = cv2.threshold(enhanced_image_cv, 240, 255, cv2.THRESH_BINARY_INV)
    if debug_path: cv2.imwrite(os.path.join(debug_path, 'math_3_segmentation_mask.png'), segmentation_mask)

    # 步骤 3: 分割与识别
    contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # --- 关键优化：筛选出最大的3个轮廓（两个数字和一个运算符） ---
    if len(contours) < 3:
        raise ValueError(f"Segmentation failed: Found only {len(contours)} contours, expected at least 3.")
    
    # 按面积降序排序，并只取前3个
    top_three_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    logging.info(f"Found {len(contours)} raw contours, selected the 3 largest ones for processing.")
    
    # 获取这3个轮廓的边界框，并按x坐标排序
    bounding_boxes = [cv2.boundingRect(c) for c in top_three_contours]
    bounding_boxes.sort(key=lambda b: b[0])

    formula_parts = []
    image_for_cropping = Image.fromarray(cv2.bitwise_not(segmentation_mask))
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        char_result = ''
        
        # 基于形状和霍夫变换的符号判断
        if w > h * 1.8:
            char_result = '-'
            logging.info(f"Shape detection: Part #{i} is '-' (w:{w}, h:{h})")
        elif abs(w - h) < 15 and w > 15:
            char_img_for_lines = segmentation_mask[y:y+h, x:x+w]
            lines = cv2.HoughLinesP(char_img_for_lines, 1, np.pi / 180, threshold=8, minLineLength=h//3, maxLineGap=4)
            
            if lines is not None:
                horizontal, vertical = 0, 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if angle < 15 or angle > 165: horizontal += 1
                    elif 75 < angle < 105: vertical += 1
                
                if horizontal >= 1 and vertical >= 1:
                    char_result = '+'
                    logging.info(f"Hough detection: Part #{i} is '+' (h:{horizontal}, v:{vertical})")
                elif len(lines) >= 3:
                    char_result = '*'
                    logging.info(f"Hough detection: Part #{i} is '*' (lines:{len(lines)})")
        
        # 如果不是符号，则必然是数字
        if not char_result:
            padding = 2
            char_image = image_for_cropping.crop((x - padding, y - padding, x + w + padding, y + h + padding))
            if debug_path: char_image.save(os.path.join(debug_path, f'math_4_char_{i}.png'))
            img_byte_arr = io.BytesIO()
            char_image.save(img_byte_arr, format='PNG')
            char_result = det.classification(img_byte_arr.getvalue())
            logging.info(f"OCR detection: Part #{i} is '{char_result}' (w:{w}, h:{h})")
        
        formula_parts.append(char_result)

    # 步骤 4: 组合并计算
    source_formula = "".join(formula_parts)
    match = re.fullmatch(r'(\d)([\+\-\*])(\d)', source_formula) # 正则表达式也简化为只匹配一位数

    if not match:
        raise ValueError(f"Could not form a valid 'digit-op-digit' expression from parts: '{source_formula}'")

    num1, op, num2 = int(match.group(1)), match.group(2), int(match.group(3))
    result = 0
    if op == '+': result = num1 + num2
    elif op == '-': result = num1 - num2
    elif op == '*': result = num1 * num2
    
    return str(result), source_formula

def solve_alphanumeric_from_bytes(image_bytes: bytes, attempts: int, debug_path: str = None) -> tuple[str, str]:
    """【V29 终极版】严格按“清理 -> 膨胀 -> 自适应二值化 -> 识别”流程。"""
    logging.info("Starting alphanumeric recognition with V29 Final Strategy...")
    # 步骤 1: 清理
    clean_image = Image.open(io.BytesIO(image_bytes)).convert("L")
    if debug_path: clean_image.save(os.path.join(debug_path, 'alphanum_1_clean_image.png'))

    # 步骤 2: 膨胀强化
    cv_clean_image = np.array(clean_image)
    kernel = np.ones((2, 2), np.uint8)
    enhanced_image_cv = cv2.dilate(cv_clean_image, kernel, iterations=1)
    if debug_path: Image.fromarray(enhanced_image_cv).save(os.path.join(debug_path, 'alphanum_2_enhanced_image.png'))

    # 步骤 3: 智能二值化 (Adaptive Binarization)
    adaptive_binary_cv = cv2.adaptiveThreshold(enhanced_image_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
    image_for_recognition = Image.fromarray(adaptive_binary_cv)
    if debug_path:
        image_for_recognition.save(os.path.join(debug_path, 'alphanum_3_binarized_for_recognition.png'))
        logging.info("Step 3: Saved adaptive binarized image, ready for recognition.")

    # 步骤 4: 基于“二值化后”的图片进行识别
    valid_candidates = []
    for i in range(attempts):
        perturbed_image = apply_random_perturbations(image_for_recognition)
        img_byte_arr = io.BytesIO()
        perturbed_image.save(img_byte_arr, format='PNG')
        text = det.classification(img_byte_arr.getvalue())
        logging.info(f"Alphanum OCR output (attempt {i+1}): '{text}'")
        if text and len(text) == 4 and text.isalnum():
            valid_candidates.append(text)
        
    final_result = get_most_common(valid_candidates)
    if not final_result:
        raise ValueError("Failed to get a consistent 4-digit result after all attempts.")
        
    return final_result, final_result

# --- 3. 辅助函数与 API 路由 (不变) ---
def download_image(url: str, cookie: str = None) -> bytes:
    logging.info(f"Downloading image from: {url}")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    if cookie:
        headers['Cookie'] = cookie
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to download image from URL. Reason: {e}")

def api_handler_factory(handler_function):
    def endpoint():
        logging.info(f"Request received for endpoint: {request.endpoint}")
        if not det:
            return jsonify({"result": None, "source": None, "success": False, "msg": "OCR service is not available."}), 503
        if not request.is_json:
            return jsonify({"result": None, "source": None, "success": False, "msg": "Request content type must be application/json."}), 400
        
        data = request.get_json()
        url = data.get('url')
        attempts = data.get('attempts', 10)
        cookie = data.get('cookie')
        is_debug_mode = data.get('debug', False)

        logging.info(f"Payload received: url={url}, attempts={attempts}, debug={is_debug_mode}, cookie_present={'YES' if cookie else 'NO'}")

        if not url:
            return jsonify({"result": None, "source": None, "success": False, "msg": "Missing 'url' parameter."}), 400
        
        debug_path = None
        if is_debug_mode:
            request_id = str(uuid.uuid4())
            debug_path = os.path.join(DEBUG_DIR, request_id)
            os.makedirs(debug_path, exist_ok=True)
            logging.info(f"Debug mode ON. Saving images for this request to: {debug_path}")
        
        try:
            image_bytes = download_image(url, cookie=cookie)
            result, source = handler_function(image_bytes, attempts, debug_path=debug_path)
            response_data = {"result": result, "source": source, "success": True, "msg": "Recognition successful."}
            return jsonify(response_data)
        except (ConnectionError, ValueError) as e:
            logging.error(f"Error for URL '{url}': {e}")
            return jsonify({"result": None, "source": None, "success": False, "msg": str(e)}), 400
        except Exception as e:
            logging.error(f"Unexpected internal error for URL '{url}': {e}", exc_info=True)
            return jsonify({"result": None, "source": None, "success": False, "msg": "An internal server error occurred."}), 500
            
    endpoint.__name__ = f"{handler_function.__name__}_endpoint"
    return endpoint

app.route('/recognize_gif_math', methods=['POST'])(api_handler_factory(solve_gif_math_from_bytes))
app.route('/recognize_alphanumeric', methods=['POST'])(api_handler_factory(solve_alphanumeric_from_bytes))

@app.route('/', methods=['GET'])
def index():
    return """
    <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Captcha Recognition API</title>
    <style>body{font-family:sans-serif;line-height:1.6;padding:2em;}code{background-color:#f4f4f4;padding:2px 6px;border-radius:4px;}pre{background-color:#f4f4f4;padding:1em;border-radius:4px;white-space:pre-wrap;}</style>
    </head><body><h1>Captcha Recognition API (Final V29)</h1><p>An API to recognize two types of captchas. It is concurrency-safe and supports on-demand debugging.</p>
    <h2>Endpoints:</h2><ul><li><code>POST /recognize_gif_math</code>: For GIF-based math captchas (+, -, *), optimized for 'digit-op-digit' format.</li>
    <li><code>POST /recognize_alphanumeric</code>: For standard 4-digit alphanumeric captchas, using a full enhancement pipeline.</li></ul>
    <h2>Request Body (JSON):</h2><pre>{
  "url": "http://example.com/captcha.gif",
  "cookie": "session_id=abc123xyz; ...",
  "debug": true
}</pre></body></html>
    """

# --- 4. 启动应用 ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=False)
