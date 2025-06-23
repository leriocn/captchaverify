# Captcha Recognition API (验证码识别API)

这是一个基于 Python Flask 和 OpenCV 的高性能Web API，旨在识别两种特定类型的图片验证码。项目经过了多次迭代和优化，集成了多种图像处理技术，以实现高准确率和高鲁棒性。

This is a high-performance Web API based on Python Flask and OpenCV, designed to recognize two specific types of image captchas. The project has undergone multiple iterations and optimizations, integrating various image processing techniques to achieve high accuracy and robustness.

![API Demo](https://img.shields.io/badge/API-Flask-blue) ![Language](https://img.shields.io/badge/Language-Python-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 功能特性 (Features)

- **双验证码类型支持 (Dual Captcha Type Support)**:
  1.  **动态GIF数学验证码**: 能够处理闪烁的、带干扰元素的GIF动图，该动图包含一个“一位数字-运算符-一位数字”的数学表达式（支持 `+`, `-`, `*`）。
  2.  **4位字母数字验证码**: 能够处理带有复杂背景、颜色和干扰线的4位字母与数字组合的验证码。

- **先进的图像处理流水线 (Advanced Image Processing Pipeline)**:
  - **GIF清理**: 通过逐帧比对，将动态GIF合并为一张无动态干扰的干净静态图。
  - **图像增强**: 使用膨胀(Dilation)操作来加粗字符笔画，并有效去除背景噪点。
  - **智能二值化**: 针对不同验证码类型，采用全局或局部自适应阈值，完美地将字符与背景分离。
  - **精确分割**: 对处理后的图像进行轮廓查找，精确地分割出每个字符。
  - **基于形状的符号识别**: 彻底摆脱对OCR识别符号的依赖，通过霍夫线条变换(Hough Line Transform)等几何学方法，精确判断 `+`, `-`, `*`。
  - **智能后处理**: 包含逻辑判断，能将因分割而断裂的 `=` 符号重新合并。

- **生产级特性 (Production-Ready Features)**:
  - **并发安全**: 为每个请求创建唯一的UUID和独立的调试文件夹，完全支持高并发场景。
  - **按需调试**: API支持一个可选的 `debug: true` 参数，只有在需要时才会在服务器上保存详细的中间处理图像，否则不产生任何磁盘I/O，保证了生产环境的性能。
  - **全面的日志记录**: 对每一个关键步骤都输出详细日志，方便追踪和排查问题。
  - **Cookie支持**: API请求可以携带`cookie`，以访问需要会话验证的验证码。

## 技术栈 (Tech Stack)

- **Web框架**: Flask
- **OCR引擎**: ddddocr
- **图像处理**: OpenCV, Pillow (PIL), NumPy
- **HTTP请求**: requests
- **其他**: uuid

---

## 快速开始 (Quick Start)

### 1. 环境准备 (Prerequisites)

- Python 3.7+

### 2. 克隆仓库 (Clone the Repository)

```bash
git clone https://your-repository-url.com/captcha-api.git
cd captcha-api
```

### 3. 安装依赖 (Install Dependencies)

项目的所有依赖都已在 `requirements.txt` 文件中列出。

```bash
pip install -r requirements.txt
```

**`requirements.txt` 内容:**
```
Flask
requests
Pillow
numpy
ddddocr
opencv-python-headless
```

### 4. 运行API服务器 (Run the API Server)

```bash
python app.py
```

服务器默认将在 `http://0.0.0.0:5005` 上启动。在生产环境中，强烈建议使用专业的WSGI服务器，如Gunicorn：

```bash
gunicorn --workers 4 --bind 0.0.0.0:5005 app:app
```

---

## API 使用文档 (API Documentation)

API提供了两个主要的识别端点。

### 请求格式 (Request Format)

所有请求都应为 `POST` 方法，`Content-Type` 为 `application/json`。

**请求体 (Request Body):**
```json
{
  "url": "http://example.com/path/to/captcha.gif",
  "cookie": "session_id=abc123xyz; ...",
  "debug": true,
  "attempts": 10
}
```
- `url` (string, **必需**): 验证码图片的URL。
- `cookie` (string, *可选*): 用于访问受保护验证码的Cookie字符串。
- `debug` (boolean, *可选*, 默认: `false`): 如果设为`true`，服务器将保存此次请求的所有中间处理图片到一个唯一的调试文件夹中。
- `attempts` (integer, *可选*, 默认: `10`): 仅用于4位字母数字验证码，表示识别尝试的次数。

### 响应格式 (Response Format)

**成功响应 (Success Response):**
```json
{
  "result": "18",
  "source": "9*2",
  "success": true,
  
  "msg": "Recognition successful."
}
```
- `result`: 最终的识别结果（计算结果或4位字符串）。
- `source`: 识别出的原始信息（数学表达式或4位字符串）。
- `success`: `true` 表示成功。
- `msg`: 成功或失败的消息。

**失败响应 (Error Response):**
```json
{
  "result": null,
  "source": null,
  "success": false,
  "msg": "Error message details."
}
```

### 端点详情 (Endpoints)

#### 1. `/recognize_gif_math`

用于识别动态GIF数学验证码。

**示例请求 (cURL):**
```bash
curl -X POST http://127.0.0.1:5005/recognize_gif_math \
-H 'Content-Type: application/json' \
-d '{
  "url": "http://example.com/math_captcha.gif",
  "debug": true
}'
```

#### 2. `/recognize_alphanumeric`

用于识别4位字母数字验证码。

**示例请求 (cURL):**
```bash
curl -X POST http://127.0.0.1:5005/recognize_alphanumeric \
-H 'Content-Type: application/json' \
-d '{
  "url": "http://example.com/alphanum_captcha.png",
  "debug": true
}'
```

---

## 调试 (Debugging)

当请求中包含 `"debug": true` 时，服务器会在其运行目录下的 `debug_images/` 文件夹中创建一个以UUID命名的子文件夹。该文件夹将包含此次请求处理过程中的所有中间图片，例如：

- `math_1_clean_composite.png`: 合并后的干净灰度图。
- `math_2_enhanced_image.png`: 经过膨胀强化的图。
- `math_3_segmentation_mask.png`: 用于分割的最终黑白蒙版。
- `math_4_char_0.png`, `math_4_char_1.png`, ...: 每个被分割出的字符块。

这为分析和优化特定类型的验证码提供了极大的便利。

---

## 许可证 (License)

本项目采用 [MIT许可证](LICENSE)。
