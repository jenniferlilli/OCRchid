import io
import os
import uuid
import re
import boto3
import cv2
import numpy as np
import timm
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import center_of_mass
from db_utils import insert_vote, insert_badge
from db_model import get_db_session, ValidBadgeIDs, Ballot
from google.oauth2 import service_account
from google.cloud import vision
import json
from credentials import decode_google_keys
decode_google_keys()

with open("even-flight.json", "r") as f:
    credentials_info = json.load(f)

credentials = service_account.Credentials.from_service_account_info(credentials_info)
client = vision.ImageAnnotatorClient(credentials=credentials)


def load_model():
    model = timm.create_model("resnet18", pretrained=False, num_classes=10)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            "https://huggingface.co/gpcarl123/resnet18_mnist/resolve/main/resnet18_mnist.pth",
            map_location="cpu",
            file_name="resnet18_mnist.pth",
        )
    )
    model.eval()
    return model

_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def badge_id_exists(session_id: str, badge_id: str) -> bool:
    session = get_db_session()
    try:
        session_id = uuid.UUID(str(session_id))
        exists = session.query(ValidBadgeIDs).filter(
            ValidBadgeIDs.session_id == session_id,
            ValidBadgeIDs.badge_id == badge_id
        ).first() is not None
    finally:
        session.close()
    return exists

def readable_badge_id_exists(session_id: str, badge_id: str) -> bool:
    session = get_db_session()
    try:
        session_id = uuid.UUID(str(session_id))
        exists = session.query(Ballot).filter(
            Ballot.session_id == session_id,
            Ballot.badge_id == badge_id,
            Ballot.badge_status == 'readable'
        ).first() is not None
    finally:
        session.close()
    return exists

def extract_and_normalize_largest_digit(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    inverted = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    height, width = binary.shape
    diag = np.sqrt(width**2 + height**2)
    hor_kernel_len = max(1, int(0.9 * width))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hor_kernel_len, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    binary = cv2.subtract(binary, detected_lines)
    open_kernel_size = max(3, int(0.005 * diag))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    dilate_kernel_size = max(3, int(0.03 * diag))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))
    dilated = cv2.dilate(binary, kernel_dilate, iterations=1)

    border_margin = int(0.1 * min(height, width))
    dilated[:border_margin, :] = 0
    dilated[-border_margin:, :] = 0
    dilated[:, :border_margin] = 0
    dilated[:, -border_margin:] = 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    image_center = np.array([width / 2, height / 2])
    best_label = -1
    best_score = -np.inf

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < 0.0003 * (width * height) or h > 0.9 * height:
            continue
        cx, cy = centroids[label]
        dist2 = (cx - image_center[0]) ** 2 + (cy - image_center[1]) ** 2
        density = area / (w * h + 1e-5)
        score = area * density - 0.1 * dist2
        if score > best_score:
            best_score = score
            best_label = label

    if best_label == -1:
        print(f"No valid digit found.")
        return None

    selected = {best_label}
    queue = [best_label]
    margin = int(0.06 * diag)

    while queue:
        label = queue.pop()
        x, y, w, h, _ = stats[label]
        grow_x1 = max(x - margin, 0)
        grow_y1 = max(y - margin, 0)
        grow_x2 = min(x + w + margin, width)
        grow_y2 = min(y + h + margin, height)

        for other_label in range(1, num_labels):
            if other_label in selected:
                continue
            ox, oy, ow, oh, oa = stats[other_label]
            if oa < 0.0004 * width * height or oh > 0.9 * height or ow > 0.9 * width:
                continue
            if ox + ow < grow_x1 or ox > grow_x2 or oy + oh < grow_y1 or oy > grow_y2:
                continue
            selected.add(other_label)
            queue.append(other_label)

    selected_centroids = [centroids[i] for i in selected]
    for other_label in range(1, num_labels):
        if other_label in selected or stats[other_label][4] < 0.0002 * width * height:
            continue
        dist = torch.cdist(
            torch.tensor([centroids[other_label]], dtype=torch.float32),
            torch.tensor(selected_centroids, dtype=torch.float32)
        )
        if dist.min().item() < 0.08 * diag:
            selected.add(other_label)

    merged_mask = np.zeros_like(dilated, dtype=np.uint8)
    for label in selected:
        merged_mask[labels == label] = 255

    ys, xs = np.where(merged_mask)
    if len(xs) == 0 or len(ys) == 0:
        print("Empty merged digit.")
        return None
    x1, x2 = np.min(xs), np.max(xs)
    y1, y2 = np.min(ys), np.max(ys)

    pad = max(5, int(0.03 * max(x2 - x1, y2 - y1)))
    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, width)
    y2 = min(y2 + pad, height)

    gray_crop = gray[y1:y2 + 1, x1:x2 + 1].astype(np.float32)
    gray_crop = 255.0 - gray_crop
    gamma = 0.5
    gray_crop = np.power(gray_crop / 255.0, gamma) * 255.0
    gray_crop -= gray_crop.min()
    if gray_crop.max() > 0:
        gray_crop /= gray_crop.max()
    else:
        gray_crop[:] = 0.0
    h_new, w_new = gray_crop.shape
    if h_new > w_new:
        diff = h_new - w_new
        pad_left = diff // 2
        pad_right = diff - pad_left
        gray_crop = np.pad(gray_crop, ((0, 0), (pad_left, pad_right)), mode='constant')
    elif w_new > h_new:
        diff = w_new - h_new
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        gray_crop = np.pad(gray_crop, ((pad_top, pad_bottom), (0, 0)), mode='constant')
    resized_digit = cv2.resize(gray_crop, (20, 20), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas[4:24, 4:24] = resized_digit
    cy, cx = center_of_mass(canvas)
    shift_y = int(np.round(14 - cy))
    shift_x = int(np.round(14 - cx))
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    canvas = cv2.warpAffine(canvas, M, (28, 28), flags=cv2.INTER_LINEAR, borderValue=0)
    digit_resized = (canvas * 255).astype(np.uint8)

    return digit_resized

def extract_cells_from_contours(contours_v, contours_h):
    vertical_x = sorted([cv2.boundingRect(cnt)[0] for cnt in contours_v])
    horizontal_y = sorted([cv2.boundingRect(cnt)[1] for cnt in contours_h])

    cells = []
    for i in range(len(horizontal_y) - 1):
        for j in range(len(vertical_x) - 1):
            x1 = vertical_x[j]
            x2 = vertical_x[j + 1]
            y1 = horizontal_y[i]
            y2 = horizontal_y[i + 1]
            cells.append((x1, y1, x2, y2))
    return cells

def enhance_faint_strokes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    inverted = 255 - gray
    gamma = 0.9
    brightened = np.power(inverted / 255.0, gamma) * 255
    brightened = np.clip(brightened, 0, 255).astype(np.uint8)
    strokes_darkened = 255 - brightened

    min_val = np.percentile(strokes_darkened, 2)
    max_val = np.percentile(strokes_darkened, 98)
    contrast_stretched = np.clip((strokes_darkened - min_val) * 255.0 / (max_val - min_val + 1e-5), 0, 255).astype(np.uint8)

    return contrast_stretched

def extract_badge_id(img, file_name):
    _, buffer = cv2.imencode('.png', img)
    content = buffer.tobytes()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    texts = response.full_text_annotation.text
    digits_only = re.sub(r'\D', '', texts)
    if not len(digits_only) == 6:
        key = upload_badge_to_s3(img, file_name)
        return digits_only, key
    else:
        return digits_only, ""

def deskew_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    edges = cv2.Canny(gray, 30, 150, apertureSize=3, L2gradient=True)
    lines = []
    hough_lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)
    if hough_lines is not None:
        lines.extend(hough_lines[:, 0].tolist())
    houghp_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                   minLineLength=min(image.shape[1] // 4, 50),
                                   maxLineGap=10)
    if houghp_lines is not None:
        for x1, y1, x2, y2 in houghp_lines[:, 0]:
            dx = x2 - x1
            dy = y2 - y1
            angle = np.arctan2(dy, dx)
            lines.append([0, angle])
    if not lines:
        return image
    angles = []
    for line in lines:
        if len(line) == 2:
            rho, theta = line
            angle = np.degrees(theta) - 90
        else:
            angle = np.degrees(line[1]) - 90

        if -10 < angle < 10:
            angles.append(angle)
    if not angles:
        return image
    mean_angle = np.mean(angles)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, mean_angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return deskewed

def upload_badge_to_s3(image, file_name, object_prefix="low_confidence_badge"):
    s3 = boto3.client("s3")
    success, encoded_image = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Failed to encode image to JPEG format.")
    buffer = io.BytesIO(encoded_image.tobytes())

    base_name = os.path.splitext(file_name)[0]
    key = f"{object_prefix}/{base_name}.jpg"

    s3.upload_fileobj(
        Fileobj=buffer,
        Bucket='techbloom-ballots',
        Key=key,
        ExtraArgs={"ContentType": "image/jpeg"}
    )

    print(f"[S3] Uploaded to s3://techbloom-ballots/{key}")
    return key

def upload_vote_to_s3(image, file_name, object_prefix="low_confidence_vote"):
    s3 = boto3.client("s3")
    success, encoded_image = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Failed to encode image to JPEG format.")
    buffer = io.BytesIO(encoded_image.tobytes())

    base_name = os.path.splitext(file_name)[0]

    unique_id = str(uuid.uuid4())

    key = f"{object_prefix}/{base_name}_{unique_id}.jpg"

    s3.upload_fileobj(
        Fileobj=buffer,
        Bucket='techbloom-ballots',
        Key=key,
        ExtraArgs={"ContentType": "image/jpeg"}
    )

    print(f"[S3] Uploaded to s3://techbloom-ballots/{key}")
    return key

def detect_grid_lines(enhanced_gray, box, sidebar_thresh=0.6):
    x, y, w, h = cv2.boundingRect(box)
    roi = enhanced_gray[y:y+h, x:x+w]
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    vertical_profile = np.sum(binary, axis=0) / 255
    norm_profile = cv2.GaussianBlur((vertical_profile / h).astype(np.float32), (15, 1), 0)
    sidebar_check_span = max(3, min(20, int(0.01 * w)))
    pre_sidebar_check_span = max(5, min(40, int(0.02 * w)))
    left_sidebar_end = 0
    for i in range(pre_sidebar_check_span + sidebar_check_span, w // 2):
        before = norm_profile[i - sidebar_check_span - pre_sidebar_check_span: i - sidebar_check_span]
        window = norm_profile[i - sidebar_check_span: i]
        if np.mean(before) > sidebar_thresh and np.mean(window) < sidebar_thresh:
            left_sidebar_end = i
            break
    right_sidebar_start = w
    for i in range(w - pre_sidebar_check_span - sidebar_check_span, w // 2, -1):
        before = norm_profile[i: i + pre_sidebar_check_span]
        window = norm_profile[i - sidebar_check_span: i]
        if np.mean(before) > sidebar_thresh and np.mean(window) < sidebar_thresh:
            right_sidebar_start = i
            break
    binary[:, :left_sidebar_end] = 0
    binary[:, right_sidebar_start:] = 0
    text_filter_width = min(max(3, int(0.003 * w)), 20)
    text_filter_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (text_filter_width, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, text_filter_kernel, iterations=1)
    vertical_kernel_len = max(10, int(0.03 * h))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_len))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)
    horizontal_kernel_len = max(10, int(0.03 * w))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_len, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)
    contours_v, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_h_all, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_line_width_ratio = 0.5
    max_line_thickness_ratio = 0.03
    contours_h = []
    for cnt in contours_h_all:
        x_, y_, w_, h_ = cv2.boundingRect(cnt)
        if (w_ >= min_line_width_ratio * w) and (h_ <= max_line_thickness_ratio * h):
            contours_h.append(cnt)
    contours_v = list(contours_v)
    if 0 < right_sidebar_start < w:
        synthetic_line = np.array([[[right_sidebar_start, 0]], [[right_sidebar_start, h - 1]]], dtype=np.int32)
        contours_v.append(synthetic_line)
    if 0 < left_sidebar_end < w:
        synthetic_left = np.array([[[left_sidebar_end, 0]], [[left_sidebar_end, h-1]]], dtype=np.int32)
        contours_v.append(synthetic_left)
    roi_cluster_thresh = int(0.02 * w)
    x_coords = [cv2.boundingRect(cnt)[0] for cnt in contours_v]
    used = [False] * len(x_coords)
    clusters = []
    for i in range(len(x_coords)):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        for j in range(i + 1, len(x_coords)):
            if not used[j] and abs(x_coords[j] - x_coords[i]) < roi_cluster_thresh:
                cluster.append(j)
                used[j] = True
        clusters.append(cluster)
    filtered_contours_v = []
    for group in clusters:
        best_idx = max(group, key=lambda idx: cv2.boundingRect(contours_v[idx])[3])  # tallest
        filtered_contours_v.append(contours_v[best_idx])
    contours_v = filtered_contours_v
    cells = extract_cells_from_contours(contours_v, contours_h)
    print(f"Detected vertical lines: {len(contours_v)}")
    print(f"Detected horizontal lines: {len(contours_h)}")
    return (contours_v, contours_h), (x, y), cells, roi

def find_main_rectangles(img, file_name):
    badge_id = ""
    key = ""
    badge_found = False
    cropped_cells = []
    enhanced = deskew_image(enhance_faint_strokes(img))
    if enhanced is None:
        return None, "", ""
    original_img = enhanced.copy()
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
    blurred = cv2.GaussianBlur(enhanced, (1, 1), 0)
    h, w = blurred.shape
    block_size = max(15, ((min(h, w) // 100) | 1))
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, block_size, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(255 - morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = h * w
    candidates = []
    cell_add_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.01 * img_area or area > 0.95 * img_area:
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        w_box = np.linalg.norm(box[0] - box[1])
        h_box = np.linalg.norm(box[1] - box[2])
        aspect_ratio = max(w_box, h_box) / (min(w_box, h_box) + 1e-5)
        if aspect_ratio > 10 or aspect_ratio < 0.1:
            continue
        candidates.append((area, box))
    print(len(candidates))
    candidates.sort(key=lambda x: -x[0])
    top_2 = candidates[:3]
    if not len(top_2) == 3:
        raise ValueError(f"Could not find third box to extract badge ID from: {file_name}")
    output = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    for idx, (_, box) in enumerate(reversed(top_2)):
        cv2.drawContours(output, [box], 0, (0, 0, 255), 2)
        if idx == 0:
            x, y, w, h = cv2.boundingRect(box)
            third_roi = original_img[y:y + h, x:x + w]
            badge_id, key = extract_badge_id(third_roi, file_name)
            continue
        (contours_v, contours_h), (x, y), cells, roi = detect_grid_lines(enhanced, box)
        print(f"Number of detected cells: {len(cells)}")
        for c in contours_v:
            c_offset = c + np.array([[x, y]])
            cv2.drawContours(output, [c_offset], -1, (0, 255, 0), thickness=3)
        for c in contours_h:
            c_offset = c + np.array([[x, y]])
            cv2.drawContours(output, [c_offset], -1, (255, 0, 0), thickness=3)
        for cell_idx, (x1, y1, x2, y2) in enumerate(cells):
            pt1 = (x + x1, y + y1)
            pt2 = (x + x2, y + y2)
            cv2.rectangle(output, pt1, pt2, (255, 255, 0), 1)
            if cell_idx % 3 == 2:
                if cell_add_count == 0:
                    cell_add_count += 1
                    continue
                cell_img = roi[y1:y2, x1:x2]
                cropped_cells.append(cell_img)
                cell_add_count += 1
    return cropped_cells, badge_id, key

def extract_digits(cell_img, file_name, model):
    h, w = cell_img.shape[:2]
    segment_width = w // 3
    digits = []
    good_vote = True

    for i in range(3):
        start_x = i * segment_width
        end_x = (i + 1) * segment_width if i < 2 else w
        digit_img = cell_img[:, start_x:end_x]

        norm_digit = extract_and_normalize_largest_digit(digit_img)
        print(f"Segment {i} norm_digit is None: {norm_digit is None}")

        if norm_digit is None:
            print(f"Segment {i}: norm_digit is None; marking '?'")
            digits.append('?')
            good_vote = False
            continue

        if isinstance(norm_digit, np.ndarray):
            norm_digit = transforms.ToPILImage()(norm_digit)
        elif not isinstance(norm_digit, Image.Image):
            digits.append('?')
            good_vote = False
            continue

        try:
            input_tensor = transform(norm_digit).unsqueeze(0)
            device = next(model.parameters()).device
            input_tensor = input_tensor.to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)[0].cpu().numpy()
        
            pred_class = int(np.argmax(probabilities))
            confidence = probabilities[pred_class]
        
            print(f"Segment {i} predicted digit: {pred_class} with conf {confidence:.2f}")
            print(f"Segment {i} confidences: " + ", ".join(f"{d}:{p:.2f}" for d, p in enumerate(probabilities)))
        
        
            if int(pred_class) in {1, 4, 5}:
                print(f"Segment {i}: digit {pred_class} is excluded; marking '?'")
                digits.append('?')
                good_vote = False
            elif confidence > 0.5:
                digits.append(str(pred_class))
            else:
                print(f"Segment {i}: low confidence; marking '?'")
                digits.append('?')
                good_vote = False
        except Exception as e:
            print(f"[Segment {i}] ERROR: {e}")
            digits.append('?')
            good_vote = False
    
        

    final = ''.join(digits)
    print(f"Full 3-digit result: {final}, good_vote={good_vote}")
    return final, good_vote



def extract_text_from_cells(image, file_name, model):
    extracted = []
    item_numbers = []
    CATEGORY_IDS = [
        "A", "B", "C", "D", "E", "G", "H", "I", "J", "F",
        "FA", "FB", "FC", "FD", "FE", "FF", "FG", "FH",
        "K", "KA", "KB", "KC", "L", "M", "N", "O", "P",
        "PA", "Q", "QA", "R", "RA", "S", "T", "U", "V",
        "W", "WA", "X", "Y", "YA"
    ]

    cropped_cells, badge_id, key = find_main_rectangles(image, file_name)
    if cropped_cells is not None:
        for i, cell_img in enumerate(cropped_cells):
            if i >= len(CATEGORY_IDS):
                continue
            current, good_vote = extract_digits(cell_img, file_name, model)
            cat_id = CATEGORY_IDS[i]
            item_numbers.append(current)

            if good_vote and len(current) == 3:
                vote_key = ""
                extracted.append({
                    'Category ID': cat_id,
                    'Item Number': current,
                    'Status': 'readable',
                    'Key': vote_key
                })
            else:
                vote_key = upload_vote_to_s3(cell_img, file_name, cat_id)
                extracted.append({
                    'Category ID': cat_id,
                    'Item Number': current,
                    'Status': 'unreadable',
                    'Key': vote_key
                })

    return extracted, badge_id, key



def process_image(image_bytes, file_name, model):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    item_extract, badge_id, key = extract_text_from_cells(image_cv, file_name, model)

    print(f"Extracted Badge ID: {badge_id}")

    print(f"[process_image] Extracted {len(item_extract)} votes from {file_name}")
    return {
        "badge_id" : badge_id,
        "badge_key" : key,
        "items" : item_extract
    }
