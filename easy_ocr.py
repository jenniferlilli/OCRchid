import io
import os
import re
import uuid
import boto3
import cv2
import numpy as np
import timm
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.ndimage import center_of_mass
from db_model import get_db_session, ValidBadgeIDs, Ballot

s3 = boto3.client("s3")
s3_bucket = 'techbloom-ballots'
textract = boto3.client('textract', region_name='us-east-2')

def load_model():
    model = timm.create_model("resnet18", pretrained=False, num_classes=10)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    url = "https://huggingface.co/gpcarl123/resnet18_mnist/resolve/main/resnet18_mnist.pth"
    state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    model.load_state_dict(state_dict)

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
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
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

#looks to see if the badge id exists
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

#checks to see if there are duplicates in the system already
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

#finds badge id, if it doesn't work, uploads entire ballot to bad badges folder
def extract_badge_id(text_response, image_bytes, file_name):
    key = ""
    lines = [block['Text'] for block in text_response['Blocks'] if block['BlockType'] == 'LINE']
    full_text = "\n".join(lines)
    match = re.search(r"reg\s*id\s*#", full_text, re.IGNORECASE | re.DOTALL)
    if match:
        start_index = match.end()
        after_text = full_text[start_index: start_index + 15]
        digits_match = re.search(r'(\d(?:\s*\d){5})', after_text)
        if digits_match:
            badge_id = re.sub(r"\s+", "", digits_match.group(1))
            print("[Badge ID] Badge ID found after: ", badge_id)
            return badge_id, key
        else:
            before_text = full_text[:match.start()]
            before_digits = re.findall(r'(\d(?:\s*\d){5})', before_text)
            if before_digits:
                badge_id = re.sub(r"\s+", "", before_digits[-1])
                print("[Badge ID] Badge ID found before: ", badge_id)
                return badge_id, key
    key = f"bad_badges/{file_name}"
    buffer = io.BytesIO(image_bytes)
    buffer.seek(0)
    s3.upload_fileobj(buffer, s3_bucket, key)
    print(f"[Badge ID] Uploaded full ballot: s3://{s3_bucket}/{key}")
    return "", key

#gets the text the table cell
def get_cell_text(cell, blocks):
    if 'Relationships' in cell:
        for rel in cell['Relationships']:
            if rel['Type'] == 'CHILD':
                texts = [blocks[child_id]['Text'] for child_id in rel['Ids'] if 'Text' in blocks[child_id]]
                return ' '.join(texts)
    return ''

#fixes most common category problems
def fix_column_text(text):
    num_to_letter = {
        '0': 'O',
        '1': 'I',
        '2': 'Z',
        '5': 'S',
        '6': 'G',
        '8': 'B',
        '|': 'I',
        '!': 'I',
        '/': 'I',
        '(': 'C',
    }
    fixed = ''.join(num_to_letter.get(ch, ch) for ch in text)
    return fixed

def extract_categories_items(analyze_response, blocks, file_name, image_bytes, model):
    extracted = []
    for block in analyze_response['Blocks']:
        if block['BlockType'] == 'TABLE':
            table = []
            for rel in block.get('Relationships', []):
                if rel['Type'] == 'CHILD':
                    cells = [blocks[cell_id] for cell_id in rel['Ids'] if blocks[cell_id]['BlockType'] == 'CELL']
                    cells.sort(key=lambda c: (c['RowIndex'], c['ColumnIndex']))
                    current_row = []
                    last_row = 1
                    for cell in cells:
                        if cell['RowIndex'] != last_row:
                            table.append(current_row)
                            current_row = []
                            last_row = cell['RowIndex']
                        current_row.append(get_cell_text(cell, blocks))
                    table.append(current_row)
            if table and all((len(row) == 0 or not row[0].strip()) for row in table):
                for row in table:
                    if row:
                        del row[0]
            print("\nTable:")
            for row_num, row in enumerate(table, start=1):
                if not row:
                    continue
                if len(row) >= 5:
                    print(row[-5])
                    if len(row[-4]) > 3 or len(row[-4].strip()) < 1 or row[-5] == "Example":
                        continue
                    original = row[-4]
                    fixed = fix_column_text(original)
                    if not fixed.isalpha():
                        print(f"[Category] Still contains non-letters: fixed to {fixed}")
                    row[-4] = fixed
                    last_three_values = [row[-3].strip(), row[-2].strip(), row[-1].strip()]
                    if all(v.strip() == "" for v in last_three_values):
                        continue
                    if all(v.isdigit() for v in last_three_values):
                        item_number = "".join(last_three_values)
                        print(f"[Item] Category" + row[-4] + f" Item number: {item_number}")
                        extracted.append({
                            'Category ID': row[-4],
                            'Item Number': item_number,
                            'Status': 'readable',
                            'Key': ""
                        })
                    else:
                        row_cells = [c for c in cells if c['RowIndex'] == row_num]
                        col_indices_sorted = sorted({c['ColumnIndex'] for c in row_cells})
                        last_three_indices = col_indices_sorted[-3:]
                        last_three_cells = [c for c in row_cells if c['ColumnIndex'] in last_three_indices]
                        column_category = "Category" + row[-4]
                        if last_three_cells:
                            lefts = [c['Geometry']['BoundingBox']['Left'] for c in last_three_cells]
                            tops = [c['Geometry']['BoundingBox']['Top'] for c in last_three_cells]
                            rights = [
                                c['Geometry']['BoundingBox']['Left'] + c['Geometry']['BoundingBox']['Width']
                                for c in last_three_cells
                            ]
                            bottoms = [
                                c['Geometry']['BoundingBox']['Top'] + c['Geometry']['BoundingBox']['Height']
                                for c in last_three_cells
                            ]
                            combined_box = {
                                "Left": min(lefts),
                                "Top": min(tops),
                                "Width": max(rights) - min(lefts),
                                "Height": max(bottoms) - min(tops)
                            }
                            image = Image.open(io.BytesIO(image_bytes))
                            img_w, img_h = image.size
                            combined_digits = []
                            for idx, cell in enumerate(last_three_cells):
                                text = get_cell_text(cell, blocks).strip()
                                if text.isdigit():
                                    combined_digits.append(text)
                                else:
                                    left = int(cell['Geometry']['BoundingBox']['Left'] * img_w)
                                    top = int(cell['Geometry']['BoundingBox']['Top'] * img_h)
                                    right = int(
                                        (cell['Geometry']['BoundingBox']['Left'] + cell['Geometry']['BoundingBox'][
                                            'Width']) * img_w)
                                    bottom = int(
                                        (cell['Geometry']['BoundingBox']['Top'] + cell['Geometry']['BoundingBox'][
                                            'Height']) * img_h)
                                    cell_crop = cv2.cvtColor(np.array(image.crop((left, top, right, bottom))),
                                                             cv2.COLOR_RGB2BGR)
                                    digit_img = extract_and_normalize_largest_digit(cell_crop)

                                    if digit_img is None:
                                        combined_digits.append("?")
                                    else:
                                        digit_img = Image.fromarray(digit_img).convert("L")
                                        digit_tensor = transform(digit_img).unsqueeze(0)
                                        with torch.no_grad():
                                            output = model(digit_tensor)
                                            probs = torch.softmax(output, dim=1)
                                            confidence, pred = torch.max(probs, dim=1)

                                            if confidence.item() < 0.50:
                                                print(f"[Item] Low confidence for " + column_category, confidence.item(), ": defaulting to ?")
                                                combined_digits.append("?")
                                            else:
                                                combined_digits.append(str(pred.item()))
                            print(f"[Item] " + column_category + f" Item number: {combined_digits}")
                            if "?" in combined_digits or len(combined_digits) != 3:
                                left = int(combined_box['Left'] * img_w)
                                top = int(combined_box['Top'] * img_h)
                                right = int((combined_box['Left'] + combined_box['Width']) * img_w)
                                bottom = int((combined_box['Top'] + combined_box['Height']) * img_h)
                                cropped = image.crop((left, top, right, bottom))
                                file_name = os.path.splitext(file_name)[0]
                                s3_key = f"bad_votes/{file_name}/{column_category}.png"
                                buffer = io.BytesIO()
                                cropped.save(buffer, format="PNG")
                                buffer.seek(0)
                                s3.upload_fileobj(buffer, s3_bucket, s3_key)
                                print(
                                    f"[S3] Uploaded combined last 3 columns for row {column_category} to s3://{s3_bucket}/{s3_key}")
                                extracted.append({
                                    'Category ID': row[-4],
                                    'Item Number': combined_digits,
                                    'Status': 'unreadable',
                                    'Key': s3_key
                                })
                            else:
                                extracted.append({
                                    'Category ID': row[-4],
                                    'Item Number': combined_digits,
                                    'Status': 'readable',
                                    'Key': ""
                                })
    return extracted

def process_image(image_bytes, file_name, model):

    text_response = textract.detect_document_text(
        Document={'Bytes': image_bytes}
    )
    analyze_response = textract.analyze_document(
        Document={'Bytes': image_bytes},
        FeatureTypes=["TABLES"]
    )
    blocks = {block['Id']: block for block in analyze_response['Blocks']}
    badge_id, key = extract_badge_id(text_response, image_bytes, file_name)
    items = extract_categories_items(analyze_response, blocks, file_name, image_bytes, model)

    print(f"Extracted Badge ID: {badge_id}")

    print(f"[process_image] Extracted {len(items)} votes from {file_name}")
    return {
        "badge_id" : badge_id,
        "badge_key" : key,
        "items" : items
    }
