import io
import os
import uuid
import re
import boto3
from PIL import Image
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

def extract_badge_id(text_response, image_bytes, file_name):
    s3 = boto3.client("s3")
    s3_bucket = 'techbloom-ballots'
    key = ""
    blocks = text_response["Blocks"]
    lines = [block['Text'] for block in text_response['Blocks'] if block['BlockType'] == 'LINE']
    full_text = "\n".join(lines)
    match = re.search(r"reg\s*id\s*#", full_text, re.IGNORECASE | re.DOTALL)
    if match:
        start_index = match.end()
        after_text = full_text[start_index: start_index + 15]
        digits_match = re.search(r'(\d(?:\s*\d){5})', after_text)
        if digits_match:
            badge_id = re.sub(r"\s+", "", digits_match.group(1))
            print("Badge ID found after: ", badge_id)
            return badge_id, key
        else:
            before_text = full_text[:match.start()]
            before_digits = re.findall(r'(\d(?:\s*\d){5})', before_text)
            if before_digits:
                badge_id = re.sub(r"\s+", "", before_digits[-1])
                print("Badge ID found before: ", badge_id)
                return badge_id, key
            else:
                line_blocks = [b for b in blocks if b["BlockType"] == "LINE"]
                reg_lines = [b for b in line_blocks if re.search(r"reg|id#", b["Text"], re.IGNORECASE)]
                if not reg_lines:
                    print("No reg/id# lines found")
                    key = "ERROR IN READING"
                    return "", key
                reg_lines.sort(key=lambda b: b["Geometry"]["BoundingBox"]["Top"])
                merged_group = [reg_lines[0]]
                groups = []

                for line in reg_lines[1:]:
                    prev_box = merged_group[-1]["Geometry"]["BoundingBox"]
                    curr_box = line["Geometry"]["BoundingBox"]
                    if curr_box["Top"] - (prev_box["Top"] + prev_box["Height"]) <= prev_box["Height"]:
                        merged_group.append(line)
                    else:
                        groups.append(merged_group)
                        merged_group = [line]
                groups.append(merged_group)

                reg_group = groups[0]
                lefts = [b["Geometry"]["BoundingBox"]["Left"] for b in reg_group]
                tops = [b["Geometry"]["BoundingBox"]["Top"] for b in reg_group]
                rights = [b["Geometry"]["BoundingBox"]["Left"] + b["Geometry"]["BoundingBox"]["Width"] for b in
                          reg_group]
                bottoms = [b["Geometry"]["BoundingBox"]["Top"] + b["Geometry"]["BoundingBox"]["Height"] for b in
                           reg_group]

                reg_box = {
                    "Left": min(lefts),
                    "Top": min(tops),
                    "Width": max(rights) - min(lefts),
                    "Height": max(bottoms) - min(tops)
                }

                reg_center_y = reg_box["Top"] + reg_box["Height"] / 2
                reg_right = reg_box["Left"] + reg_box["Width"]

                right_candidates = []
                for line in line_blocks:
                    box = line["Geometry"]["BoundingBox"]
                    if box["Left"] > reg_right:
                        vert_center = box["Top"] + box["Height"] / 2
                        vert_dist = abs(vert_center - reg_center_y)
                        area = box["Width"] * box["Height"]
                        right_candidates.append((vert_dist, -area, box))

                if right_candidates:

                    right_candidates.sort(key=lambda x: (x[0], x[1]))
                    target_box = right_candidates[0][2]
                    print("Selected box: largest line to the right of REG/ID# region")
                else:
                    target_box = reg_box
                    print("No box to the right: using REG/ID# region")

                image = Image.open(io.BytesIO(image_bytes))
                img_w, img_h = image.size
                left = int(target_box['Left'] * img_w)
                top = int(target_box['Top'] * img_h)
                right = int((target_box['Left'] + target_box['Width']) * img_w)
                bottom = int((target_box['Top'] + target_box['Height']) * img_h)

                cropped = image.crop((left, top, right, bottom))
                file_name = os.path.splitext(file_name)[0]
                s3_key = f"bad_badges/{file_name}_badge.png"

                buffer = io.BytesIO()
                cropped.save(buffer, format="PNG")
                buffer.seek(0)
                s3.upload_fileobj(buffer, s3_bucket, s3_key)
                print(f"Uploaded bad badge to s3://{s3_bucket}/{s3_key}")
                return "", s3_key
    key = "ERROR IN READING"
    return "", key


def get_cell_text(cell, blocks):
    if 'Relationships' in cell:
        for rel in cell['Relationships']:
            if rel['Type'] == 'CHILD':
                texts = [blocks[child_id]['Text'] for child_id in rel['Ids'] if 'Text' in blocks[child_id]]
                return ' '.join(texts)
    return ''

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

def extract_categories_items(analyze_response, blocks, file_name, image_bytes):
    s3 = boto3.client("s3")
    s3_bucket = 'techbloom-ballots'
    key = ""
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
                        current_row.append(get_cell_text(cell))
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
                        print(f"Warning: still contains non-letters -> {fixed}")
                    row[-4] = fixed
                    last_three_values = [row[-3].strip(), row[-2].strip(), row[-1].strip()]
                    if all(v.strip() == "" for v in last_three_values):
                        continue
                    if all(v.isdigit() for v in last_three_values):
                        item_number = "".join(last_three_values)
                        print(row[-4] + f" Item number: {item_number}")
                        extracted.append({
                            'Category ID': row[-4],
                            'Item Number': item_number,
                            'Status': 'readable',
                            'Key': ""
                        })
                    else:
                        item_number = "".join(last_three_values)
                        column_category = "Category" + row[-4]
                        row_cells = [c for c in cells if c['RowIndex'] == row_num]
                        col_indices_sorted = sorted({c['ColumnIndex'] for c in row_cells})
                        last_three_indices = col_indices_sorted[-3:]
                        last_three_cells = [c for c in row_cells if c['ColumnIndex'] in last_three_indices]
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

                            print(f"Uploaded combined last 3 columns for row {row_num} to s3://{s3_bucket}/{s3_key}")
                            key = s3_key
                            extracted.append({
                                'Category ID': row[-4],
                                'Item Number': item_number,
                                'Status': 'unreadable',
                                'Key': key
                            })
                        else:
                            print(f"Could not locate last 3 Textract cells for row {row_num}")
                            extracted.append({
                                'Category ID': row[-4],
                                'Item Number': item_number,
                                'Status': 'unreadable',
                                'Key': "COULD NOT UPLOAD TO S3"
                            })
                else:
                    print("Error with table detection, please check " + file_name)
    return extracted

textract = boto3.client('textract', region_name='us-east-2')

def process_image(image_bytes, file_name):

    text_response = textract.detect_document_text(
        Document={'Bytes': image_bytes}
    )
    analyze_response = textract.analyze_document(
        Document={'Bytes': image_bytes},
        FeatureTypes=["TABLES"]
    )
    blocks = {block['Id']: block for block in analyze_response['Blocks']}
    badge_id, key = extract_badge_id(text_response, image_bytes, file_name)
    items = extract_categories_items(analyze_response, blocks, file_name, image_bytes)

    print(f"Extracted Badge ID: {badge_id}")

    print(f"[process_image] Extracted {len(items)} votes from {file_name}")
    return {
        "badge_id" : badge_id,
        "badge_key" : key,
        "items" : items
    }
