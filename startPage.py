import os, base64
import boto3
import json
import re
import uuid
from uuid import uuid4
from uuid import UUID
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from markupsafe import Markup
from flask_cors import CORS
from sqlalchemy import func, desc
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
load_dotenv()
from db_model import (
    ValidBadgeIDs,
    Ballot,
    UploadedZip,
    UserSession,
    OCRResult,
    BallotVotes,
    SessionLocal,
    Product
)
from db_utils import validate_user_session, insert_user_session, insert_products
from easy_ocr import process_image, badge_id_exists, readable_badge_id_exists
from io import BytesIO
from openpyxl import Workbook
import zipfile
from botocore.exceptions import NoCredentialsError, ClientError
from collections import defaultdict, Counter
from openpyxl import load_workbook
import gspread
from google.oauth2.service_account import Credentials

import random
from tasks import preprocess_zip_task  
from flask import jsonify
from credentials import decode_google_keys
decode_google_keys()
s3 = boto3.client('s3', region_name='us-east-2')
bucket_name = 'techbloom-ballots'

app = Flask(__name__, template_folder='.')
app.secret_key = os.getenv("FLASK_SECRET_KEY")
CORS(app)

ALLOWED_BADGE_EXTENSIONS = {'csv', 'txt'}
ALLOWED_ZIP_EXTENSIONS = {'zip'}

from celery_app import make_celery
celery = make_celery()

def get_db_session():
    return SessionLocal()

def upload_to_s3(file_obj, bucket, key):
    try:
        s3.upload_fileobj(file_obj, bucket, key)
        print(f"Uploaded to S3: {key}")
        return True
    except (NoCredentialsError, ClientError) as e:
        print(f"Upload failed: {e}")
        return False

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def is_junk_file(file_info):
    filename = file_info.filename
    basename = os.path.basename(filename)
    return (
        filename.startswith('__MACOSX/') or
        '/__MACOSX/' in filename or
        basename.startswith('._') or
        basename.startswith('.') or
        basename in ('Thumbs.db', 'desktop.ini') or
        file_info.is_dir() or
        not basename.strip()
    )


@app.route('/login')
def login():
    return render_template('templates/a_login.html')

@app.route('/logout')
def logout():
    return render_template('templates/a_login.html')



@app.route('/create-session', methods=['GET', 'POST'])
def create_session():
    session['joined_existing'] = False

    if request.method == 'POST':
        password = request.form.get('password')
        db_session = get_db_session()
        session_id = uuid4()
        print("Generated UUID:", session_id)
        existing = db_session.query(UserSession).filter_by(session_id=session_id).first()
        while existing:
            session_id = uuid4()
            existing = db_session.query(UserSession).filter_by(session_id=session_id).first()

        db_session.add(UserSession(session_id=session_id, password=password))
        db_session.commit()
        db_session.close()

        session['session_id'] = str(session_id)
        session['short_session_id'] = str(session_id)[:8]

        flash('Generated Session ID successfully.')
        return redirect(url_for('upload_files'))

    return render_template('templates/a_createSession.html')


@app.route('/join-session', methods=['GET', 'POST'])
def join_session():
    session['joined_existing'] = True
    if request.method == 'POST':
        session_id = UUID(request.form.get('session_id'))
        password = request.form.get('password')
        db_session = get_db_session()
        user_session = db_session.query(UserSession).filter_by(
            session_id=session_id, password=password
        ).first()
        db_session.close()
        if user_session:
            session['session_id'] = str(session_id)
            session['joined_existing'] = True
            flash(f'Joined session successfully.')
            return redirect(url_for('upload_files'))
        else:
            flash('Invalid session ID or password.')
            return redirect(request.url)
    return render_template('templates/a_joinSession.html')

@app.route('/upload-file', methods=['GET', 'POST'])
def upload_files():
    session_id = session.get('session_id')
    short_session_id = session.get('short_session_id')
    if not session_id:
        flash('Please log in or create a session first.')
        return redirect(url_for('login'))

    db_session = get_db_session()
    joined_existing = session.get('joined_existing', False)

    if request.method == 'POST':
        badgeFile = request.files.get('badge_file')
        zipFile = request.files.get('zip_file')

        if not badgeFile and not zipFile:
            if joined_existing:
                flash('No new files selected. Using existing files.')
                db_session.close()
                return redirect(url_for('dashboard'))
            else:
                flash('Please upload at least badge file or ZIP file.')
                db_session.close()
                return redirect(request.url)

        if badgeFile and allowed_file(badgeFile.filename, ALLOWED_BADGE_EXTENSIONS):
            try:
                badge_lines = badgeFile.read().decode('utf-8').splitlines()
                for line in badge_lines:
                    badge_id = line.strip()
                    if badge_id:
                        db_session.add(ValidBadgeIDs(session_id=session_id, badge_id=badge_id))
                db_session.commit()
                flash('Badge IDs uploaded successfully.')
            except UnicodeDecodeError:
                flash('Badge file must be UTF-8 encoded text (.csv or .txt).', 'error')
                db_session.close()
                return redirect(request.url)
        elif badgeFile:
            flash('Invalid badge file. Must be .csv or .txt')
            db_session.close()
            return redirect(request.url)

        if zipFile and allowed_file(zipFile.filename, ALLOWED_ZIP_EXTENSIONS):
            filename = secure_filename(zipFile.filename)
            zip_bytes = zipFile.read()
            zip_key = f'{session_id}/{filename}'

            if upload_to_s3(BytesIO(zip_bytes), bucket_name, zip_key):
                db_session.add(UploadedZip(session_id=session_id, filename=filename))
                db_session.commit()

                try:
                    local_zip_path = os.path.join(os.getcwd(), 'uploads', filename)
                    os.makedirs('uploads', exist_ok=True)
                    with open(local_zip_path, "wb") as f:
                        f.write(zip_bytes)
                    print(f"[Flask] Sending task to Celery with zip_key: {zip_key}, session_id: {session_id}")
                    preprocess_zip_task.delay(zip_key, session_id)
                    print(f"[Flask] Task sent successfully")                        
                    flash("ZIP file uploaded to S3. Processing started in background.")
                except Exception as e:
                    flash(f'Error starting background task: {str(e)}')
                    db_session.close()
                    return redirect(request.url)
            else:
                flash('Failed to upload ZIP to S3.')
                db_session.close()
                return redirect(request.url)
        elif zipFile:
            flash('Invalid file type. ZIP required.')
            db_session.close()
            return redirect(request.url)

        db_session.close()
        return redirect(url_for('dashboard'))

    db_session.close()
    return render_template('templates/a_upload.html',
                           short_session_id=short_session_id,
                           session_id=session_id,
                           joined_existing=joined_existing)

@app.route('/revisit-upload')
def revisit_upload():
    session['joined_existing'] = True
    return redirect(url_for('upload_files'))


@app.route('/dashboard')
def dashboard():
    session_id = session.get('session_id')
    if not session_id:
        flash('Please log in first.')
        return redirect(url_for('login'))

    top3_per_category = get_top3_votes_by_category(session_id)
    return render_template('templates/a_dashboard.html', top3_per_category=top3_per_category)


def get_top3_votes_by_category(session_id):
    session_uuid = uuid.UUID(session_id)
    db_session = get_db_session()

    vote_records = (
        db_session.query(BallotVotes)
        .join(Ballot, BallotVotes.ballot_id == Ballot.id)
        .filter(
            Ballot.session_id == session_uuid,
            Ballot.badge_status == 'readable',
            Ballot.validity == True,
            BallotVotes.is_valid == True,
            BallotVotes.vote_status == 'readable'
        )
        .all()
    )

    category_votes = defaultdict(list)
    seen_votes = set()

    for vote in vote_records:
        if not vote.category_id or not vote.vote:
            continue

        category_id = vote.category_id.upper()
        product_number = vote.vote.strip()
        key = (vote.badge_id, category_id, product_number)

        if key not in seen_votes:
            category_votes[category_id].append(product_number)
            seen_votes.add(key)

    product_records = db_session.query(Product).all()
    product_number_to_name = {p.product_number.strip(): p.product_name for p in product_records}

    valid_categories = set(category_to_name.keys())
    top3_per_category = {}
    for category, votes in sorted(category_votes.items()):
        if category not in valid_categories:
            continue
        counts = Counter(votes)
        sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

        result = []
        current_place = 1
        i = 0
        while i < len(sorted_items) and current_place <= 3:
            current_count = sorted_items[i][1]
            tied_items = []
            while i < len(sorted_items) and sorted_items[i][1] == current_count:
                tied_items.append(sorted_items[i])
                i += 1
            for num, count in tied_items:
                result.append({
                    "product_number": num,
                    "product_name": product_number_to_name.get(num, num),
                    "count": count,
                    "place": current_place,
                    "is_tie": len(tied_items) > 1
                })
            current_place += len(tied_items)

        top3_per_category[category] = result

    db_session.close()
    return top3_per_category


SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
category_to_name = {"AA": "Freshwater Rod", "AB": "Saltwater Rod", "AC": "Rod & Reel Combo", "BA": "Freshwater Reel", "BB": "Saltwater Reel", "CA": "Freshwater Soft Lure", "CB": "Saltwater Soft Lure", "CC": "Freshwater Hard Lure", "CD": "Saltwater Hard Lure", "CE": "Fly Fishing Rod", "FA": "Fly Fishing Reel", "FB": "Fly Fishing Rod & Reel Combo", "FC": "Fly Fishing Waders & Wading Boots", "FD": "Fly Line, Leader, Tippet & Line Accessory", "FE": "Fly Fishing Technical & General Apparel", "GA": "Fly Tying Vise, Tool & Material", "GB": "Fly Fishing Backpack, Bag & Luggage", "HA": "Fly Fishing Tool & Accessory", "JB": "Fishing Line", "JC": "Terminal Tackle", "KB": "Tackle Management", "KC": "Kids' Tackle", "LD": "Fishing Accessory", "ME": "Cutlery, Hand Pliers or Tool", "NF": "Soft & Hard Cooler", "PA": "Custom Tackle & Component", "PB": "Cold Weather Technical Apparel for Men", "PC": "Cold Weather Technical Apparel for Women", "PD": "Warm Weather Technical Apparel for Men", "PE": "Warm Weather Technical Apparel for Women", "QA": "Lifestyle Apparel for Men", "RB": "Lifestyle Apparel for Women", "SC": "Footwear", "TD": "Eyewear", "UE": "Novelty & Wellness", "VF": "Boat & Watercraft", "WG": "Motorized Boating Accessory", "XH": "Non Motorized Boating Accessory", "YJ": "Ice Fishing", "ZK": "Electronic"}

def get_gsheet_client():
    service_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    service_account_info = json.loads(service_json)
    
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    
    from google.oauth2.service_account import Credentials
    creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
    gc = gspread.authorize(creds)
    return gc, creds


@app.route('/export_gsheet')
def export_gsheet():
    gc, creds = get_gsheet_client()

    session_id = session.get("session_id")
    if not session_id:
        flash('Please log in or create a session first.')
        return redirect(url_for('login'))

    db_session = get_db_session()  # ← was missing
    user_session = db_session.query(UserSession).filter_by(session_id=session_id).first()  # ← was missing

    top3_per_category = get_top3_votes_by_category(session_id)

    spreadsheet = None
    if user_session.spreadsheet_id:
        try:
            spreadsheet = gc.open_by_key(user_session.spreadsheet_id)
            worksheet = spreadsheet.sheet1
            worksheet.clear()
        except Exception:
            spreadsheet = None

    if spreadsheet is None:
        spreadsheet_name = f"Top3Votes_Session_{session_id}"
        
        # Create spreadsheet in service account's drive first
        spreadsheet = gc.create(spreadsheet_name)
        
        # Immediately share with yourself as OWNER
        spreadsheet.share(
            "smcs2027.techbloom@gmail.com", 
            perm_type="user", 
            role="owner",  # This transfers ownership
            transfer_ownership=True
        )
        
        # Now remove service account's access (optional, to save its quota)
        from googleapiclient.discovery import build
        drive_service = build('drive', 'v3', credentials=creds)
        
        # Get service account email
        service_email = creds.service_account_email
        
        # Remove service account permission
        permissions = drive_service.permissions().list(fileId=spreadsheet.id).execute()
        for perm in permissions.get('permissions', []):
            if perm.get('emailAddress') == service_email:
                drive_service.permissions().delete(
                    fileId=spreadsheet.id,
                    permissionId=perm['id']
                ).execute()
        
        worksheet = spreadsheet.sheet1
        worksheet.update_title("Top 3 Results")
        
        user_session.spreadsheet_id = spreadsheet.id
        db_session.commit()

    header = [
        "Category Name", "Category ID",
        "1st Place ID", "1st Votes",
        "2nd Place ID", "2nd Votes",
        "3rd Place ID", "3rd Votes"
    ]
    worksheet.append_row(header)

    for category_id, top_votes in top3_per_category.items():
        category_name = category_to_name.get(category_id, "Unknown Category")
        row = [category_name, category_id]

        by_place = {}
        for item in top_votes:
            place = item["place"]
            if place not in by_place:
                by_place[place] = []
            by_place[place].append(item)

        for place in [1, 2, 3]:
            if place in by_place:
                items = by_place[place]
                names = ", ".join(i["product_number"] for i in items)
                votes = items[0]["count"]
                row.extend([names, votes])
            else:
                row.extend(["", ""])

        worksheet.append_row(row)

    db_session.close()  # ← was missing
    sheet_url = spreadsheet.url
    flash(Markup(f"Google Sheet created/updated: <a href='{sheet_url}' target='_blank'>{sheet_url}</a>"))
    return redirect(url_for('dashboard'))

@app.route('/cleanup_service_account_drive')
def cleanup_drive():
    """List and optionally delete files in service account's Drive"""
    from googleapiclient.discovery import build
    gc, creds = get_gsheet_client()
    drive_service = build('drive', 'v3', credentials=creds)
    
    # List all files owned by service account
    results = drive_service.files().list(
        pageSize=100,
        fields="files(id, name, createdTime, size)"
    ).execute()
    
    files = results.get('files', [])
    
    # Delete files (BE CAREFUL - this deletes permanently)
    # Comment out this loop if you just want to see the list first
    for file in files:
        print(f"Deleting: {file['name']} ({file['id']})")
        drive_service.files().delete(fileId=file['id']).execute()
    
    return jsonify({
        'message': f'Deleted {len(files)} files',
        'files': files
    })

@app.route('/review')
def review_dashboard():
    session_id = session.get("session_id")
    if not session_id:
        flash("Please log in or create a session first.")
        return redirect(url_for("login"))

    session_uuid = uuid.UUID(session_id)
    db_session = get_db_session()

    ballots_with_badge_issues = (
        db_session.query(Ballot)
        .filter(Ballot.session_id == session_uuid, Ballot.badge_status == 'unreadable', Ballot.validity == True)
        .all()
    )
    
    badges_data = []
    for ballot in ballots_with_badge_issues:
        s3_url = None
        if ballot.s3_key:
            s3_url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': ballot.s3_key},
                ExpiresIn=3600
            )
        badges_data.append({
            'name': ballot.name,
            'id': ballot.id,
            'badge_id': ballot.badge_id,
            's3_url': s3_url,
        })

    votes_with_errors = (
        db_session.query(BallotVotes, Ballot)
        .join(Ballot, BallotVotes.ballot_id == Ballot.id)
        .filter(
            Ballot.session_id == session_uuid,
            BallotVotes.vote_status == "unreadable",
            BallotVotes.is_valid == True
        )
        .all()
    )
     
    votes_data = []
    for vote, ballot in votes_with_errors:
        print("Vote:", vote.id, "ballot_id:", vote.ballot_id, "badge_id:", ballot.badge_id)
        s3_url = None
        if vote.key:
            s3_url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': vote.key},
                ExpiresIn=3600
            )
        votes_data.append({
            'vote_id': vote.id,
            'ballot_id': ballot.id,
            'category': vote.category_id,
            'current_vote': vote.vote,
            'badge_id': ballot.badge_id, 
            's3_url': s3_url,
            'name': vote.name
        })
    print(votes_data)

    print("Session ID:", session_id)
    print("Bad ballots found:", len(badges_data))
    print("Unreadable votes found:", len(votes_data))

    db_session.close()
    return render_template('templates/a_review_db.html', badges=badges_data, votes=votes_data)

@app.route('/download_ballot/<int:ballot_id>')
def download_ballot(ballot_id):
    db = get_db_session()
    ballot = db.query(Ballot).filter_by(id=ballot_id).first()
    db.close()

    if not ballot:
        return "Ballot not found", 404

    s3_key = ballot.name  # the original filename stored in S3

    file_obj = s3.get_object(Bucket=bucket_name, Key=s3_key)

    return send_file(
        BytesIO(file_obj['Body'].read()),
        as_attachment=True,
        download_name=s3_key
    )

@app.route('/fix_vote', methods=['POST'])
def fix_vote():
    session_id = session.get('session_id')
    session_id = uuid.UUID(session_id)
    vote_id = request.form.get('vote_id')
    new_vote = request.form.get('vote', '').strip()

    if not vote_id or not new_vote:
        flash('Invalid input. Please provide a vote.', 'error')
        return redirect(request.referrer or url_for('review_dashboard'))

    db_session = get_db_session()
    vote = (
        db_session.query(BallotVotes)
        .join(Ballot, BallotVotes.ballot_id == Ballot.id)
        .filter(
            BallotVotes.id == vote_id,
            Ballot.session_id == session_id
        )
        .first()
    )

    if vote is None:
        flash('Vote not found.', 'error')
        db_session.close()
        return redirect(request.referrer or url_for('review_dashboard'))

    vote.vote = new_vote
    vote.vote_status = 'readable'

    if vote.key:
        try:
            s3.delete_object(Bucket=bucket_name, Key=vote.key)
            vote.key = "" 
        except Exception as e:
            print(f"Failed to delete S3 object {vote.key}: {e}")

    db_session.commit()

    flash(f'Vote updated successfully for badge {vote.badge_id}.', 'success')
    db_session.close()
    return redirect(request.referrer or url_for('review_dashboard'))

@app.route('/fix_badge', methods=['POST'])
def fix_badge():
    session_id = session.get('session_id')
    session_id = uuid.UUID(session_id)
    id = int(request.form['id'])
    print(id)
    new_badge = request.form['badge_id'].strip()

    db_session = get_db_session()
    ballot = (
        db_session.query(Ballot)
        .filter(Ballot.id == id, Ballot.session_id == session_id)
        .first()
    )
    if not ballot:
        flash('Ballot not found.')
        db_session.close()
        return redirect(request.referrer)

    old_s3_key = ballot.s3_key

    is_valid = badge_id_exists(session_id, new_badge)
    is_duplicate = readable_badge_id_exists(session_id, new_badge)

    if is_duplicate:
        flash('Badge ID already exists.')
        db_session.close()
        return redirect(request.referrer)

    if not is_valid:
        flash('Badge ID does not exist.')
        db_session.close()
        return redirect(request.referrer)
    try:
        ballot = db_session.query(Ballot).filter_by(id=id).one()
        ballot.badge_status = 'readable'
        ballot.badge_id = new_badge
        ballot.validity = is_valid
        ballot.s3_key = ""
        db_session.commit()
    except Exception as e:
        db_session.rollback()
        print(f"DB commit failed: {e}")
        flash('Failed to update badge. Please try again.')
        db_session.close()
        return redirect(request.referrer)

    votes = db_session.query(BallotVotes).filter(BallotVotes.ballot_id == id).all()
    for vote in votes:
        vote.badge_id = new_badge
        vote.is_valid = is_valid
    db_session.commit()

    if old_s3_key:
        try:
            s3.delete_object(Bucket=bucket_name, Key=old_s3_key)
        except Exception as e:
            print(f"Failed to delete S3 object {old_s3_key}: {e}")

    db_session.close()
    flash('Badge ID updated successfully and validity checked.')
    return redirect(url_for('review_dashboard'))

@app.route('/delete_vote/<int:vote_id>')
def delete_vote(vote_id):
    session_id = session.get('session_id')
    session_id = uuid.UUID(session_id)
    db_session = get_db_session()
    vote = (
        db_session.query(BallotVotes)
        .join(Ballot, BallotVotes.ballot_id == Ballot.id)
        .filter(BallotVotes.id == vote_id, Ballot.session_id == session_id)
        .first()
    )

    if vote:
        if vote.key:
            try:
                s3.delete_object(Bucket=bucket_name, Key=vote.key)
            except Exception as e:
                print(f"Error deleting vote image {vote.key} from S3:", e)

        db_session.delete(vote)
        db_session.commit()
        flash('Vote deleted successfully', 'success')
    else:
        flash('Vote not found', 'error')

    db_session.close()
    return redirect(request.referrer or url_for('review_dashboard'))

@app.route('/delete_ballot/<int:id>')
def delete_ballot(id):
    session_id = session.get('session_id')
    session_id = uuid.UUID(session_id)
    db_session = get_db_session()
    ballot = (
        db_session.query(Ballot)
        .filter(Ballot.id == id, Ballot.session_id == session_id)
        .first()
    )

    if ballot:
        id = ballot.id
        badge_id = ballot.badge_id
        session_id = ballot.session_id

        if ballot.s3_key:
            try:
                s3.delete_object(Bucket=bucket_name, Key=ballot.s3_key)
            except Exception as e:
                print(f"Error deleting ballot S3 image: {e}")

        ocr_result = db_session.query(OCRResult).filter_by(session_id=session_id, filename=ballot.name).first()
        if ocr_result:
            if ocr_result.filename:
                try:
                    s3.delete_object(Bucket=bucket_name, Key=ocr_result.filename)
                except Exception as e:
                    print(f"Error deleting OCR S3 image: {e}")
            db_session.delete(ocr_result)

        votes = db_session.query(BallotVotes).filter_by(ballot_id=id).all()
        for vote in votes:
            if vote.key:
                try:
                    s3.delete_object(Bucket=bucket_name, Key=vote.key)
                except Exception as e:
                    print(f"Error deleting vote image {vote.key} from S3: {e}")
            db_session.delete(vote)
        
        db_session.flush()

        db_session.delete(ballot)

        db_session.commit()
        flash(f'Deleted badge ID "{badge_id}", all associated ballots, votes, and OCR result.', 'success')
    else:
        flash('Ballot not found.', 'error')

    db_session.close()
    return redirect(request.referrer or url_for('review_dashboard'))

@app.route('/')
def home():
    return redirect(url_for('login'))


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port, debug=True)
