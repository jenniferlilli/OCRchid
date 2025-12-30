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
                    preprocess_zip_task.delay(zip_key, session_id)
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

    valid_products = db_session.query(Product.category_id, Product.product_number).all()
    valid_product_set = set((cat.upper(), num.strip()) for cat, num in valid_products)

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

        if key not in seen_votes and (category_id, product_number) in valid_product_set:
            category_votes[category_id].append(product_number)
            seen_votes.add(key)

    all_categories = sorted(set(cat.upper() for cat, _ in valid_products))

    top3_per_category = {}
    for category in all_categories:
        votes = category_votes.get(category, [])
        counts = Counter(votes)
        top_votes = counts.most_common(3)

        product_name_map = {
            num: db_session.query(Product.product_name)
                        .filter_by(category_id=category, product_number=num)
                        .scalar()
            for num, _ in top_votes
        }

        top3_per_category[category] = [
            {
                "product_number": num,
                "product_name": product_name_map.get(num, "Unknown"),
                "count": count
            }
            for num, count in top_votes
        ]

    db_session.close()
    return top3_per_category


SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
category_to_name = {"A":"Freshwater Rods","B":"Saltwater Rods","C":"Rod & Reel Combo","D":"Freshwater Reels","E":"Saltwater Reels","G":"Freshwater Soft Lures","H":"Saltwater Soft Lures","I":"Freshwater Hard Lures","J":"Saltwater Hard Lures","F":"Fly Fishing Rods","FA":"Fly Fishing Reels","FB":"Fly Fishing Rod & Reel Combo","FC":"Fly Fishing Waders & Wading Boots","FD":"Fly Lines, Leaders, Tippet & Line Accessories","FE":"Fly Fishing Technical & General Apparel","FF":"Fly Tying Vise, Tool & Material","FG":"Fly Fishing Backpacks, Bag & Luggage","FH":"Fly Fishing Tool & Accessories","K":"Fishing Line","KA":"Terminal Tackle","KB":"Tackle Management","KC":"Kids’ Tackle","L":"Fishing Accessories","M":"Cutlery, Hand Pliers or Tools","N":"Soft and Hard Coolers","O":"Custom Tackle & Components","P":"Cold Weather Technical Apparel for Men","PA":"Cold Weather Technical Apparel for Women","Q":"Warm Weather Technical Apparel for Men","QA":"Warm Weather Technical Apparel for Women","R":"Lifestyle Apparel for Men","RA":"Lifestyle Apparel for Women","S":"Footwear","T":"Eyewear","U":"Novelties & Wellness","V":"Boats & Watercraft","W":"Motorized Boating Accessories","WA":"Non Motorized Boating Accessories","X":"Ice Fishing","Y":"Electronics","YA":"Energy"}
#2026 category_to_name = {"AA":"Freshwater Rod","AB":"Saltwater Rod","AC":"Rod and Reel Combo","BA":"Freshwater Reel","BB":"Saltwater Reel","CA":"Freshwater Soft Lure","CB":"Saltwater Soft Lure","CC":"Freshwater Hard Lure","CD":"Saltwater Hard Lures","CE":"Wired and Skirted Lure","FA":"Fly Fishing Accessory","FB":"Fly Fishing Technical Apparel","FC":"Fly Tying Tools and Materials","FD":"Fly Fishing Rod","FE":"Fly Fishing Reel","GA":"Braided Fishing Line","GB":"Non-Braided Fishing Line","HA":"Terminal Tackle","JB":"Tackle Management","JC":"Kids Tackle","KC":"Fishing Accessory","LD":"Cutlery, Hand Pliers or Tool","ME":"Cooler","NF":"Custom Tackle and Component","PA":"Men’s Technical Apparel","PB":"Women’s Technical Apparel","PC":"Extreme Weather Technical Apparel","PD":"Men's Lifestyle Apparel","PE":"Women’s Lifestyle Apparel","QA":"Footwear","RB":"Eyewear","SC":"Novelties and Wellness","TD":"Boats and Watercraft","UE":"Motorized Boating Accessory","VF":"Non-motorized Boating Accessory","WG":"Ice Fishing","XH":"Electronics","YJ":"Energy","ZK":"First Time Exhibitor Product"}

def get_gsheet_client():
    service_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info = json.loads(service_json)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    client = gspread.authorize(creds)
    return client, creds


@app.route('/export_gsheet')
def export_gsheet():
    gc, creds = get_gsheet_client()

    print("USING GOOGLE ACCOUNT:", creds.service_account_email)

    session_id = session.get("session_id")
    if not session_id:
        flash('Please log in or create a session first.')
        return redirect(url_for('login'))

    top3_per_category = get_top3_votes_by_category(session_id)

    spreadsheet_id = session.get('spreadsheet_id')
    if spreadsheet_id:
        try:
            spreadsheet = gc.open_by_key(spreadsheet_id)
            worksheet = spreadsheet.sheet1
            worksheet.clear()
        except Exception:
            spreadsheet = None
    else:
        spreadsheet = None

    if spreadsheet is None:
        spreadsheet_name = f"Top3Votes_Session_{session_id}"
        spreadsheet = gc.create(spreadsheet_name)
        spreadsheet.share(creds.service_account_email, perm_type="user", role="writer")
        spreadsheet.share("npsicast@gmail.com", perm_type="user", role="writer")
        spreadsheet.transfer_ownership("npsicast@gmail.com")
        worksheet = spreadsheet.sheet1
        worksheet.update_title("Top 3 Results")
        session['spreadsheet_id'] = spreadsheet.id

    header = [
        "Category Name", "Category ID",
        "1st Place ID", "1st Votes",
        "2nd Place ID", "2nd Votes",
        "3rd Place ID", "3rd Votes"
    ]
    worksheet.append_row(header)

    for category_id, top_votes in top3_per_category.items():
        product_name = category_to_name.get(category_id, "Unknown Category")
        row = [product_name, category_id]
        for i in range(3):
            if i < len(top_votes):
                row.extend([
                    top_votes[i].get("product_number", ""),
                    top_votes[i].get("count", 0)
                ])
            else:
                row.extend(["", ""])
        worksheet.append_row(row)

    sheet_url = spreadsheet.url
    flash(Markup(f"Google Sheet created/updated: <a href='{sheet_url}' target='_blank'>{sheet_url}</a>"))
    return redirect(url_for('dashboard'))


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
