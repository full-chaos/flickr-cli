import os
import sys
import webbrowser
import time
import io
import requests
import click
from collections import defaultdict
from requests_oauthlib import OAuth1Session

# Optional/fuzzy/AI
try:
    from rapidfuzz import fuzz
    from PIL import Image
    from tqdm import tqdm
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError:
    pass

FLICKR_API_KEY = os.getenv('FLICKR_API_KEY')
FLICKR_API_SECRET = os.getenv('FLICKR_API_SECRET')
FLICKR_BASE = "https://api.flickr.com/services/rest"

@click.group()
def cli():
    """Flickr CLI Tool (dedupe, upload, update, etc)"""
    pass

def get_oauth_session():
    oauth = OAuth1Session(FLICKR_API_KEY, client_secret=FLICKR_API_SECRET, callback_uri='oob')
    request_token_url = "https://www.flickr.com/services/oauth/request_token"
    fetch_response = oauth.fetch_request_token(request_token_url)
    resource_owner_key = fetch_response.get('oauth_token')
    resource_owner_secret = fetch_response.get('oauth_token_secret')

    authorization_url = oauth.authorization_url("https://www.flickr.com/services/oauth/authorize")
    print(f"Go here and authorize: {authorization_url}")
    webbrowser.open(authorization_url)
    verifier = input("Paste the verifier code: ")

    oauth = OAuth1Session(
        FLICKR_API_KEY,
        client_secret=FLICKR_API_SECRET,
        resource_owner_key=resource_owner_key,
        resource_owner_secret=resource_owner_secret,
        verifier=verifier,
    )
    access_token_url = "https://www.flickr.com/services/oauth/access_token"
    oauth_tokens = oauth.fetch_access_token(access_token_url)
    return oauth_tokens

def get_user_id(oauth):
    user_info = oauth.get(FLICKR_BASE, params={
        'method': 'flickr.test.login',
        'api_key': FLICKR_API_KEY,
        'format': 'json',
        'nojsoncallback': 1,
    }).json()
    return user_info['user']['id']

def fetch_all_photos(oauth, user_id):
    per_page = 500
    page = 1
    all_photos = []
    while True:
        params = {
            'method': 'flickr.people.getPhotos',
            'api_key': FLICKR_API_KEY,
            'user_id': user_id,
            'format': 'json',
            'nojsoncallback': 1,
            'per_page': per_page,
            'page': page,
            'extras': 'date_taken,original_format,url_m'
        }
        r = oauth.get(FLICKR_BASE, params=params)
        r.raise_for_status()
        data = r.json()
        photos = data['photos']['photo']
        all_photos.extend(photos)
        print(f"Fetched {len(photos)} photos on page {page}")
        if page >= data['photos']['pages']:
            break
        page += 1
        time.sleep(1)
    return all_photos

@cli.command()
def auth():
    """Authenticate with Flickr and print/store tokens."""
    tokens = get_oauth_session()
    print("Your OAuth tokens:", tokens)
    # Save to disk/Keychain for later (not shown here)

@cli.command()
@click.option('--by', type=click.Choice(['title', 'filename', 'datetaken'], case_sensitive=False), multiple=True, default=['title', 'filename'])
def scan(by):
    """Scan Flickr for duplicate photos (by title, filename, date taken)."""
    tokens = get_oauth_session()
    oauth = OAuth1Session(
        FLICKR_API_KEY,
        client_secret=FLICKR_API_SECRET,
        resource_owner_key=tokens['oauth_token'],
        resource_owner_secret=tokens['oauth_token_secret']
    )
    user_id = get_user_id(oauth)
    photos = fetch_all_photos(oauth, user_id)
    print(f"Total photos fetched: {len(photos)}")

    by_title = defaultdict(list)
    by_filename = defaultdict(list)
    by_datetaken = defaultdict(list)
    for p in photos:
        by_title[p['title']].append(p)
        fname = p.get('originalformat') or ''
        by_filename[fname].append(p)
        by_datetaken[p['datetaken'][:10]].append(p)  # By date (YYYY-MM-DD)
    
    if 'title' in by:
        print("\nDuplicates by title:")
        for k, v in by_title.items():
            if k and len(v) > 1:
                print(f"\nTitle: {k} ({len(v)} photos)")
                for p in v:
                    print(f"  - ID: {p['id']} | Date: {p.get('datetaken','?')}")
    if 'filename' in by:
        print("\nDuplicates by filename:")
        for k, v in by_filename.items():
            if k and len(v) > 1:
                print(f"\nFilename: {k} ({len(v)} photos)")
                for p in v:
                    print(f"  - ID: {p['id']} | Title: {p['title']} | Date: {p.get('datetaken','?')}")
    if 'datetaken' in by:
        print("\nDuplicates by date taken:")
        for k, v in by_datetaken.items():
            if k and len(v) > 1:
                print(f"\nDate: {k} ({len(v)} photos)")
                for p in v:
                    print(f"  - ID: {p['id']} | Title: {p['title']}")

@cli.command()
@click.option('--threshold', default=85, help="Fuzzy match threshold (0-100)")
def fuzzy_scan(threshold):
    """Fuzzy duplicate scan for photo titles using RapidFuzz."""
    try:
        from rapidfuzz import fuzz
    except ImportError:
        print("RapidFuzz is required for fuzzy scanning.")
        sys.exit(1)
    tokens = get_oauth_session()
    oauth = OAuth1Session(
        FLICKR_API_KEY,
        client_secret=FLICKR_API_SECRET,
        resource_owner_key=tokens['oauth_token'],
        resource_owner_secret=tokens['oauth_token_secret']
    )
    user_id = get_user_id(oauth)
    photos = fetch_all_photos(oauth, user_id)
    print(f"Fuzzy matching {len(photos)} titles...")
    import itertools
    pairs = []
    for a, b in itertools.combinations(photos, 2):
        score = fuzz.ratio(a['title'], b['title'])
        if score >= threshold:
            pairs.append((a, b, score))
    if not pairs:
        print("No fuzzy duplicate titles found.")
    else:
        for a, b, score in pairs:
            print(f"({score}%) '{a['title']}' [ID:{a['id']}] <--> '{b['title']}' [ID:{b['id']}]")

@cli.command()
@click.option('--similarity-threshold', default=0.95, help="Cosine similarity threshold (0-1)")
@click.option('--max-images', default=100, help="Max images to scan for AI dedupe (for speed)")
def ai_scan(similarity_threshold, max_images):
    """AI-based visual duplicate scan with CLIP embeddings."""
    try:
        from PIL import Image
        import torch
        from tqdm import tqdm
        from transformers import CLIPProcessor, CLIPModel
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError:
        print("Install torch, transformers, pillow, tqdm, scikit-learn for AI scanning.")
        sys.exit(1)
    tokens = get_oauth_session()
    oauth = OAuth1Session(
        FLICKR_API_KEY,
        client_secret=FLICKR_API_SECRET,
        resource_owner_key=tokens['oauth_token'],
        resource_owner_secret=tokens['oauth_token_secret']
    )
    user_id = get_user_id(oauth)
    photos = fetch_all_photos(oauth, user_id)
    photos = [p for p in photos if 'url_m' in p][:max_images]
    print(f"Processing {len(photos)} images for AI similarity (CLIP)...")

    images = []
    for p in tqdm(photos, desc="Downloading thumbnails"):
        try:
            resp = requests.get(p['url_m'], timeout=8)
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Could not download image {p['id']}: {e}")
            images.append(None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    embeddings = []
    for img in tqdm(images, desc="Embedding images with CLIP"):
        if img is None:
            embeddings.append(np.zeros((512,)))
            continue
        inputs = processor(images=img, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
        emb = emb.cpu().numpy().flatten()
        embeddings.append(emb / np.linalg.norm(emb))
    embeddings = np.array(embeddings)
    similarities = cosine_similarity(embeddings)
    n = len(photos)
    reported = set()
    for i in range(n):
        for j in range(i+1, n):
            if similarities[i, j] > similarity_threshold:
                key = tuple(sorted([photos[i]['id'], photos[j]['id']]))
                if key in reported:
                    continue
                print(
                    f"[{similarities[i, j]:.2f}] AI match: "
                    f"{photos[i]['title']} (ID:{photos[i]['id']}) <-> "
                    f"{photos[j]['title']} (ID:{photos[j]['id']})"
                )
                reported.add(key)

if __name__ == "__main__":
    cli()