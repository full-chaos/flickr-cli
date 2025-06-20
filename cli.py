#!/usr/bin/env python3
import os
import sys
import io
import time
import requests
import click
from collections import defaultdict
from PIL import Image
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
from transformers import CLIPProcessor
from requests_oauthlib import OAuth1Session
from sklearn.metrics.pairwise import cosine_similarity
from imagededup.methods import CNN
from pathlib import Path

FLICKR_API_KEY = os.getenv("FLICKR_API_KEY")
FLICKR_API_SECRET = os.getenv("FLICKR_API_SECRET")
FLICKR_BASE = "https://api.flickr.com/services/rest"

# === ONNX CLIP SETUP ===
ort_session = ort.InferenceSession("output_onnx_clip/model.onnx")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
cnn = CNN()


def get_onnx_clip_embedding(image: Image.Image):
    inputs = processor(images=image, return_tensors="np")
    ort_inputs = {"pixel_values": inputs["pixel_values"].astype(np.float32)}
    outputs = ort_session.run(None, ort_inputs)
    emb = outputs[0][0]
    return emb / np.linalg.norm(emb)


@click.group()
def cli():
    """Flickr CLI Tool (dedupe, upload, update, etc)"""
    pass


def get_oauth_session():
    oauth = OAuth1Session(
        FLICKR_API_KEY, client_secret=FLICKR_API_SECRET, callback_uri="oob"
    )
    req_tok = oauth.fetch_request_token(
        "https://www.flickr.com/services/oauth/request_token"
    )
    auth_url = oauth.authorization_url(
        "https://www.flickr.com/services/oauth/authorize"
    )
    print("Authorize this app by visiting:\n", auth_url)
    webbrowser = __import__("webbrowser")
    webbrowser.open(auth_url)
    verifier = input("Verifier code: ")
    oauth = OAuth1Session(
        FLICKR_API_KEY,
        client_secret=FLICKR_API_SECRET,
        resource_owner_key=req_tok["oauth_token"],
        resource_owner_secret=req_tok["oauth_token_secret"],
        verifier=verifier,
    )
    tok_response = oauth.fetch_access_token(
        "https://www.flickr.com/services/oauth/access_token"
    )
    return oauth, tok_response


def get_user_id(oauth):
    user_info = oauth.get(
        FLICKR_BASE,
        params={
            "method": "flickr.test.login",
            "api_key": FLICKR_API_KEY,
            "format": "json",
            "nojsoncallback": 1,
        },
    ).json()
    return user_info["user"]["id"]


def fetch_all_photos(oauth, user_id, max_images=None):
    per_page = 500
    page = 1
    all_photos = []
    while True:
        params = {
            "method": "flickr.people.getPhotos",
            "api_key": FLICKR_API_KEY,
            "user_id": user_id,
            "format": "json",
            "nojsoncallback": 1,
            "per_page": per_page,
            "page": page,
            "extras": "date_taken,original_format,url_m,url_l,url_s",
        }
        r = oauth.get(FLICKR_BASE, params=params)
        r.raise_for_status()
        data = r.json()
        photos = data["photos"]["photo"]
        all_photos.extend(photos)
        print(f"Fetched {len(photos)} photos on page {page}")
        if max_images is not None and len(all_photos) >= max_images:
            all_photos = all_photos[:max_images]
            break
        if page >= data["photos"]["pages"]:
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
@click.option(
    "--by",
    type=click.Choice(["title", "filename", "datetaken"], case_sensitive=False),
    multiple=True,
    default=["title", "filename"],
)
def scan(by):
    """Scan Flickr for duplicate photos (by title, filename, date taken)."""
    tokens = get_oauth_session()
    oauth = OAuth1Session(
        FLICKR_API_KEY,
        client_secret=FLICKR_API_SECRET,
        resource_owner_key=tokens["oauth_token"],
        resource_owner_secret=tokens["oauth_token_secret"],
    )
    user_id = get_user_id(oauth)
    photos = fetch_all_photos(oauth, user_id)
    print(f"Total photos fetched: {len(photos)}")

    by_title = defaultdict(list)
    by_filename = defaultdict(list)
    by_datetaken = defaultdict(list)
    for p in photos:
        by_title[p["title"]].append(p)
        fname = p.get("originalformat") or ""
        by_filename[fname].append(p)
        by_datetaken[p["datetaken"][:10]].append(p)  # By date (YYYY-MM-DD)

    if "title" in by:
        print("\nDuplicates by title:")
        for k, v in by_title.items():
            if k and len(v) > 1:
                print(f"\nTitle: {k} ({len(v)} photos)")
                for p in v:
                    print(f"  - ID: {p['id']} | Date: {p.get('datetaken', '?')}")
    if "filename" in by:
        print("\nDuplicates by filename:")
        for k, v in by_filename.items():
            if k and len(v) > 1:
                print(f"\nFilename: {k} ({len(v)} photos)")
                for p in v:
                    print(
                        f"  - ID: {p['id']} | Title: {p['title']} | Date: {p.get('datetaken', '?')}"
                    )
    if "datetaken" in by:
        print("\nDuplicates by date taken:")
        for k, v in by_datetaken.items():
            if k and len(v) > 1:
                print(f"\nDate: {k} ({len(v)} photos)")
                for p in v:
                    print(f"  - ID: {p['id']} | Title: {p['title']}")


@cli.command()
@click.option("--threshold", default=85, help="Fuzzy match threshold (0-100)")
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
        resource_owner_key=tokens["oauth_token"],
        resource_owner_secret=tokens["oauth_token_secret"],
    )
    user_id = get_user_id(oauth)
    photos = fetch_all_photos(oauth, user_id)
    print(f"Fuzzy matching {len(photos)} titles...")
    import itertools

    pairs = []
    for a, b in itertools.combinations(photos, 2):
        score = fuzz.ratio(a["title"], b["title"])
        if score >= threshold:
            pairs.append((a, b, score))
    if not pairs:
        print("No fuzzy duplicate titles found.")
    else:
        for a, b, score in pairs:
            print(
                f"({score}%) '{a['title']}' [ID:{a['id']}] <--> '{b['title']}' [ID:{b['id']}]"
            )


@cli.command()
@click.option("--max-images", default=None, help="Max images to scan")
@click.option(
    "--similarity-threshold", default=0.95, help="Cosine similarity threshold"
)
def ai_scan(max_images, similarity_threshold):
    """AI duplicate scan using ONNX CLIP embeddings."""
    oauth, _ = get_oauth_session()
    user_id = get_user_id(oauth)
    photos = fetch_all_photos(oauth, user_id, max_images=max_images)
    images = []
    for p in tqdm(photos, desc="Download images"):
        try:
            img = Image.open(
                io.BytesIO(requests.get(p["url_m"], timeout=10).content)
            ).convert("RGB")
        except Exception:
            img = None
        images.append(img)

    embeddings = []
    for img in tqdm(images, desc="Embed with ONNX CLIP"):
        embeddings.append(get_onnx_clip_embedding(img) if img else np.zeros(512))
    embeddings = np.stack(embeddings)

    sims = cosine_similarity(embeddings)
    n = len(photos)
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            sim = sims[i, j]
            if sim >= similarity_threshold:
                pair = tuple(sorted((photos[i]["id"], photos[j]["id"])))
                if pair not in seen:
                    seen.add(pair)
                    print(f"[{sim:.2f}] ID {photos[i]['id']} <-> ID {photos[j]['id']}")


@cli.command()
@click.option("--max-images", default=None, help="Max images to scan")
@click.option("--similarity-threshold", default=0.97, help="Min similarity threshold")
def ai_scan_cnn(max_images, similarity_threshold):
    """AI duplicate scan using imagededup's CNN encoder."""
    oauth, _ = get_oauth_session()
    user_id = get_user_id(oauth)
    photos = fetch_all_photos(oauth, user_id, max_images)

    # 1️⃣ Download photos and save to temp folder
    temp = Path("/tmp/flickr_ai")
    temp.mkdir(exist_ok=True, parents=True)
    for p in tqdm(photos, desc="Downloading photos"):
        img_path = temp / f"{p['id']}.jpg"
        if img_path.exists():
            continue  # skip download if file already exists
        url = p.get("url_m") or p.get("url_l") or p.get("url_s")
        if not url:
            continue  # skip if no image url is available
        try:
            img_path.write_bytes(requests.get(url, timeout=10).content)
        except Exception:
            continue  # optionally, log errors or add a counter for skipped photos

    # 2️⃣ Generate CNN encodings from downloaded images
    encodings = cnn.encode_images(image_dir=str(temp), recursive=False)

    # 3️⃣ Find near-duplicates
    dupes = cnn.find_duplicates(
        encoding_map=encodings,
        min_similarity_threshold=similarity_threshold,
        scores=True,
    )

    # 4️⃣ Report duplicates
    for img, sim_list in dupes.items():
        id1 = Path(img).stem
        for dup_path, score in sim_list:
            id2 = Path(dup_path).stem
            print(f"[{score:.2f}] ID {id1} <-> ID {id2}")


def collect_images(
    source, max_images=None, flickr_photos=None, local_dir=None, temp_dir=None
):
    """
    Collect images based on source.

    Returns (list_of_paths, directory_used)
    """
    if source == "flickr":
        if flickr_photos is None:
            raise ValueError("flickr_photos must be provided for source 'flickr'")
        temp = temp_dir or Path("/tmp/flickr_ai")
        temp.mkdir(exist_ok=True, parents=True)
        paths = []
        flickr_photos = flickr_photos[:max_images] if max_images else flickr_photos
        for p in tqdm(flickr_photos[:max_images], desc="Downloading photos"):
            img_path = temp / f"{p['id']}.jpg"
            if img_path.exists():
                paths.append(str(img_path))
                continue  # skip download if file already exists
            url = p.get("url_m") or p.get("url_l") or p.get("url_s")
            if not url:
                continue  # skip if no image url is available
            try:
                img_path.write_bytes(requests.get(url, timeout=10).content)
                paths.append(str(img_path))
            except Exception:
                continue  # optionally, log errors or add a counter for skipped photos
        return paths, temp

    elif source == "local":
        if local_dir is None:
            raise ValueError("local_dir must be provided for source 'local'")
        local_path = Path(local_dir)
        if not local_path.is_dir():
            raise ValueError(f"{local_dir} is not a valid directory")
        # Scan for JPEGs (case insensitive) up to max_images
        images = (
            list(local_path.rglob("*.jpg"))
            + list(local_path.rglob("*.jpeg"))
            + list(local_path.rglob("*.JPG"))
            + list(local_path.rglob("*.JPEG"))
        )
        images = images[:max_images]
        return [str(p) for p in images], local_path

    else:
        raise ValueError(f"Unknown source: {source}")


def find_ai_duplicates(method, image_dir, similarity_threshold):
    """
    Find AI duplicates using the specified method.

    image_dir: directory containing images
    similarity_threshold: min similarity threshold
    """
    if method == "cnn":
        encodings = cnn.encode_images(image_dir=str(image_dir), recursive=False)
        dupes = cnn.find_duplicates(
            encoding_map=encodings,
            min_similarity_threshold=similarity_threshold,
            scores=True,
        )
        for img, sim_list in dupes.items():
            id1 = Path(img).stem
            for dup_path, score in sim_list:
                id2 = Path(dup_path).stem
                print(f"[{score:.2f}] ID {id1} <-> ID {id2}")

    elif method == "onnx":
        # Load images and compute embeddings
        image_paths = list(Path(image_dir).glob("*"))
        images = []
        for p in tqdm(image_paths, desc="Loading images"):
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                img = None
            images.append(img)

        embeddings = []
        for img in tqdm(images, desc="Embed with ONNX CLIP"):
            embeddings.append(get_onnx_clip_embedding(img) if img else np.zeros(512))
        embeddings = np.stack(embeddings)

        sims = cosine_similarity(embeddings)
        n = len(image_paths)
        seen = set()
        for i in range(n):
            for j in range(i + 1, n):
                sim = sims[i, j]
                if sim >= similarity_threshold:
                    pair = tuple(sorted((image_paths[i].stem, image_paths[j].stem)))
                    if pair not in seen:
                        seen.add(pair)
                        print(f"[{sim:.2f}] ID {pair[0]} <-> ID {pair[1]}")

    else:
        raise ValueError(f"Unknown method: {method}")


@cli.command()
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory to save downloaded Flickr images",
)
@click.option(
    "--max-images",
    default=None,
    type=int,
    show_default=False,
    help="Maximum number of images to sync (default: all)",
)
def sync_flickr(directory, max_images):
    """Download all (or up to max-images) Flickr photos to a directory. Skips existing files."""
    oauth, _ = get_oauth_session()
    user_id = get_user_id(oauth)
    photos = fetch_all_photos(oauth, user_id, max_images=max_images)
    dir_path = Path(directory)
    print(f"Downloading {len(photos)} photos to {dir_path}")
    for p in tqdm(photos, desc="Downloading photos"):
        img_path = dir_path / f"{p['id']}.jpg"
        if img_path.exists():
            continue  # Skip if already exists
        url = p.get("url_m") or p.get("url_l") or p.get("url_s")
        if not url:
            print(f"Warning: No image URL for photo ID {p['id']}")
            continue
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            img_path.write_bytes(resp.content)
        except Exception as e:
            print(f"Error downloading ID {p['id']}: {e}")


# === REFACTORED: ai_dedupe (local only) ===
@cli.command()
@click.option(
    "--method",
    type=click.Choice(["cnn", "onnx"], case_sensitive=False),
    default="cnn",
    show_default=True,
    help="AI method to use for deduplication",
)
@click.option(
    "--max-images",
    default=None,
    show_default=True,
    help="Maximum number of images to process",
)
@click.option(
    "--similarity-threshold",
    default=0.95,
    show_default=True,
    help="Similarity threshold for duplicate detection",
)
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Local directory containing images to deduplicate",
)
def ai_dedupe(method, max_images, similarity_threshold, directory):
    """
    AI duplicate detection for local images in a directory.

    Scans JPEG images in the given directory (non-recursive), up to --max-images.
    Uses the selected AI method (cnn or onnx) for deduplication.
    """
    image_paths, image_dir = collect_images(
        source="local", max_images=max_images, local_dir=directory
    )
    find_ai_duplicates(
        method=method,
        image_dir=image_dir,
        similarity_threshold=similarity_threshold,
    )


if __name__ == "__main__":
    cli()
