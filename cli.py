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
import keyring
import json

# CoreML imports
try:
    import coremltools as ct
    import coremltools.models.datatypes as dt

    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    print("Warning: CoreML not available. Install coremltools for faster inference.")


FLICKR_API_KEY = os.getenv("FLICKR_API_KEY")
FLICKR_API_SECRET = os.getenv("FLICKR_API_SECRET")
FLICKR_BASE = "https://api.flickr.com/services/rest"

# === ONNX CLIP SETUP ===
onnx_model_paths = [
    Path("cache/clip/ViT-B-32__openai/visual/model.onnx"),
    Path("output_onnx_clip/model.onnx"),
    Path("models/clip_vit_b32.onnx"),
]

ort_session = None
processor = None

for onnx_path in onnx_model_paths:
    if onnx_path.exists():
        try:
            ort_session = ort.InferenceSession(str(onnx_path))
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            print(f"ONNX CLIP model loaded successfully from {onnx_path}")
            break
        except Exception as e:
            print(f"Failed to load ONNX model from {onnx_path}: {e}")
            continue

if ort_session is None:
    print("ONNX model not found - ONNX method will be unavailable")

cnn = CNN()

# === COREML CLIP SETUP ===
coreml_model = None
# Look for CoreML models in multiple locations
coreml_model_paths = [
    Path("cache/clip/ViT-B-32__openai/visual/model.mlmodel"),
    Path("models/clip_vit_b32.mlmodel"),
    Path("output_onnx_clip/model.mlmodel"),
]

if COREML_AVAILABLE:
    import coremltools as ct

    for coreml_path in coreml_model_paths:
        if coreml_path.exists():
            try:
                coreml_model = ct.models.MLModel(str(coreml_path))
                print(f"CoreML model loaded successfully from {coreml_path}")
                break
            except Exception as e:
                print(f"Failed to load CoreML model from {coreml_path}: {e}")
                continue

    if coreml_model is None:
        print(
            "CoreML available but no model found. Use 'convert-to-coreml' command to create one."
        )

# === IMMICH MODEL SETUP ===
immich_model = None
immich_model_path = Path(".")  # Look in current directory

MAX_IMAGES = None  # Default to no limit


def get_onnx_clip_embedding(image: Image.Image):
    if ort_session is None or processor is None:
        raise RuntimeError("ONNX CLIP model not available. Use a different method.")
    inputs = processor(images=image, return_tensors="np")
    ort_inputs = {"pixel_values": inputs["pixel_values"].astype(np.float32)}
    outputs = ort_session.run(None, ort_inputs)
    emb = outputs[0][0]
    return emb / np.linalg.norm(emb)


def get_coreml_clip_embedding(image: Image.Image):
    """Get CLIP embedding using CoreML for faster inference."""
    if not COREML_AVAILABLE or coreml_model is None:
        raise RuntimeError("CoreML model not available. Use ONNX method instead.")

    # Preprocess image using CLIP processor
    inputs = processor(images=image, return_tensors="np")
    pixel_values = inputs["pixel_values"].astype(np.float32)

    # Run CoreML inference
    outputs = coreml_model.predict({"pixel_values": pixel_values})

    # Extract embedding (key name may vary depending on conversion)
    # Common output names: "last_hidden_state", "pooler_output", or "output"
    if "pooler_output" in outputs:
        emb = outputs["pooler_output"][0]
    elif "last_hidden_state" in outputs:
        emb = outputs["last_hidden_state"][0]
    elif "output" in outputs:
        emb = outputs["output"][0]
    else:
        # Fallback: use first output
        emb = list(outputs.values())[0][0]

    return emb / np.linalg.norm(emb)


# Alternative CoreML approach using Apple's pre-trained models
def get_apple_coreml_embedding(image: Image.Image):
    """Get embeddings using Apple's pre-trained CoreML models."""
    try:
        import coremltools as ct
        import numpy as np

        # Use Apple's pre-trained MobileNet or ResNet models
        # These are simpler but still effective for duplicate detection
        model_url = "https://ml-assets.apple.com/coreml/models/Image/ImageClassification/MobileNetV2/MobileNetV2.mlmodel"

        # For now, let's use a simple approach with PIL and numpy
        # Resize image to standard size
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized).astype(np.float32) / 255.0

        # Simple feature extraction using image statistics
        # This is a fallback when CoreML models don't work
        features = []

        # Color histogram features
        for channel in range(3):
            hist = np.histogram(img_array[:, :, channel], bins=32, range=(0, 1))[0]
            features.extend(hist)

        # Texture features (edge detection approximation)
        gray = np.mean(img_array, axis=2)
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        texture_features = [
            np.mean(np.abs(grad_x)),
            np.mean(np.abs(grad_y)),
            np.std(gray),
            np.mean(gray),
        ]
        features.extend(texture_features)

        # Normalize features
        features = np.array(features)
        return features / np.linalg.norm(features)

    except Exception as e:
        print(f"Apple CoreML embedding failed: {e}")
        # Fallback to ONNX method
        return get_onnx_clip_embedding(image)


def get_immich_clip_embedding(image: Image.Image):
    """Get CLIP embedding using Immich's models and approach."""
    global immich_model

    if immich_model is None:
        # Load using OpenCLIP with Immich's default model
        try:
            import open_clip

            # Use Immich's default model configuration: ViT-B-32__openai
            model_name = "ViT-B-32"
            pretrained = "openai"

            print(f"Loading Immich's default CLIP model: {model_name}__{pretrained}...")

            # Load model using OpenCLIP exactly like Immich does
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device="cpu",  # Start with CPU, can be optimized later
            )
            model.eval()

            immich_model = {
                "model": model,
                "preprocess": preprocess,
                "type": "openclip",
            }
            print("Successfully loaded Immich CLIP model!")

        except ImportError as e:
            print(f"OpenCLIP not available: {e}")
            print("Install with: pip install open-clip-torch")
            return get_onnx_clip_embedding(image)  # Final fallback to ONNX

        except Exception as e:
            print(f"Failed to load Immich models: {e}")
            print("Using ONNX fallback.")
            return get_onnx_clip_embedding(image)

        if immich_model is None:
            print("Failed to load any Immich models. Using ONNX fallback.")
            return get_onnx_clip_embedding(image)

    try:
        import torch
        import numpy as np

        # Check if using Immich's OpenCLIPEncoder
        if hasattr(immich_model, "encode"):
            # Using Immich's OpenCLIPEncoder
            embedding = immich_model.encode(image)
            # Convert to numpy if it's a tensor
            if hasattr(embedding, "cpu"):
                embedding = embedding.cpu().numpy()
            if embedding.ndim > 1:
                embedding = embedding.flatten()

        elif "model" in immich_model and immich_model.get("type") == "openclip":
            # Use OpenCLIP model directly
            model = immich_model["model"]
            preprocess = immich_model["preprocess"]

            # Preprocess and get embedding
            with torch.no_grad():
                image_tensor = preprocess(image).unsqueeze(0)
                embedding = model.encode_image(image_tensor)
                embedding = embedding.cpu().numpy()[0]

        elif "session" in immich_model:
            # Use ONNX model like Immich does
            # Apply Immich preprocessing (simplified version)
            preprocess_cfg = immich_model["preprocess"]
            size = preprocess_cfg["size"]
            if isinstance(size, list):
                size = size[0]

            # Simple preprocessing without Immich's transform modules
            processed_image = image.resize((size, size))
            image_np = np.array(processed_image).astype(np.float32) / 255.0

            mean = np.array(preprocess_cfg["mean"], dtype=np.float32)
            std = np.array(preprocess_cfg["std"], dtype=np.float32)

            # Normalize
            for i in range(3):
                image_np[:, :, i] = (image_np[:, :, i] - mean[i]) / std[i]

            # Format input for ONNX (CHW format)
            input_data = {"image": np.expand_dims(image_np.transpose(2, 0, 1), 0)}

            # Run inference
            outputs = immich_model["session"].run(None, input_data)
            embedding = outputs[0][0]
        else:
            raise ValueError("Unknown Immich model type")

        # Normalize embedding
        return embedding / np.linalg.norm(embedding)

    except Exception as e:
        print(f"Immich embedding failed: {e}")
        return get_onnx_clip_embedding(image)  # Fallback to ONNX


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


def load_tokens_from_keyring():
    """Load OAuth tokens from keyring."""

    data = keyring.get_password("flickr-cli", "oauth_tokens")
    if data:
        return json.loads(data)
    raise RuntimeError("No Flickr tokens found in keychain. Please run 'auth' first.")


@cli.command()
def auth():
    """Authenticate with Flickr and print/store tokens."""
    _, tokens = get_oauth_session()
    print("Your OAuth tokens:")
    for k, v in tokens.items():
        print(f"{k}: {v}")

    # Save to disk for later use
    # You can store these in a hidden file in the user's home directory
    token_path = Path.home() / ".flickr_tokens"
    try:
        with open(token_path, "w") as f:
            for k, v in tokens.items():
                f.write(f"{k}={v}\n")
        print(f"Tokens saved to {token_path}")
    except Exception as e:
        print(f"Could not save tokens to {token_path}: {e}")


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

    # Group photos by various criteria with progress bar
    for p in tqdm(photos, desc="Analyzing photos for duplicates"):
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
    # Use tqdm to show progress for fuzzy matching combinations
    total_combinations = len(photos) * (len(photos) - 1) // 2
    for a, b in tqdm(
        itertools.combinations(photos, 2),
        desc="Comparing photo titles",
        total=total_combinations,
    ):
        score = fuzz.ratio(a["title"], b["title"])
        if score >= threshold:
            pairs.append((a, b, score))
    if not pairs:
        print("No fuzzy duplicate titles found.")
    else:
        for a, b, score in pairs:
            print(
                f"({score}%) '{a['title']}' [ID:{a['id']}] <--> "
                f"'{b['title']}' [ID:{b['id']}]"
            )


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

    elif method == "immich":
        # Load images and compute embeddings using Immich models
        image_paths = list(Path(image_dir).glob("*"))
        images = []
        for p in tqdm(image_paths, desc="Loading images"):
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                img = None
            images.append(img)

        embeddings = []
        for img in tqdm(images, desc="Embed with Immich CLIP"):
            if img:
                embeddings.append(get_immich_clip_embedding(img))
            else:
                embeddings.append(np.zeros(512))
        embeddings = np.stack(embeddings)

        sims = cosine_similarity(embeddings)
        n = len(image_paths)
        seen = set()
        total_comparisons = n * (n - 1) // 2

        with tqdm(total=total_comparisons, desc="Finding similar pairs") as pbar:
            for i in range(n):
                for j in range(i + 1, n):
                    sim = sims[i, j]
                    pbar.update(1)
                    if sim >= similarity_threshold:
                        pair = tuple(sorted((image_paths[i].stem, image_paths[j].stem)))
                        if pair not in seen:
                            seen.add(pair)
                            print(f"[{sim:.2f}] ID {pair[0]} <-> ID {pair[1]}")

    elif method == "coreml":
        # Load images and compute embeddings using CoreML
        image_paths = list(Path(image_dir).glob("*"))
        images = []
        for p in tqdm(image_paths, desc="Loading images"):
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                img = None
            images.append(img)

        embeddings = []
        for img in tqdm(images, desc="Embed with CoreML CLIP"):
            if img:
                embeddings.append(get_coreml_clip_embedding(img))
            else:
                embeddings.append(np.zeros(512))
        embeddings = np.stack(embeddings)

        sims = cosine_similarity(embeddings)
        n = len(image_paths)
        seen = set()
        total_comparisons = n * (n - 1) // 2
        comparison_count = 0

        with tqdm(total=total_comparisons, desc="Finding similar pairs") as pbar:
            for i in range(n):
                for j in range(i + 1, n):
                    sim = sims[i, j]
                    comparison_count += 1
                    pbar.update(1)
                    if sim >= similarity_threshold:
                        pair = tuple(sorted((image_paths[i].stem, image_paths[j].stem)))
                        if pair not in seen:
                            seen.add(pair)
                            print(f"[{sim:.2f}] ID {pair[0]} <-> ID {pair[1]}")

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
        total_comparisons = n * (n - 1) // 2
        comparison_count = 0

        with tqdm(total=total_comparisons, desc="Finding similar pairs") as pbar:
            for i in range(n):
                for j in range(i + 1, n):
                    sim = sims[i, j]
                    comparison_count += 1
                    pbar.update(1)
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


@cli.command()
@click.option(
    "--method",
    type=click.Choice(["cnn", "onnx", "coreml", "immich"], case_sensitive=False),
    default="immich" if Path("cache").exists() or Path("models").exists() else "cnn",
    show_default=True,
    help="AI method to use for deduplication",
)
@click.option(
    "--max-images",
    default=-1,
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



@cli.command()
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing test images",
)
@click.option(
    "--num-images", default=10, help="Number of images to test (for speed comparison)"
)
def benchmark_methods(directory, num_images):
    """Benchmark different AI methods (CNN, ONNX, CoreML) for speed comparison."""
    import time

    # Get test images
    local_path = Path(directory)
    images = (
        list(local_path.rglob("*.jpg"))
        + list(local_path.rglob("*.jpeg"))
        + list(local_path.rglob("*.JPG"))
        + list(local_path.rglob("*.JPEG"))
    )

    if len(images) < num_images:
        print(f"Only found {len(images)} images, using all of them")
        num_images = len(images)

    images = images[:num_images]
    print(f"Benchmarking with {num_images} images...")

    # Load images once
    loaded_images = []
    for img_path in tqdm(images, desc="Loading images for benchmark"):
        try:
            img = Image.open(img_path).convert("RGB")
            loaded_images.append(img)
        except Exception:
            continue

    methods_to_test = []

    # Test CNN method
    methods_to_test.append(("CNN", "cnn", None))

    # Test ONNX method if available
    if ort_session:
        methods_to_test.append(("ONNX CLIP", "onnx", get_onnx_clip_embedding))

    # Test CoreML method if available
    if COREML_AVAILABLE and coreml_model:
        methods_to_test.append(("CoreML CLIP", "coreml", get_coreml_clip_embedding))

    # Test Immich method if available
    if Path("cache").exists() or Path("models").exists():
        methods_to_test.append(("Immich CLIP", "immich", get_immich_clip_embedding))

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)

    for method_name, method_key, embedding_func in methods_to_test:
        print(f"\nTesting {method_name}...")

        start_time = time.time()

        if method_key == "cnn":
            # For CNN, we need to save images temporarily
            temp_dir = Path("/tmp/benchmark_cnn")
            temp_dir.mkdir(exist_ok=True, parents=True)

            for i, img in enumerate(loaded_images):
                img.save(temp_dir / f"img_{i}.jpg")

            encodings = cnn.encode_images(image_dir=str(temp_dir), recursive=False)

            # Cleanup
            import shutil

            shutil.rmtree(temp_dir)

        else:
            # For CLIP methods, compute embeddings directly
            embeddings = []
            for img in loaded_images:
                emb = embedding_func(img)
                embeddings.append(emb)

        end_time = time.time()
        total_time = end_time - start_time
        per_image_time = total_time / len(loaded_images)

        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Per image: {per_image_time:.3f} seconds")
        print(f"  Images/sec: {1 / per_image_time:.1f}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    cli()
