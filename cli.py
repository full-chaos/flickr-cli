#!/usr/bin/env python3
import os
import sys
import io
import time
import requests
import click
from collections import defaultdict
from pathlib import Path
import keyring
import json

# Heavy imports that cause warnings - load only when needed
_heavy_imports_loaded = False
_coreml_available = None
_onnx_available = None
_sklearn_available = None
_imagededup_available = None


def load_heavy_imports():
    """Load heavy ML libraries only when needed to avoid startup warnings."""
    global _heavy_imports_loaded, _coreml_available, _onnx_available
    global _sklearn_available, _imagededup_available
    global Image, np, tqdm, ort, CLIPProcessor, OAuth1Session
    global cosine_similarity, CNN, COREML_AVAILABLE

    if _heavy_imports_loaded:
        return

    try:
        from PIL import Image
        import numpy as np
        from tqdm import tqdm
        from requests_oauthlib import OAuth1Session
    except ImportError as e:
        print(f"Error importing basic dependencies: {e}")
        sys.exit(1)

    # Try ONNX imports
    try:
        import onnxruntime as ort
        from transformers import CLIPProcessor

        _onnx_available = True
    except ImportError:
        _onnx_available = False
        ort = None
        CLIPProcessor = None

    # Try sklearn imports
    try:
        from sklearn.metrics.pairwise import cosine_similarity

        _sklearn_available = True
    except ImportError:
        _sklearn_available = False
        cosine_similarity = None

    # Try imagededup imports
    try:
        from imagededup.methods import CNN

        _imagededup_available = True
    except ImportError:
        _imagededup_available = False
        CNN = None

    # Try CoreML imports (suppress warnings)
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import coremltools as ct
            import coremltools.models.datatypes as dt
        _coreml_available = True
        COREML_AVAILABLE = True
    except ImportError:
        _coreml_available = False
        COREML_AVAILABLE = False

    _heavy_imports_loaded = True


# Initialize these as None - they'll be loaded when needed
Image = None
np = None
tqdm = None
ort = None
CLIPProcessor = None
OAuth1Session = None
cosine_similarity = None
CNN = None
COREML_AVAILABLE = False


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


def initialize_onnx():
    """Initialize ONNX model if available."""
    global ort_session, processor

    if not _onnx_available:
        return False

    for onnx_path in onnx_model_paths:
        if onnx_path.exists():
            try:
                ort_session = ort.InferenceSession(str(onnx_path))
                processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch16"
                )
                print(f"ONNX CLIP model loaded successfully from {onnx_path}")
                return True
            except Exception as e:
                print(f"Failed to load ONNX model from {onnx_path}: {e}")
                continue

    print("ONNX model not found - will download when needed")
    return False


def initialize_cnn():
    """Initialize CNN model if available."""
    global cnn
    if _imagededup_available and CNN:
        cnn = CNN()
        return True
    return False


# === COREML CLIP SETUP ===
coreml_model = None
# Look for CoreML models in multiple locations
coreml_model_paths = [
    Path("cache/clip/ViT-B-32__openai/visual/model.mlmodel"),
    Path("models/clip_vit_b32.mlmodel"),
    Path("output_onnx_clip/model.mlmodel"),
]


def initialize_coreml():
    """Initialize CoreML model if available."""
    global coreml_model

    if not _coreml_available:
        return False

    import coremltools as ct

    for coreml_path in coreml_model_paths:
        if coreml_path.exists():
            try:
                coreml_model = ct.models.MLModel(str(coreml_path))
                print(f"CoreML model loaded successfully from {coreml_path}")
                return True
            except Exception as e:
                print(f"Failed to load CoreML model from {coreml_path}: {e}")
                continue

    print(
        "CoreML available but no model found. Use 'convert-to-coreml' command to create one."
    )
    return False


# === IMMICH MODEL SETUP ===
immich_model = None
immich_model_path = Path(".")  # Look in current directory

MAX_IMAGES = None  # Default to no limit


def download_clip_onnx_model():
    """Download CLIP ONNX model if not present."""
    import urllib.request
    import zipfile
    import tempfile
    import shutil

    # Check if model already exists
    model_paths = [
        "output_onnx_clip/model.onnx",
        "cache/model.onnx",
        "models/model.onnx",
    ]

    for path in model_paths:
        if os.path.exists(path):
            print(f"ONNX model found at {path}")
            return path

    print("Downloading CLIP ONNX model...")
    # Use Hugging Face's ONNX model endpoint
    base_url = "https://huggingface.co/openai/clip-vit-base-patch16/resolve/main/"
    model_files = [
        "model.onnx",
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "special_tokens_map.json",
    ]

    # Create output directory
    output_dir = "cache"
    os.makedirs(output_dir, exist_ok=True)

    try:
        for filename in model_files:
            url = base_url + filename
            output_path = os.path.join(output_dir, filename)
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, output_path)

        print(f"ONNX model downloaded to {output_dir}/")
        return os.path.join(output_dir, "model.onnx")
    except Exception as e:
        print(f"Failed to download ONNX model: {e}")
        return None


def download_coreml_model():
    """Download or convert CoreML model if not present."""
    model_paths = [
        "cache/clip_model.mlmodel",
        "models/clip_model.mlmodel",
        "cache/clip.mlmodel",
        "models/clip.mlmodel",
    ]

    for path in model_paths:
        if os.path.exists(path):
            print(f"CoreML model found at {path}")
            return path

    print("CoreML model not found. Converting from ONNX...")

    # First ensure we have the ONNX model
    onnx_path = download_clip_onnx_model()
    if not onnx_path:
        print("Cannot convert to CoreML without ONNX model")
        return None

    try:
        import coremltools as ct

        output_dir = "cache"
        os.makedirs(output_dir, exist_ok=True)

        # Convert ONNX to CoreML
        print("Converting ONNX model to CoreML...")
        coreml_model = ct.convert(
            onnx_path, source="onnx", compute_units=ct.ComputeUnit.ALL
        )

        output_path = os.path.join(output_dir, "clip_model.mlmodel")
        coreml_model.save(output_path)
        print(f"CoreML model saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Failed to convert to CoreML: {e}")
        return None


def check_model_requirements():
    """Check what model files are available and suggest downloads."""
    available_methods = []

    # Check Immich/OpenCLIP (auto-downloads)
    try:
        import open_clip

        available_methods.append("immich")
    except ImportError:
        pass

    # Check CNN (imagededup - no model files needed)
    try:
        from imagededup.methods import CNN

        available_methods.append("cnn")
    except ImportError:
        pass

    # Check ONNX
    onnx_paths = [
        "output_onnx_clip/model.onnx",
        "cache/model.onnx",
        "models/model.onnx",
    ]
    if any(os.path.exists(p) for p in onnx_paths):
        available_methods.append("onnx")

    # Check CoreML
    coreml_paths = [
        "cache/clip_model.mlmodel",
        "models/clip_model.mlmodel",
        "cache/clip.mlmodel",
        "models/clip.mlmodel",
    ]
    if any(os.path.exists(p) for p in coreml_paths) and COREML_AVAILABLE:
        available_methods.append("coreml")

    return available_methods


def get_onnx_clip_embedding(image):
    """Get CLIP embedding using ONNX model."""
    load_heavy_imports()
    global ort_session, processor

    if ort_session is None or processor is None:
        # Try to download and load the model
        print("Loading ONNX model...")
        onnx_path = download_clip_onnx_model()
        if onnx_path:
            try:
                ort_session = ort.InferenceSession(onnx_path)
                # Try to load processor from local files first
                config_dir = os.path.dirname(onnx_path)
                try:
                    processor = CLIPProcessor.from_pretrained(config_dir)
                except Exception:
                    # Fallback to online model
                    processor = CLIPProcessor.from_pretrained(
                        "openai/clip-vit-base-patch16"
                    )
                print("ONNX model loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load ONNX model: {e}")
        else:
            raise RuntimeError("ONNX model download failed")

    inputs = processor(images=image, return_tensors="np")
    ort_inputs = {"pixel_values": inputs["pixel_values"].astype(np.float32)}
    outputs = ort_session.run(None, ort_inputs)
    emb = outputs[0][0]
    return emb / np.linalg.norm(emb)


def get_coreml_clip_embedding(image):
    """Get CLIP embedding using CoreML for faster inference."""
    load_heavy_imports()
    global coreml_model, processor

    if not _coreml_available:
        raise RuntimeError("CoreML not available. Use ONNX method instead.")

    if coreml_model is None:
        # Try to download and load the model
        print("Loading CoreML model...")
        coreml_path = download_coreml_model()
        if coreml_path:
            try:
                import coremltools as ct

                coreml_model = ct.models.MLModel(coreml_path)
                print("CoreML model loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load CoreML model: {e}")
        else:
            raise RuntimeError("CoreML model download/conversion failed")

    # Ensure we have the processor
    if processor is None:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

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
def get_apple_coreml_embedding(image):
    """Get embeddings using Apple's pre-trained CoreML models."""
    load_heavy_imports()

    try:
        import coremltools as ct
        import numpy as np

        # Use Apple's pre-trained MobileNet or ResNet models
        # These are simpler but still effective for duplicate detection
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


def get_immich_clip_embedding(image):
    """Get CLIP embedding using Immich's models and approach."""
    load_heavy_imports()
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
    load_heavy_imports()

    if method == "cnn":
        if not _imagededup_available:
            raise RuntimeError(
                "imagededup not available. Install with: pip install imagededup"
            )

        # Initialize CNN if not already done
        if "cnn" not in globals() or cnn is None:
            initialize_cnn()

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

    elif method in ["immich", "coreml", "onnx"]:
        if not _sklearn_available:
            raise RuntimeError(
                "scikit-learn not available. Install with: pip install scikit-learn"
            )

        # Load images and compute embeddings using the selected method
        image_paths = list(Path(image_dir).glob("*"))
        images = []
        for p in tqdm(image_paths, desc="Loading images"):
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                img = None
            images.append(img)

        embeddings = []
        embedding_func = {
            "immich": get_immich_clip_embedding,
            "coreml": get_coreml_clip_embedding,
            "onnx": get_onnx_clip_embedding,
        }[method]

        for img in tqdm(images, desc=f"Embed with {method.upper()} CLIP"):
            if img:
                embeddings.append(embedding_func(img))
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


def get_default_method():
    """Get the best available dedupe method as default."""
    available = check_model_requirements()

    # Prefer order: immich (best quality), onnx (good balance), coreml (fast), cnn (fallback)
    if "immich" in available:
        return "immich"
    elif "onnx" in available:
        return "onnx"
    elif "coreml" in available:
        return "coreml"
    elif "cnn" in available:
        return "cnn"
    else:
        # Default to immich - it will auto-download models
        return "immich"


@cli.command()
@click.option(
    "--method",
    type=click.Choice(["cnn", "onnx", "coreml", "immich"], case_sensitive=False),
    default=None,  # Will be set dynamically
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
    Uses the selected AI method (cnn, onnx, coreml, or immich) for deduplication.
    """

    # Set default method if none specified
    if method is None:
        method = get_default_method()
        print(f"Auto-selected method: {method}")

    image_paths, image_dir = collect_images(
        source="local", max_images=max_images, local_dir=directory
    )
    find_ai_duplicates(
        method=method,
        image_dir=image_dir,
        similarity_threshold=similarity_threshold,
    )


@cli.command()
def convert_to_coreml():
    """Convert the ONNX CLIP model to CoreML format for faster inference on macOS."""
    global coreml_model

    if not COREML_AVAILABLE:
        print("Error: coremltools not available. Install it with:")
        print("pip install coremltools")
        return

    import coremltools as ct

    # Try multiple ONNX source locations
    onnx_paths = [
        Path("cache/clip/ViT-B-32__openai/visual/model.onnx"),
        Path("models/clip_vit_b32.onnx"),
        Path("output_onnx_clip/model.onnx"),
    ]

    # Default output path in simplified structure
    coreml_path = Path("cache/clip/ViT-B-32__openai/visual/model.mlmodel")

    # Find available ONNX model
    onnx_path = None
    for path in onnx_paths:
        if path.exists():
            onnx_path = path
            break

    if onnx_path is None:
        print("Error: No ONNX model found in:")
        for path in onnx_paths:
            print(f"  {path}")
        print("Please ensure you have a CLIP ONNX model available.")
        return

    # Create output directory if needed
    coreml_path.parent.mkdir(parents=True, exist_ok=True)

    if coreml_path.exists():
        overwrite = click.confirm(
            f"CoreML model already exists at {coreml_path}. Overwrite?"
        )
        if not overwrite:
            return

    print(f"Converting {onnx_path} to CoreML...")
    print("This may take a few minutes...")

    try:
        # Try basic conversion first
        print("Attempting basic ONNX to CoreML conversion...")
        model = ct.converters.onnx.convert(str(onnx_path))

        # Save the CoreML model
        model.save(str(coreml_path))
        print(f"Successfully converted to CoreML: {coreml_path}")
        print("You can now use --method=coreml for faster inference!")

    except Exception as e:
        print(f"Direct ONNX conversion failed: {e}")
        print("\nTrying alternative approach with PyTorch conversion...")

        try:
            # Alternative: Load with transformers and convert via PyTorch
            from transformers import CLIPVisionModel
            import torch

            # Load the PyTorch model
            model_name = "openai/clip-vit-base-patch16"
            vision_model = CLIPVisionModel.from_pretrained(model_name)
            vision_model.eval()

            # Create dummy input
            dummy_input = torch.randn(1, 3, 224, 224)

            # Convert to CoreML via PyTorch
            traced_model = torch.jit.trace(vision_model, dummy_input)
            coreml_model_converted = ct.convert(
                traced_model, inputs=[ct.TensorType(shape=(1, 3, 224, 224))]
            )

            coreml_model_converted.save(str(coreml_path))
            print(f"Successfully converted via PyTorch: {coreml_path}")

        except Exception as e2:
            print(f"PyTorch conversion also failed: {e2}")
            print("\nPlease check:")
            print("1. CoreML Tools installation: 'pip install --upgrade coremltools'")
            print("2. PyTorch installation: 'pip install torch'")
            print("3. ONNX model validity")
            return

    # Reload the global CoreML model
    try:
        coreml_model = ct.models.MLModel(str(coreml_path))
        print("CoreML model loaded and ready to use!")
    except Exception as e:
        print(f"Warning: Could not load converted model: {e}")


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
