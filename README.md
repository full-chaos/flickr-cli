# flickr-cli

**A Python CLI for syncing your Flickr library and deduplicating images using AI or perceptual hash.**

---

## 🚀 Features

* **Sync your Flickr photo library** to a local folder (skips already-downloaded images)
* **AI-powered deduplication** of local images:

  * **CNN (MobileNetV3):** Fast, robust detection of visually similar/near-duplicate photos
  * **ONNX CLIP:** (Optional) For semantic/visual dedupe using CLIP Vision Transformer
* **Text and metadata-based duplicate scans** (by title, filename, or date)
* **Fuzzy matching for titles**
* Designed for performance with large libraries (thousands of photos)

---

## 📦 Requirements

* Python 3.8 or newer
* Flickr API key & secret ([get yours here](https://www.flickr.com/services/api/misc.api_keys.html))

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔐 Setup

1. **Set your API credentials** (in your shell)**:

   ```bash
   export FLICKR_API_KEY=your_api_key
   export FLICKR_API_SECRET=your_api_secret
   ```

2. **(First time) Authenticate via CLI:**

   ```bash
   python cli.py auth
   ```

---

## 🏃‍♂️ Usage

### 1. **Sync your Flickr photos to a local directory**

```bash
python cli.py sync_flickr --directory /path/to/myphotos --max-images 5000
```

* Downloads all (or up to `--max-images`) Flickr photos to the directory, skipping images that are already present.

---

### 2. **Deduplicate your local photo folder with AI**

```bash
python cli.py ai_dedupe --directory /path/to/myphotos --method cnn --max-images 1000 --similarity-threshold 0.97
```

* `--method cnn` (default): Fast, robust deep-learning dedupe (MobileNetV3)
* `--method onnx`: (Optional) Uses ONNX CLIP Vision Transformer (requires model)
* Results: prints pairs of duplicates with similarity scores

---

### 3. **Text/metadata-based duplicate scan**

```bash
python cli.py scan
python cli.py fuzzy_scan --threshold 90
```

* Finds duplicates by title, filename, or date
* Fuzzy scan uses RapidFuzz for inexact title matches

---

## 🧠 Tips

* Use `sync_flickr` as your main fetcher—run dedupe many times offline!
* For existing archives, skip sync and just point dedupe at your folder.
* The CLI skips already-downloaded images, so you can resume sync at any time.

---

## 🧩 Extending

* Add custom dedupe actions (move, delete, tag)
* Visualize duplicate clusters (see [imagededup docs](https://github.com/idealo/imagededup))
* Integrate recursive search or batch reporting

---

## 📜 License

MIT

---
