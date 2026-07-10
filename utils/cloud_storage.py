"""
Cloud image storage via Cloudinary.
When CLOUDINARY_* env vars are set, images are uploaded and the URL returned.
When not set (local dev), returns None and the caller uses the local static path.
"""
import os

_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
_API_KEY    = os.getenv("CLOUDINARY_API_KEY")
_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

ENABLED = bool(_CLOUD_NAME and _API_KEY and _API_SECRET)

if ENABLED:
    print("[INFO] Cloud storage: Cloudinary enabled.")
else:
    print("[INFO] Cloud storage: disabled (using local static files).")


def upload_image(local_path, folder="acnevision"):
    if not ENABLED or not os.path.exists(local_path):
        return None
    try:
        import cloudinary
        import cloudinary.uploader
        cloudinary.config(
            cloud_name=_CLOUD_NAME,
            api_key=_API_KEY,
            api_secret=_API_SECRET,
        )
        result = cloudinary.uploader.upload(
            local_path, folder=folder, resource_type="image"
        )
        url = result.get("secure_url")
        print(f"[INFO] Uploaded to Cloudinary: {url}")
        return url
    except Exception as e:
        print(f"[WARNING] Cloudinary upload failed: {e}")
        return None
