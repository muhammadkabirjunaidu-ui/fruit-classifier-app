
import os
from pathlib import Path
from io import BytesIO
from contextlib import asynccontextmanager

import gdown

from fastai.vision.all import load_learner, PILImage

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route
from starlette.concurrency import run_in_threadpool


# ==========================================
# 1. PATHS AND CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "export.pkl"

MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://drive.google.com/uc?export=download&id=1bZQIlGTzTD-U8YESD1weqRnNQCH3cVvW"
)

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB


# ==========================================
# 2. MODEL DOWNLOAD AND VALIDATION
# ==========================================

def download_and_load_model():

    if not MODEL_URL:
        raise RuntimeError("MODEL_URL is not configured.")

    # Remove empty or obviously corrupted previous downloads.
    if MODEL_PATH.exists():

        with open(MODEL_PATH, "rb") as file:
            header = file.read(512).lstrip().lower()

        is_html = (
            header.startswith(b"<!doctype html")
            or header.startswith(b"<html")
            or header.startswith(b"<?xml")
        )

        if is_html or MODEL_PATH.stat().st_size < 1024:
            print("Invalid model detected. Removing old file.")
            MODEL_PATH.unlink()

    # Download the model if it doesn't exist.
    if not MODEL_PATH.exists():

        print("Downloading model from Google Drive...")

        downloaded = gdown.download(
            url=MODEL_URL,
            output=str(MODEL_PATH),
            quiet=False,
            fuzzy=True
        )

        if not downloaded or not MODEL_PATH.exists():
            raise RuntimeError(
                "Model download failed. Check your Google Drive link."
            )

    # Validate downloaded file.
    with open(MODEL_PATH, "rb") as file:
        header = file.read(512).lstrip().lower()

    if (
        header.startswith(b"<!doctype html")
        or header.startswith(b"<html")
        or header.startswith(b"<?xml")
    ):
        MODEL_PATH.unlink(missing_ok=True)

        raise RuntimeError(
            "Google Drive returned HTML instead of export.pkl. "
            "Check file permissions and download link."
        )

    if MODEL_PATH.stat().st_size < 1024:
        MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError("Downloaded model is too small.")

    print(f"Model file size: {MODEL_PATH.stat().st_size} bytes")

    # Load trained model.
    print("Loading trained AI model...")

    learner = load_learner(MODEL_PATH)

    print("AI MODEL LOADED SUCCESSFULLY!")

    return learner


# ==========================================
# 3. APPLICATION STARTUP
# ==========================================

@asynccontextmanager
async def lifespan(app):

    print("Starting AI Image Classifier...")

    # Download and load model without blocking the event loop.
    app.state.learn = await run_in_threadpool(
        download_and_load_model
    )

    print("Application is ready!")

    yield

    print("Application shutting down.")


# ==========================================
# 4. HOMEPAGE
# ==========================================

async def homepage(request):

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Image Classifier</title>

        <style>
            body {
                font-family: Arial;
                text-align: center;
                margin-top: 70px;
                background: #f4f6f8;
            }

            .container {
                background: white;
                padding: 35px;
                margin: auto;
                max-width: 450px;
                border-radius: 12px;
                box-shadow: 0 3px 15px #ddd;
            }

            button {
                padding: 12px 25px;
                background: #2563eb;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
            }
        </style>
    </head>

    <body>

        <div class="container">

            <h2>AI Image Classifier</h2>

            <p>Classify an image into one of three categories:</p>

            <p><b>Fruit | Phone | Bag</b></p>

            <form action="/predict" method="post"
                  enctype="multipart/form-data">

                <input type="file" name="file"
                       accept="image/*" required>

                <br><br>

                <button type="submit">Upload & Classify</button>

            </form>

        </div>

    </body>
    </html>
    """

    return HTMLResponse(html_content)


# ==========================================
# 5. IMAGE PREDICTION API
# ==========================================

async def predict(request):

    try:

        form = await request.form()

        uploaded_file = form.get("file")

        if uploaded_file is None or not hasattr(uploaded_file, "read"):
            return JSONResponse(
                {"error": "Please upload an image."},
                status_code=400
            )

        bytes_data = await uploaded_file.read()

        if not bytes_data:
            return JSONResponse(
                {"error": "Uploaded image is empty."},
                status_code=400
            )

        if len(bytes_data) > MAX_IMAGE_SIZE:
            return JSONResponse(
                {"error": "Image exceeds the 10 MB limit."},
                status_code=413
            )

        # Convert uploaded bytes into an image.
        img = PILImage.create(BytesIO(bytes_data))

        # Get trained model.
        learner = request.app.state.learn

        # Run inference outside the async event loop.
        pred, pred_idx, probs = await run_in_threadpool(
            learner.predict,
            img
        )

        confidence = float(probs[int(pred_idx)])

        return JSONResponse({
            "success": True,
            "prediction": str(pred),
            "probability": confidence,
            "confidence_percentage": round(confidence * 100, 2)
        })

    except Exception as error:

        print(f"Prediction error: {error}")

        return JSONResponse(
            {"error": "Unable to classify this image. Please try another image."},
            status_code=400
        )


# ==========================================
# 6. HEALTH CHECK
# ==========================================

async def health(request):

    model_loaded = hasattr(request.app.state, "learn")

    return JSONResponse({
        "status": "healthy" if model_loaded else "starting",
        "model_loaded": model_loaded,
        "classes": ["fruit", "phone", "bag"]
    })


# ==========================================
# 7. REGISTER ROUTES
# ==========================================

routes = [
    Route("/", homepage, methods=["GET"]),
    Route("/predict", predict, methods=["POST"]),
    Route("/health", health, methods=["GET"])
]

app = Starlette(
    debug=False,
    routes=routes,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)
