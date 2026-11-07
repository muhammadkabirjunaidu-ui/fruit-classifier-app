from pathlib import Path
from fastai.vision.all import *
import aiohttp
import os
from contextlib import asynccontextmanager
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route, WebSocketRoute, Mount

# 1. Path setup
app_path = Path(__file__).parent
model_file_name = 'export.pkl'
model_file_url = 'https://drive.google.com/uc?export=download&id=1bZQIlGTzTD-U8YESD1weqRnNQCH3cVvW'  # Replace with your direct download link

# 2. Define lifespan context manager for startup tasks (replaces @app.on_event)
@asynccontextmanager
async def lifespan(app: Starlette):
    local_model_path = app_path.parent / model_file_name
    
    # Remove corrupted download attempts if they exist
    if local_model_path.exists():
        with open(local_model_path, 'rb') as f:
            header = f.read(5)
            if header.startswith(b'<') or header.startswith(b'<!DOC'):
                print("Detected invalid HTML file instead of model binary. Removing corrupted file...")
                local_model_path.unlink()

    if not local_model_path.exists():
        print("Downloading model from cloud storage...")
        async with aiohttp.ClientSession() as session:
            async with session.get(model_file_url) as response:
                with open(local_model_path, 'wb') as f:
                    f.write(await response.read())
    
    global learn
    learn = load_learner(local_model_path)
    print("Model loaded successfully!")
    yield

# 3. Initialize Starlette with lifespan
app = Starlette(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, 
    allow_origins=['*'], 
    allow_credentials=True, 
    allow_methods=['*'], 
    allow_headers=['*']
)

# 4. Define the Homepage with a simple upload form
async def homepage(request):
    html_content = """
    <html>
        <head><title>Fruit, Phone & Bag Classifier</title></head>
        <body style="font-family: Arial; text-align: center; margin-top: 50px;">
            <h2>AI Image Classifier (Fruit, Phone, Bag)</h2>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file"><br><br>
                <input type="submit" value="Upload & Classify">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(html_content)

# 5. Define the Prediction endpoint
async def predict(request):
    form = await request.form()
    bytes_data = await form["file"].read()
    
    # Predict image category using your Fastai model
    img = PILImage.create(BytesIO(bytes_data))
    pred, pred_idx, probs = learn.predict(img)
    
    return JSONResponse({
        "prediction": str(pred),
        "probability": float(probs[pred_idx])
    })

# Register routes
app.routes.append(Route("/", homepage, methods=["GET"]))
app.routes.append(Route("/predict", predict, methods=["POST"]))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
