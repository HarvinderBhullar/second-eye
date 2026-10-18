"""Hugging Face Space entry point (Gradio SDK). Serves the same FastAPI + Gradio app as Docker."""

import os

import uvicorn

from second_eye.main import app

if __name__ == "__main__":
    port = int(os.getenv("PORT") or os.getenv("GRADIO_SERVER_PORT") or 7860)
    uvicorn.run(app, host=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"), port=port)
