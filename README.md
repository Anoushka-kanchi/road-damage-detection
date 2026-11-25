# Road Damage Detection

A road damage detection project built for DataSprint 2.0 using YOLOv8. It includes a web UI for uploading road images and viewing model detections.

## Run locally

Python 3.11 is recommended. The trained model is provided as `best.pt` in the repository root.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open http://localhost:5000. The `/api/health` endpoint reports whether the model loaded successfully.

## Model

- YOLOv8n, pretrained on COCO and fine-tuned on the provided road damage dataset
- Classes: Longitudinal Crack, Transverse Crack, Alligator Crack, and Pothole

## Tech stack

- Python
- Flask
- Ultralytics
- PyTorch

## Team

DataSprint 2.0
