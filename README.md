# 🎨 3D Model Generator

Turn any 2D image into a 3D model automatically. Simple as that!

## What You Need

- Docker (with Docker Compose)
- NVIDIA GPU with CUDA 12.x
- At least 61GB VRAM (80GB recommended for best results)

## Quick Start

### Build It

```bash
docker build -f docker/Dockerfile -t forge3d-pipeline:latest .
```

### Run It

**Option 1: Docker Compose (easiest)**
```bash
cd docker
docker-compose up -d --build
```

**Option 2: Direct Docker run**
```bash
docker run --gpus all -p 10006:10006 forge3d-pipeline:latest
```

**Option 3: With custom settings**

Copy `.env.sample` to `.env`, tweak the settings, then:
```bash
docker run --gpus all -p 10006:10006 --env-file .env forge3d-pipeline:latest
```

**Option 4: Development mode**

Mount your code for live changes:
```bash
docker run --gpus all \
  -v ./pipeline_service:/workspace/pipeline_service \
  -p 10006:10006 \
  --env-file .env \
  forge3d-pipeline:latest
```

## How to Use

### 🎲 About Seeds
- Use `seed: 42` (or any number) for consistent results every time
- Use `seed: -1` to get random variations (default)

### 📤 Upload an Image, Get a 3D Model

**Get PLY file:**
```bash
curl -X POST "http://localhost:10006/generate" \
  -F "prompt_image_file=@your-image.png" \
  -F "seed=42" \
  -o output-model.ply
```

**Get compressed SPZ file:**
```bash
curl -X POST "http://localhost:10006/generate-spz" \
  -F "prompt_image_file=@your-image.png" \
  -F "seed=42" \
  -o output-model.spz
```

### 📦 Send Base64, Get JSON

Perfect for API integrations:
```bash
curl -X POST "http://localhost:10006/generate_from_base64" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_type": "image",
    "prompt_image": "<your-base64-encoded-image>",
    "seed": 42
  }'
```

### ❤️ Health Check

Make sure everything's running:
```bash
curl http://localhost:10006/health
```