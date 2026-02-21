# Python OpenGL Mini Renderer

A small educational 3D renderer built with Python, `pygame`, and the OpenGL fixed-function pipeline.

It loads OBJ/MTL assets, builds a simple scene, applies player physics, and renders with lighting and textures.

## Features

- OBJ/MTL model loading with face triangulation
- Material parsing (`Kd`, `map_Kd`) and texture caching
- Fixed-function OpenGL rendering (`GL_LIGHT0`, depth test, textured triangles)
- Third-person orbit camera with smoothing and zoom
- Basic player movement, gravity, and jump
- Perspective and orthographic projection toggle
- Optional screenshot capture (`F12`)

## Repository Layout

```text
.
├── assets/
│   └── Models/
│       ├── room.obj
│       ├── room.mtl
│       ├── Egg.obj
│       ├── Egg.mtl
│       └── *.png textures
├── src/
│   ├── __main__.py
│   ├── app.py
│   ├── assets_io/
│   │   └── obj_loader.py
│   ├── engine/
│   │   ├── camera.py
│   │   ├── framebuffer.py
│   │   ├── physics.py
│   │   └── scene.py
│   ├── math_utils/
│   │   └── linalg.py
│   └── rendering/
│       └── gl_renderer.py
├── requirements.txt
└── .gitignore
```

## Requirements

- Python 3.10+
- OpenGL-capable GPU/driver

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

## Run

From project root:

```bash
python -m src
```

## Controls

- `W / A / S / D`: move player
- `Space`: jump
- `Right Mouse Button + Mouse`: rotate camera
- `Mouse Wheel`: zoom
- `P`: perspective projection
- `O`: orthographic projection
- `T`: toggle camera smoothing
- `R`: reset camera
- `F12`: save screenshot to `output_render.png`
- `Esc`: quit

## GitHub Notes

This repository includes a Python/OpenGL-focused `.gitignore` so local and generated files are not committed, including:

- virtual environments (`myenv/`, `.venv/`, `venv/`)
- bytecode/caches (`__pycache__/`, `*.pyc`)
- editor/OS files (`.vscode/`, `.idea/`, `.DS_Store`)
- runtime outputs (`output_render.png`, logs)

## Troubleshooting

- If window/context creation fails, verify your graphics drivers and OpenGL support.
- If textures are missing, confirm referenced files exist under `assets/Models`.
- If `pygame` import fails, ensure the same Python environment is used for install and run.
