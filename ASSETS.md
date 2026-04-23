# NexusLock Assets

## FaceNet Model

The FaceNet TFLite model is **not** bundled in the repository (it's 44 MB).
Instead, it is automatically downloaded on first app launch from:

```
https://raw.githubusercontent.com/shubham0204/OnDevice-Face-Recognition-Android/main/app/src/main/assets/facenet_512.tflite
```

**Model specs:**
- Architecture: InceptionResnetV1 (David Sandberg / deepface)
- Input: `[1, 160, 160, 3]` float32, pixels in `[-1, 1]`
- Output: `[1, 512]` float32, L2-normalized embedding
- Format: TFLite with FP16 quantization
- Size: ~44 MB
- Dimension: 512-dim embeddings

The model is cached locally in the app's document directory (`models/facenet_512.tflite`)
after the first download. Subsequent launches load from the local cache.

## App Icons

Place your app icons in the `assets/` directory:
- `assets/icon.png` — App icon (1024×1024)
- `assets/adaptive-icon.png` — Android adaptive icon (1024×1024)
- `assets/splash-icon.png` — Splash screen icon

These are referenced in `app.json`. If they are missing, the build will
use Expo's default placeholder icons.
