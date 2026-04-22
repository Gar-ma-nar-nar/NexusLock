# Nexus-Lock™ v2.1.2

Real-time face-based phone security. 100% on-device FaceNet inference.

**Model:** InceptionResnetV1 FaceNet (89.6 MB)
- Input: 160×160×3 RGB float32, normalized to [-1, 1]
- Output: 512-dim L2-normalized embedding

**Setup:**
```bash
npm install
npx expo install --fix
# facenet.tflite MUST be in assets/models/
npx expo prebuild
npx expo run:android
```

© 2026 Nickol Joy Bowman. All Rights Reserved.
