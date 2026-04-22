# Required Assets

Place these files before building:

assets/models/facenet.tflite    ← Your FaceNet InceptionResnetV1 model (89.6 MB)
                                   Input:  [1, 160, 160, 3] float32 (pixels in [-1, 1])
                                   Output: [1, 512] float32 (L2-normalized embedding)
