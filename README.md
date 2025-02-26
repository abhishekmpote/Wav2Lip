# Lip Sync Model Implementation

## Overview

This repository provides an implementation of a lip-sync model using the open-source Wav2Lip framework. It takes an input image of a face and an audio file, then generates a video where the lip movements of the face are synchronized with the audio. This project was developed as part of an internship assignment.

## Features

- **Lip-syncing**: Uses Wav2Lip to synchronize lip movements with the provided audio.
- **Text-to-Speech (TTS)**: Converts a given text into speech using `gTTS`.
- **Face Alignment (Optional)**: Detects and aligns the face before processing to improve lip-sync accuracy.

## Requirements

Make sure to install the required dependencies before running the script.

```bash
apt-get install -y ffmpeg
pip install gtts opencv-python torch torchvision librosa==0.8.1
pip install face-alignment
```

## Setup

### Clone the Repository

```bash
git clone https://github.com/Rudrabha/Wav2Lip
```

### Download Pre-trained Model

```bash
wget "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA" -O "/content/Wav2Lip/checkpoints/Wav2Lip.pth"
```

## Usage

### 1. Convert Text to Speech (TTS)

```python
from gtts import gTTS

text = "Namaste Mathangi! My name is Anika, and I’m here to guide you through managing your credit card dues..."
tts = gTTS(text, lang='en', tld='co.in')
tts.save("/content/output_audio.mp3")
print("Audio file saved as output_audio.mp3")
```

### 2. Upload and Process Input Image

```python
from google.colab import files
uploaded = files.upload()
input_image = list(uploaded.keys())[0]
print(f"Uploaded image: {input_image}")
```

### 3. (Optional) Face Alignment

**Note**: This feature was added optionally and is not required to run Wav2Lip.

```python
import face_alignment
from PIL import Image
import numpy as np
import torch

image = Image.open(f"/content/{input_image}").convert("RGB")
image = np.array(image)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu')
preds = fa.get_landmarks(image)

if preds:
    landmarks = preds[0]
    x_min, y_min = landmarks.min(axis=0)
    x_max, y_max = landmarks.max(axis=0)
    padding = 50  
    cropped_image = image[max(0, int(y_min) - padding):min(image.shape[0], int(y_max) + padding),
                          max(0, int(x_min) - padding):min(image.shape[1], int(x_max) + padding)]
    cropped_image = Image.fromarray(cropped_image)
    cropped_image.save("/content/aligned_face.jpg")
    input_image = "aligned_face.jpg"
    print("Face aligned and cropped. Saved as aligned_face.jpg")
else:
    print("No face detected. Using the original image.")
```

### 4. Run Wav2Lip Inference

```bash
cd /content/Wav2Lip && python inference.py \
  --checkpoint_path /content/Wav2Lip/checkpoints/Wav2Lip.pth \
  --face "/content/{input_image}" \
  --audio "/content/output_audio.mp3" \
  --outfile "/content/output_video.mp4"
```

### 5. Download Output Video

```python
from google.colab import files
files.download("/content/output_video.mp4")
```

## Notes

- The **face alignment** step is optional and was an additional improvement considered to enhance lip-sync accuracy.
- The Wav2Lip model requires an audio file and a face image as inputs.
- The output video will be saved as `output_video.mp4`.

## Acknowledgments

- **Wav2Lip**: [GitHub](https://github.com/Rudrabha/Wav2Lip)
- **Google Text-to-Speech (gTTS)**: [PyPI](https://pypi.org/project/gTTS/)
- **Face Alignment**: [GitHub](https://github.com/1adrianb/face-alignment)

## License

This project follows the open-source license guidelines as per the Wav2Lip repository.

