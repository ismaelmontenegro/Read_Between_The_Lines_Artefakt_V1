# Meeting Analysis Tool

A web-based application for real-time analysis of video meetings, built with oTree. The tool provides automated transcription, speaker diarization, topic modeling, sentiment analysis, and visual analytics of meeting dynamics.

## Features

- Real-time video/audio processing
- Speaker diarization and tracking
- Speech-to-text transcription
- Topic modeling and analysis
- Sentiment analysis
- Interactive visualization dashboard
- Support for multiple video formats

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for optimal performance)
- [oTree](https://www.otree.org/)
- FFmpeg

## Installation

1. Clone the repository:
```bash
git clone [https://github.com/ismaelmontenegro/Read_Between_The_Lines_Artefakt_V1.git]
cd [Read_Between_The_Lines_Artefakt_V1]
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up your Hugging Face authentication token for the pyannote models:
```bash
export HF_TOKEN="your-token-here"
```

## Usage

1. Start the oTree development server:
```bash
otree devserver
```

2. Navigate to the application in your web browser (default: http://localhost:8000)

3. Upload or provide the path to your video file

4. View real-time analysis results in the interactive dashboard

## Application Structure

- `DemoApp/` - Main application directory
  - `__init__.py` - oTree page and player definitions
  - `videoTranscriberV2.py` - Core video analysis functionality
  - `MyPage.html` - Video upload interface
  - `Results.html` - Analysis results dashboard

## Dependencies

Key dependencies include:
- pyannote.audio - For speaker diarization
- whisper - For speech transcription
- textblob - For sentiment analysis
- scikit-learn - For topic modeling
- moviepy - For video processing
- torch - For deep learning models

## Notes

- The application creates temporary audio files during processing which are automatically cleaned up
- Analysis results are saved in JSON format in the `_static/DemoApp/` directory
- Real-time processing simulates live analysis with configurable buffer sizes and overlap

## Author

Ismael Montenegro
