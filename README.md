# 🎙️ Urdu Speech Translation App

A modern, real-time speech translation application that translates Urdu speech to multiple languages and supports bidirectional conversation translation. Built with FastAPI backend and React frontend.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![React](https://img.shields.io/badge/React-18.2-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Usage](#usage)
- [Workflow](#workflow)
- [API Documentation](#api-documentation)
- [Technical Details](#technical-details)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## ✨ Features

### 🎯 Core Capabilities

- **🎤 Browser-Based Recording**: Record audio directly in your browser - no external software needed
- **🌍 Multi-Language Support**: Translate to 10+ languages including English, French, Spanish, German, Italian, Portuguese, Japanese, Korean, Chinese, Arabic, and Hindi
- **🔄 Two-Way Translation**: 
  - **Single Mode**: Urdu → Target Language
  - **Conversation Mode**: Bidirectional translation (Urdu ↔ Target Language)
- **⚡ Real-Time Processing**: See status updates as your audio is processed
- **🔊 Audio Playback**: Listen to translated speech instantly
- **📝 Conversation History**: Save and review all conversation exchanges
- **🎨 Modern UI**: Clean, minimal, and responsive design

### 🚀 Advanced Features

- **Fine-Tuned Urdu Model**: Uses `whisper-small-urdu` for superior Urdu transcription accuracy
- **GPU Acceleration**: Leverages Apple Silicon (MPS) for faster processing
- **Multi-Language Transcription**: Supports transcription in multiple languages (not just Urdu)
- **Automatic Speaker Switching**: In conversation mode, automatically switches between speakers
- **Audio Format Conversion**: Automatically converts WebM to WAV for processing

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React)                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Browser Audio Recording (MediaRecorder API)     │   │
│  │  WebM → WAV Conversion                            │   │
│  │  Base64 Encoding                                  │   │
│  └─────────────────────────────────────────────────┘   │
│                        ↓                                │
│  ┌─────────────────────────────────────────────────┐   │
│  │  UI Components                                   │   │
│  │  - Mode Toggle (Single/Conversation)            │   │
│  │  - Language Selector                             │   │
│  │  - Recording Controls                            │   │
│  │  - Conversation History                          │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          ↓ HTTP/REST API
┌─────────────────────────────────────────────────────────┐
│              Backend (FastAPI)                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Step 1: Speech-to-Text                         │   │
│  │  - Urdu: whisper-small-urdu (fine-tuned)        │   │
│  │  - Others: whisper-small (multilingual)          │   │
│  └─────────────────────────────────────────────────┘   │
│                        ↓                                │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Step 2: Translation                            │   │
│  │  - TranslateGemma:4b via Ollama                │   │
│  │  - Supports bidirectional translation           │   │
│  └─────────────────────────────────────────────────┘   │
│                        ↓                                │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Step 3: Text-to-Speech                         │   │
│  │  - Piper TTS (local, fast)                      │   │
│  │  - Language-specific voices                     │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Technology Stack

**Backend:**
- **FastAPI**: Modern, fast web framework for building APIs
- **Whisper**: OpenAI's speech recognition (fine-tuned for Urdu)
- **TranslateGemma**: Ollama-based translation model
- **Piper TTS**: Fast, local neural text-to-speech
- **PyTorch**: Deep learning framework (with MPS support)

**Frontend:**
- **React 18**: Modern UI library
- **Vite**: Fast build tool and dev server
- **Axios**: HTTP client for API calls
- **Lucide React**: Beautiful icon library

## 📸 Screenshots

### Main Interface

![Main UI](screenshots/Screenshot%202026-01-21%20at%204.55.01%20PM.png)
*Clean, modern interface with mode toggle and language selector*

### Single Translation Mode

![Single Mode](screenshots/Screenshot%202026-01-21%20at%204.55.01%20PM.png)
*Translate Urdu speech to any target language*

### Conversation Mode

![Conversation Mode](screenshots/Screenshot%202026-01-21%20at%204.47.06%20PM.png)
*Two-way translation with speaker indicators*

### Conversation History

![History](screenshots/conversation-history.png)
*View and replay all conversation exchanges*

**What to capture**: Conversation history panel showing multiple exchanges with source text, translations, and play buttons.

### Processing Status

![Processing](screenshots/processing.png)
*Real-time status updates during processing*

**What to capture**: Status message showing "Processing..." or "Transcribing..." with loading indicator.

### Example Translation Output

![Example Output](screenshots/example-output.png)
*Example: Urdu "آپ کیسے ہیں؟" → English "How are you?"*

**What to capture**: A successful translation showing both Urdu text and translated text side by side.

## 🚀 Installation

### Prerequisites

- **Python 3.8+**
- **Node.js 18+** and npm
- **Ollama** with `translategemma:4b` model installed

### Step 1: Install Ollama

```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai

# Pull the translation model
ollama pull translategemma:4b
```

### Step 2: Install Backend Dependencies

```bash
cd Text_to_Speech
pip install -r requirements.txt
```

### Step 3: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

## 💻 Usage

### Quick Start

**Option 1: Use the startup script**
```bash
./start_ui.sh
```

**Option 2: Manual start**

Terminal 1 - Backend:
```bash
python api_server.py
```

Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

Then open `http://localhost:3000` in your browser.

### Single Translation Mode

1. Select target language from dropdown
2. Click "Start Recording"
3. Speak in Urdu
4. Click "Stop Recording"
5. View transcribed Urdu text and translation
6. Click "Play Translation" to hear the audio

### Conversation Mode

1. Click "Conversation Mode" toggle
2. Select target language (e.g., French)
3. **Person A**: Click "Start Recording" → Speak Urdu → Stop
4. System automatically translates and plays French audio
5. **Person B**: Click "Start Recording" → Speak French → Stop
6. System automatically translates and plays Urdu audio
7. Continue conversation back and forth
8. View full conversation history below

## 🔄 Workflow

### Single Mode Workflow

```
User Records Urdu Audio
         ↓
Browser: WebM → WAV Conversion
         ↓
Base64 Encoding
         ↓
POST /api/process-base64
         ↓
Backend: Transcribe Urdu (Whisper)
         ↓
Backend: Translate Urdu → Target (TranslateGemma)
         ↓
Backend: TTS in Target Language (Piper)
         ↓
Return: Urdu Text + Translation + Audio
         ↓
Frontend: Display Results + Play Audio
```

### Conversation Mode Workflow

```
Person A Records Urdu
         ↓
Transcribe Urdu → Translate to French → TTS French
         ↓
Auto-play French Audio → Switch to Person B
         ↓
Person B Records French
         ↓
Transcribe French → Translate to Urdu → TTS Urdu
         ↓
Auto-play Urdu Audio → Switch to Person A
         ↓
Repeat...
```

## 📡 API Documentation

### Endpoints

#### `GET /api/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

#### `GET /api/languages`
Get list of supported languages.

**Response:**
```json
{
  "languages": [
    {"code": "en", "name": "English"},
    {"code": "fr", "name": "French"},
    ...
  ]
}
```

#### `POST /api/process-base64`
Process audio in single mode (Urdu → Target).

**Request:**
```json
{
  "audio_data": "data:audio/wav;base64,...",
  "target_language": "English"
}
```

**Response:**
```json
{
  "status": "success",
  "urdu_text": "آپ کیسے ہیں؟",
  "translated_text": "How are you?",
  "audio_base64": "data:audio/wav;base64,..."
}
```

#### `POST /api/conversation`
Process audio in conversation mode (bidirectional).

**Request:**
```json
{
  "audio_data": "data:audio/wav;base64,...",
  "source_language": "Urdu",
  "target_language": "French",
  "conversation_mode": true
}
```

**Response:**
```json
{
  "status": "success",
  "source_text": "آپ کا نام کیا ہے؟",
  "source_language": "Urdu",
  "translated_text": "Comment vous appelez-vous?",
  "target_language": "French",
  "audio_base64": "data:audio/wav;base64,..."
}
```

## 🔧 Technical Details

### Speech-to-Text

- **Urdu**: Uses fine-tuned `khawajaaliarshad/whisper-small-urdu` model
- **Other Languages**: Uses standard `openai/whisper-small` multilingual model
- **GPU Acceleration**: Apple Silicon MPS (Metal Performance Shaders)
- **Sample Rate**: 16 kHz (resampled from 48 kHz browser recording)

### Translation

- **Model**: TranslateGemma:4b via Ollama
- **Supports**: 100+ languages
- **Bidirectional**: Can translate between any two supported languages
- **Format**: Professional translator prompt format for accuracy

### Text-to-Speech

- **Engine**: Piper TTS (local, fast)
- **Voices**: Language-specific voices (e.g., `en_US-lessac-medium` for English)
- **Format**: 16-bit PCM WAV
- **Sample Rate**: Model-dependent (typically 22 kHz)

### Audio Processing

- **Browser Recording**: MediaRecorder API (WebM format)
- **Conversion**: Web Audio API converts WebM → WAV
- **Encoding**: Base64 for API transmission
- **Processing**: DC offset removal, high-pass filtering, normalization

## 📝 Examples

### Example 1: Simple Translation

**Input (Urdu Speech):**
> "آپ کیسے ہیں؟"

**Output (English):**
> "How are you?"

**Audio**: Plays English TTS

**Screenshot**: `screenshots/example-simple.png`

---

### Example 2: Conversation Exchange

**Person A (Urdu):**
> "میرا نام احمد ہے"

**Translation (French):**
> "Mon nom est Ahmed"

**Person B (French):**
> "Enchanté, Ahmed"

**Translation (Urdu):**
> "خوشی ہوئی، احمد"

**Screenshot**: `screenshots/example-conversation.png`

---

### Example 3: Complex Sentence

**Input (Urdu):**
> "جب تک میں نے یہ کام مکمل نہیں کیا، میں نہیں سو سکتا"

**Output (English):**
> "Until I complete this work, I cannot sleep"

**Screenshot**: `screenshots/example-complex.png`

---

### Example 4: Multiple Languages

**Urdu → Spanish:**
- Input: "شکریہ"
- Output: "Gracias"

**Urdu → German:**
- Input: "آپ کہاں رہتے ہیں؟"
- Output: "Wo wohnen Sie?"

**Screenshot**: `screenshots/example-multilang.png`

## 🐛 Troubleshooting

### Backend Issues

**Problem**: API server won't start
- **Solution**: Check if port 8000 is available: `lsof -ti:8000`
- **Solution**: Ensure all dependencies installed: `pip install -r requirements.txt`

**Problem**: Translation fails
- **Solution**: Verify Ollama is running: `ollama serve`
- **Solution**: Check model is installed: `ollama list`
- **Solution**: Pull model if missing: `ollama pull translategemma:4b`

**Problem**: Transcription is slow
- **Solution**: First run loads models (slow), subsequent runs faster
- **Solution**: Ensure GPU acceleration enabled (MPS on Apple Silicon)

### Frontend Issues

**Problem**: Microphone not working
- **Solution**: Use HTTPS or localhost (required for microphone access)
- **Solution**: Check browser permissions for microphone
- **Solution**: Try different browser (Chrome recommended)

**Problem**: CORS errors
- **Solution**: Backend CORS configured for `localhost:3000`
- **Solution**: Ensure backend running on port 8000

**Problem**: Audio playback fails
- **Solution**: Check browser audio permissions
- **Solution**: Verify audio format is supported

### Model Issues

**Problem**: Urdu transcription inaccurate
- **Solution**: Ensure using fine-tuned model (`whisper-small-urdu`)
- **Solution**: Speak clearly and reduce background noise
- **Solution**: Check audio quality (48 kHz recording recommended)

**Problem**: Translation quality poor
- **Solution**: Ensure TranslateGemma model is latest version
- **Solution**: Check source language is correctly identified
- **Solution**: For complex sentences, break into shorter phrases

## 📊 Performance

- **Transcription**: ~2-5 seconds (depending on audio length)
- **Translation**: ~3-8 seconds (depending on text length)
- **TTS**: ~1-3 seconds (depending on text length)
- **Total Pipeline**: ~6-16 seconds per translation

*Note: First run is slower due to model loading. Subsequent runs are faster.*

## 🎯 Supported Languages

### Transcription Support
- Urdu (fine-tuned model)
- English, French, Spanish, German, Italian, Portuguese
- Japanese, Korean, Chinese, Arabic, Hindi
- And 100+ more via Whisper multilingual model

### Translation Support
- All languages supported by TranslateGemma (100+)
- Common pairs: Urdu ↔ English, French, Spanish, German, etc.

### TTS Support
- English (US/UK), French, Spanish, German, Italian, Portuguese
- Japanese, Korean, Chinese, Arabic, Hindi
- And more via Piper voice models

## 📁 Project Structure

```
Text_to_Speech/
├── api_server.py              # FastAPI backend server
├── speech_to_text.py          # Urdu transcription (fine-tuned model)
├── speech_to_text_multi.py    # Multi-language transcription
├── translate_text.py           # Urdu → Target translation
├── translate_text_multi.py     # Bidirectional translation
├── text_to_speech.py          # Piper TTS integration
├── ollama_integration.py      # Ollama client wrapper
├── language_codes.py          # Language code mappings
├── text_cleaner.py            # Text cleaning utilities
├── requirements.txt           # Python dependencies
├── start_ui.sh                # Startup script
├── frontend/                  # React frontend
│   ├── src/
│   │   ├── App.jsx            # Main React component
│   │   ├── App.css            # Styles
│   │   ├── main.jsx           # Entry point
│   │   └── index.css          # Global styles
│   ├── package.json           # Frontend dependencies
│   └── vite.config.js         # Vite configuration
└── screenshots/               # UI screenshots
    └── README.md              # Screenshot instructions
```

## 🎯 What's Been Accomplished

### Phase 1: Core Translation Pipeline
- ✅ Implemented Urdu speech-to-text using fine-tuned Whisper model
- ✅ Integrated TranslateGemma for high-quality translation
- ✅ Added Piper TTS for natural-sounding speech synthesis
- ✅ Created complete pipeline: Record → Transcribe → Translate → TTS → Play

### Phase 2: Modern Web UI
- ✅ Built React frontend with Vite
- ✅ Implemented browser-based audio recording
- ✅ Created modern, responsive UI design
- ✅ Added real-time status updates and progress indicators
- ✅ Integrated audio playback functionality

### Phase 3: Two-Way Translation
- ✅ Implemented bidirectional translation support
- ✅ Created conversation mode with speaker switching
- ✅ Added conversation history tracking
- ✅ Built multi-language transcription support
- ✅ Added automatic turn-taking indicators

### Phase 4: API & Integration
- ✅ Created FastAPI backend with REST endpoints
- ✅ Implemented CORS for frontend-backend communication
- ✅ Added base64 audio handling for browser compatibility
- ✅ Created comprehensive error handling

### Key Achievements
- 🎯 **100+ Language Support**: Translate between any supported language pairs
- ⚡ **Fast Processing**: GPU-accelerated transcription on Apple Silicon
- 🎨 **Modern UX**: Clean, intuitive interface with real-time feedback
- 🔄 **Bidirectional**: Full conversation support, not just one-way translation
- 📱 **Responsive**: Works on desktop and mobile devices

## 📄 License

MIT License - feel free to use this project for your own purposes.

## 🙏 Acknowledgments

- **Whisper**: OpenAI's speech recognition model
- **TranslateGemma**: Translation model via Ollama
- **Piper TTS**: Fast, local text-to-speech
- **Fine-tuned Urdu Model**: `khawajaaliarshad/whisper-small-urdu`

## 📧 Contact

For questions or issues, please open an issue on the repository, or contact me via
Email: mortazaameer8@gmail.com
Website: Mortaza76.github.io

Muhammad Ameer Mortaza
Ghulam Ishaq Khan Institute (GIKI)
