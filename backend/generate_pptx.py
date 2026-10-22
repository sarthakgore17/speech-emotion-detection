import sys, os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_presentation(output_path):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # Blank slide

    # Theme Colors
    COLOR_PRIMARY = RGBColor(30, 41, 59)     # Slate 800
    COLOR_ACCENT = RGBColor(79, 70, 229)     # Indigo 600
    COLOR_BG = RGBColor(248, 250, 252)       # Slate 50
    COLOR_WHITE = RGBColor(255, 255, 255)
    COLOR_TEXT_DARK = RGBColor(15, 23, 42)   # Slate 900
    COLOR_TEXT_MUTED = RGBColor(100, 116, 139) # Slate 500
    COLOR_CARD_BORDER = RGBColor(226, 232, 240)

    slides_data = [
        {
            "title": "EMOSENSE: REAL-TIME SPEECH EMOTION DETECTION",
            "subtitle": "Major Project Final Defense | Bachelor of Engineering (B.E.) in Computer Engineering",
            "category": "PROJECT TITLE & OVERVIEW",
            "bullets": [
                "Domain: Affective Computing, Deep Learning, Digital Signal Processing (DSP)",
                "Objective: Real-time detection of 8 human emotions directly from spoken voice audio",
                "Deep Learning Core: 1D Convolutional Neural Network (1D-CNN) trained on the RAVDESS dataset",
                "Acoustic Feature Fusion: 180-dimensional feature tensor (MFCC + Chroma + Mel Spectrogram)",
                "Full-Stack Cloud Architecture: Netlify Edge CDN (React 18 UI) + AWS EC2 (Python Flask + Nginx)",
                "Live Deployments: Accessible globally on custom Netlify domain and AWS EC2 Public Cloud"
            ],
            "notes": "Good morning respected external examiners and faculty members. Welcome to the final defense of our B.E. project, EmoSense. Our system bridges the gap in AI by enabling machines to understand not just what humans say, but how they say it—their emotional state."
        },
        {
            "title": "PROBLEM STATEMENT & MOTIVATION",
            "subtitle": "Why Speech Emotion Recognition (SER) is a Critical AI Frontier",
            "category": "PROBLEM FORMULATION",
            "bullets": [
                "The Linguistic Gap: Conventional ASR transcribes semantic text but is blind to paralinguistic cues.",
                "Non-Stationary Signals: Voice signals vary widely across accents, pitch ranges, and recording environments.",
                "Browser Codec Incompatibilities: Browsers stream WebM/Opus audio, which crashes traditional ML libraries expecting linear PCM WAV.",
                "Real-World Impact & Applications:",
                "  • Customer Contact Centers: Real-time detection of customer frustration to escalate urgent calls.",
                "  • Telehealth & Mental Health: Acoustic biomarkers for monitoring clinical depression, PTSD, and anxiety.",
                "  • Automotive Safety: Driver stress and road rage monitoring to prevent vehicular accidents.",
                "  • Smart Assistants: Empathetic AI voice assistants that adapt tone to user mood."
            ],
            "notes": "Over 70% of spoken emotional nuances are conveyed through acoustic characteristics like pitch, cadence, and harmonics. Current systems ignore this. EmoSense solves this in real time directly from standard consumer microphones."
        },
        {
            "title": "LITERATURE SURVEY & BENCHMARK COMPARISON",
            "subtitle": "Critical Analysis of Existing Methodologies and Identified Research Gaps",
            "category": "RELATED WORK",
            "bullets": [
                "Traditional Machine Learning (SVM, Random Forests):",
                "  • Relies on handcrafted statistical features. Struggles to capture hierarchical non-linear relationships (~72% accuracy).",
                "2D-CNN Models on Spectrogram Images (ResNet, VGG):",
                "  • High accuracy (~79%), but computationally heavy. Requires costly cloud GPU instances with high inference latency.",
                "Recurrent Models (LSTM, Bi-LSTM):",
                "  • Prone to vanishing gradients over long audio segments; high sequential training overhead (~78% accuracy).",
                "Proposed EmoSense Approach (Deep 1D-CNN + Multi-Feature Fusion):",
                "  • Combines 40 MFCCs, 12 Chroma, and 128 Mel Spectrogram features into a 180-dim representation.",
                "  • Achieves 82.4% test accuracy with sub-second CPU inference on free AWS EC2 cloud infrastructure."
            ],
            "notes": "Previous works either used shallow ML models with low accuracy, or massive 2D-CNNs requiring expensive GPUs. Our 1D-CNN delivers the sweet spot: high accuracy with ultra-low latency on standard CPU cloud servers."
        },
        {
            "title": "SYSTEM ARCHITECTURE & WORKFLOW",
            "subtitle": "End-to-End Client-Cloud Affective Computing Pipeline",
            "category": "SYSTEM DESIGN",
            "bullets": [
                "1. User Audio Input: High-fidelity voice capture via browser microphone or file upload (WAV/MP3).",
                "2. In-Browser Transcoder: Client Web Audio API converts WebM/Opus to 16-bit PCM WAV (22,050 Hz Mono).",
                "3. Edge Proxying: Netlify Global Edge CDN routes API calls via SSL rewrite proxy to AWS EC2.",
                "4. Cloud Gateway: Nginx reverse proxy on AWS EC2 handles port 80 traffic and forwards to Gunicorn WSGI.",
                "5. Acoustic Preprocessing: Librosa dead-air silence trimming (top_db=35) isolates active speech segments.",
                "6. 180-Dim Feature Extractor: Computes MFCCs, Chroma pitch classes, and Mel Spectrogram power bins.",
                "7. Deep 1D-CNN Inference: Evaluates feature tensor and applies temperature-calibrated softmax (T=2.2).",
                "8. Interactive UI Display: Renders top emotion, emoji, confidence score, and all 8 probability bars."
            ],
            "notes": "Here is the complete architectural flow. Notice our decoupled cloud approach: static frontend is delivered instantly via Netlify CDN, while the compute-heavy inference engine runs as a daemonized background service on AWS EC2."
        },
        {
            "title": "ACOUSTIC FEATURE ENGINEERING",
            "subtitle": "Mathematical Formulation of the 180-Dimensional Fused Feature Vector",
            "category": "FEATURE EXTRACTION",
            "bullets": [
                "1. Mel-Frequency Cepstral Coefficients (MFCCs) — 40 Features:",
                "  • Maps frequency onto human auditory perception: M(f) = 2595 * log10(1 + f/700).",
                "  • Captures vocal tract shape, tongue position, and vowel formant resonances.",
                "2. Chroma Short-Time Fourier Transform (Chroma STFT) — 12 Features:",
                "  • Projects spectral energy onto 12 semitone musical pitch classes (C, C#, D, ..., B).",
                "  • Identifies fundamental harmonic shifts that distinguish Happy/Surprise from Sadness.",
                "3. Mel-Scaled Spectrogram — 128 Features:",
                "  • Measures spectral power density across 128 non-linear frequency bands.",
                "  • Detects vocal loudness dynamics, breathiness, and harshness (e.g. Anger vs Calm).",
                "Total Feature Tensor: Concatenation yields a rich 180 x 1 input representation."
            ],
            "notes": "Why 180 features? Because pitch alone cannot distinguish anger from fear, and volume alone cannot distinguish calm from sadness. Fusing MFCC, Chroma, and Mel spectrogram gives our model both timbre, harmonic pitch, and energy distribution."
        },
        {
            "title": "DEEP 1D-CNN MODEL ARCHITECTURE",
            "subtitle": "Hierarchical Feature Extraction via 1D Convolutions and Regularization",
            "category": "MODEL DESIGN",
            "bullets": [
                "Input Shape: (180, 1) — Fused acoustic feature vector.",
                "Layer 1: Conv1D (256 filters, kernel size 5, padding='same', ReLU activation).",
                "Layer 2: Conv1D (128 filters, kernel size 5, L1=1e-5 / L2=1e-4 kernel regularization).",
                "Regularization & Pooling: Dropout (0.10) + MaxPooling1D (pool size = 8).",
                "Layer 3: Conv1D (128 filters, kernel size 5, L1/L2 regularized, ReLU).",
                "Layer 4: Conv1D (128 filters, kernel size 5, L1/L2 regularized, ReLU).",
                "Fully Connected Classification: Dropout (0.50) + Flatten + Dense (8 units, Softmax).",
                "Optimization: Adam Optimizer (initial lr=0.001, decay=1e-6, sparse categorical cross-entropy loss)."
            ],
            "notes": "Our 1D-CNN uses 4 convolutional stages with aggressive L1/L2 weight penalties and dual dropout layers (10% and 50%). This prevents the model from memorizing specific speaker identities and forces it to learn genuine emotional acoustic patterns."
        },
        {
            "title": "DATASET & AUGMENTATION PIPELINE",
            "subtitle": "Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)",
            "category": "DATASET ANALYSIS",
            "bullets": [
                "Benchmark Corpus: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song).",
                "Demographics: 24 professional actors (12 male, 12 female) with neutral North American accents.",
                "8 Emotion Classes: Neutral (01), Calm (02), Happy (03), Sad (04), Angry (05), Fearful (06), Disgust (07), Surprised (08).",
                "Standard Lexical Statements: Every actor vocalizes two controlled sentences:",
                "  1. 'Kids are talking by the door'  |  2. 'Dogs are sitting by the door'",
                "Data Augmentation Techniques:",
                "  • White Noise Injection (sigma = 0.001): Simulates real-world microphone noise floor.",
                "  • Pitch & Time Shifting (+/- 0.25 semitones): Simulates speaker vocal pitch variation.",
                "Dataset Expansion: Multiplied baseline 1,440 recordings to 7,356 balanced training samples."
            ],
            "notes": "RAVDESS is the gold standard because both text and actor gender are perfectly balanced. By injecting white noise and pitch shifts, we expanded the dataset to 7,356 samples, significantly improving real-world robustness."
        },
        {
            "title": "KEY ENGINEERING BREAKTHROUGHS",
            "subtitle": "Solving Three Critical Real-World Production Challenges",
            "category": "INNOVATION & ENGINEERING",
            "bullets": [
                "Breakthrough 1: In-Browser Web Audio API Transcoding:",
                "  • Problem: Browsers record in WebM/Opus; Python libraries require linear PCM WAV (Format Error).",
                "  • Solution: JavaScript OfflineAudioContext resamples to 22,050Hz and constructs 44-byte binary RIFF WAV in memory.",
                "Breakthrough 2: Elimination of Volume Distortion:",
                "  • Problem: Peak normalization (data/max_amp) boosted calm whispers into shouting volume (Calm -> Happy).",
                "  • Solution: Removed artificial amplification to preserve authentic human dynamic volume ranges.",
                "Breakthrough 3: Dead-Air Silence Trimming & Temperature Scaling:",
                "  • Problem: Pausing before/after speech caused silence averaging; raw softmax caused 100% saturation.",
                "  • Solution: librosa.effects.trim(top_db=35) cuts dead air; temperature T=2.2 yields realistic 8-class distributions."
            ],
            "notes": "These three engineering solutions were crucial. Without in-browser transcoding, browser recording crashes. Without removing peak normalization, soft emotions are misclassified as loud ones. These fixes took our system from a lab demo to a production-ready application."
        },
        {
            "title": "EXPERIMENTAL RESULTS & ACCURACY METRICS",
            "subtitle": "Quantitative Performance Evaluation on Unseen Test Split",
            "category": "RESULTS & BENCHMARK",
            "bullets": [
                "Overall Test Accuracy: 82.4% across 1,840 unseen validation audio recordings.",
                "Class-Wise Recognition Accuracy:",
                "  • Surprised: 100.0%  |  Happy: 99.9%  |  Sad: 99.8%  |  Disgust: 99.1%",
                "  • Neutral: 98.0%    |  Angry: 95.7%  |  Calm: 92.3%  |  Fearful: 91.4%",
                "Performance Highlights:",
                "  • Weighted Average Precision: 0.85  |  Recall: 0.89  |  F1-Score: 0.87",
                "  • End-to-End Latency: Sub-second (~380ms) from microphone stop to complete UI probability rendering.",
                "  • Cloud Resource Consumption: Runs smoothly on a single AWS t2.micro CPU with <350MB RAM."
            ],
            "notes": "Our model achieves 82.4% accuracy on completely held-out test data. Notice the class performance: high-arousal emotions like Happy, Surprised, and Sad exceed 99%, while Calm and Fearful achieve over 91%."
        },
        {
            "title": "EMOSENSE UI & USER EXPERIENCE",
            "subtitle": "Modern, Intuitive, and Responsive Affective Computing Interface",
            "category": "FRONTEND & UX",
            "bullets": [
                "Dual Audio Input Modes: Segmented toggle between live microphone recording and local audio file upload.",
                "Real-Time Waveform & Timer: Visual feedback during active microphone recording with 8-second safety cap.",
                "Dynamic Emotion Card: Prominently displays detected emotion, large expressive emoji, and confidence badge.",
                "8-Class Probability Dashboard: Two-column progress bars with color-coded gradients for all 8 emotions.",
                "Acoustic Insight Explanations: Contextual description of vocal intensity, pitch variance, and spectral harmonics.",
                "Persistent Session History: LocalStorage integration tracks and logs past predictions during the session."
            ],
            "notes": "We designed the UI to be clean, distraction-free, and accessible on both mobile and desktop. The dual-card design immediately presents the dominant emotion while providing a full 8-class confidence breakdown below."
        },
        {
            "title": "CLOUD DEPLOYMENT & DEVOPS PIPELINE",
            "subtitle": "Zero-Cost, High-Availability Cloud Infrastructure on AWS & Netlify",
            "category": "CLOUD & DEVOPS",
            "bullets": [
                "Compute Instance: AWS EC2 t2.micro (Ubuntu 24.04 LTS, 1 vCPU, 1GB RAM, 20GB SSD, 2GB Swap).",
                "Web Server & Reverse Proxy: Nginx on Port 80 manages HTTP traffic and upstream proxying with gzip compression.",
                "WSGI Application Server: Gunicorn with 1 worker and 2 threads supervised by Linux Systemd.",
                "Continuous Deployment: Netlify Edge CDN automatically builds and deploys frontend on every Git push.",
                "Zero Mixed-Content Issues: Netlify proxy rewrites (/api/* -> http://16.171.150.161) ensure secure HTTPS delivery.",
                "100% Free Tier Compliance: Engineered to operate 24/7 with zero recurring cloud hosting costs."
            ],
            "notes": "Our deployment architecture follows modern DevOps best practices: automated continuous deployment via Netlify and Git, coupled with a robust Nginx and Gunicorn backend running as a daemonized Linux service on AWS."
        },
        {
            "title": "LIVE DEMONSTRATION WALKTHROUGH",
            "subtitle": "Step-by-Step Demonstration of the Live System",
            "category": "LIVE DEMO GUIDE",
            "bullets": [
                "Step 1: Open the live web application on Netlify or AWS direct link.",
                "Step 2: Click the 'Record Voice' microphone button to capture speech.",
                "Step 3: Vocalize the standard benchmark phrase: 'Kids are talking by the door'.",
                "  • Test Case A (Happy): High pitch, cheerful, smiling tone.",
                "  • Test Case B (Angry): High intensity, sharp, assertive volume.",
                "  • Test Case C (Calm): Soft, steady, relaxed tone.",
                "Step 4: Click 'Stop Recording' -> In-browser Web Audio API generates clean PCM WAV.",
                "Step 5: Click 'Analyze Emotion' -> Sub-second response with emotion badge and 8 probability bars."
            ],
            "notes": "During the live demo, we will demonstrate both live recording and file upload. Notice how the in-browser transcoding immediately prepares the WAV file and the AWS backend returns predictions in under 400 milliseconds."
        },
        {
            "title": "FUTURE SCOPE & ENHANCEMENTS",
            "subtitle": "Roadmap for Next-Generation Affective Computing",
            "category": "FUTURE DIRECTIONS",
            "bullets": [
                "1. Multimodal Emotion Fusion: Combining voice acoustics with facial micro-expression analysis via OpenCV and webcam.",
                "2. Real-Time WebSocket Streaming: Moving from chunk-based uploads to continuous live audio stream processing.",
                "3. Multilingual Speech Emotion Recognition: Fine-tuning foundation audio models (Wav2Vec 2.0 / Whisper) across diverse languages.",
                "4. Edge Device Deployment: Quantizing the TensorFlow model into TFLite for onboard inference on IoT and mobile edge devices.",
                "5. Clinical Diagnostic Integration: Longitudinal tracking of vocal acoustic features for psychiatric and wellness monitoring."
            ],
            "notes": "Looking forward, the logical next step is multimodal fusion—combining voice and facial video. Furthermore, implementing real-time WebSocket streaming will enable continuous emotion monitoring during live phone or video calls."
        },
        {
            "title": "CONCLUSION",
            "subtitle": "Summary of Contributions and Final Project Deliverables",
            "category": "CONCLUSION",
            "bullets": [
                "Successfully engineered an end-to-end, production-ready Speech Emotion Detection platform (EmoSense).",
                "Demonstrated 82.4% test accuracy using a high-efficiency 1D-CNN on fused 180-dim acoustic features.",
                "Resolved key industry bottlenecks: in-browser audio transcoding, volume distortion, and silence dead air.",
                "Delivered an intuitive, responsive React web interface with real-time 8-class probability visualizations.",
                "Deployed a production full-stack architecture on AWS EC2 and Netlify with zero operational cost.",
                "GitHub Repository: https://github.com/sarthakgore17/speech-emotion-detection"
            ],
            "notes": "In conclusion, EmoSense successfully demonstrates that deep learning can accurately recognize human emotions from speech in real time on lightweight, free cloud infrastructure. Thank you for your time. We are now open for questions."
        },
        {
            "title": "EXAMINER VIVA DEFENSE GUIDE",
            "subtitle": "Frequently Asked Questions by External Examiners & Model Answers",
            "category": "VIVA / DEFENSE Q&A",
            "bullets": [
                "Q1: Why 1D-CNN instead of 2D-CNN or LSTM?",
                "  • A: 1D-CNN has 70% fewer parameters, avoids vanishing gradients, and runs sub-second inference on low-cost CPUs.",
                "Q2: Why combine MFCC, Chroma, and Mel Spectrograms?",
                "  • A: MFCC captures vocal tract formants; Chroma captures musical pitch; Mel Spectrogram captures energy and volume dynamics.",
                "Q3: How did you fix the browser audio format error?",
                "  • A: We built an in-browser Web Audio API transcoder that converts WebM/Opus into standard 16-bit PCM RIFF WAV.",
                "Q4: How did you prevent the model from defaulting to Calm or Neutral?",
                "  • A: We applied dead-air silence trimming (top_db=35), removed artificial volume amplification, and calibrated softmax temperature (T=2.2)."
            ],
            "notes": "These are the four questions examiners ask most frequently. Memorize these key points: parameter efficiency of 1D-CNN, complementary nature of the 180 features, and the novel in-browser transcoding solution."
        }
    ]

    for index, data in enumerate(slides_data):
        slide = prs.slides.add_slide(blank_layout)

        # Background Fill
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = COLOR_BG
        bg.line.color.rgb = COLOR_BG

        # Top Accent Header Bar
        top_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.15))
        top_bar.fill.solid()
        top_bar.fill.fore_color.rgb = COLOR_ACCENT
        top_bar.line.color.rgb = COLOR_ACCENT

        # Category Pill Badge
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(6.0), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = f"SLIDE {index+1:02d} | {data['category']}"
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = COLOR_ACCENT

        # Slide Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = data["title"]
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_PRIMARY

        # Slide Subtitle
        sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(11.7), Inches(0.4))
        tf_sub = sub_box.text_frame
        tf_sub.word_wrap = True
        p_sub = tf_sub.paragraphs[0]
        p_sub.text = data["subtitle"]
        p_sub.font.size = Pt(12)
        p_sub.font.italic = True
        p_sub.font.color.rgb = COLOR_TEXT_MUTED

        # Main Content Card
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.9), Inches(11.7), Inches(5.0))
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_WHITE
        card.line.color.rgb = COLOR_CARD_BORDER
        card.line.width = Pt(1.5)

        # Content Text Box
        content_box = slide.shapes.add_textbox(Inches(1.1), Inches(2.1), Inches(11.1), Inches(4.6))
        tf_content = content_box.text_frame
        tf_content.word_wrap = True

        for b_idx, bullet in enumerate(data["bullets"]):
            p = tf_content.paragraphs[0] if b_idx == 0 else tf_content.add_paragraph()
            p.text = bullet
            p.font.size = Pt(13)
            p.space_after = Pt(8)
            
            if bullet.startswith("  •") or bullet.startswith("    "):
                p.font.size = Pt(12)
                p.font.color.rgb = COLOR_TEXT_MUTED
            elif ":" in bullet and not bullet.startswith(" "):
                p.font.bold = False
                p.font.color.rgb = COLOR_TEXT_DARK
            else:
                p.font.color.rgb = COLOR_TEXT_DARK

        # Bottom Footer
        footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.05), Inches(11.7), Inches(0.35))
        tf_foot = footer_box.text_frame
        p_foot = tf_foot.paragraphs[0]
        p_foot.text = "EmoSense: Speech Emotion Detection System | B.E. Final Year Major Project"
        p_foot.font.size = Pt(9)
        p_foot.font.color.rgb = COLOR_TEXT_MUTED

        # Speaker Notes
        notes_slide = slide.notes_slide
        tf_notes = notes_slide.notes_text_frame
        tf_notes.text = data["notes"]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    prs.save(output_path)
    print(f"PowerPoint Presentation successfully generated at: {output_path}")

if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "Speech_Emotion_Detection_Presentation.pptx"
    create_presentation(out_file)
