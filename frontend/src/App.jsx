import React, { useState, useRef, useEffect } from 'react';

// Emotion definitions matching RAVDESS & model classes
const EMOTIONS_CONFIG = {
  Happy: {
    emoji: '😊',
    color: 'from-amber-400 to-yellow-500',
    barColor: 'bg-yellow-400',
    bgSoft: 'bg-yellow-50 dark:bg-yellow-950/30',
    textColor: 'text-yellow-600 dark:text-yellow-400',
    description: 'Joyful, uplifting, and positive acoustic cadence.',
  },
  Sad: {
    emoji: '😢',
    color: 'from-blue-400 to-indigo-600',
    barColor: 'bg-blue-500',
    bgSoft: 'bg-blue-50 dark:bg-blue-950/30',
    textColor: 'text-blue-600 dark:text-blue-400',
    description: 'Subdued pitch, slower tempo, and somber acoustic traits.',
  },
  Fearful: {
    emoji: '😨',
    color: 'from-purple-400 to-indigo-700',
    barColor: 'bg-purple-500',
    bgSoft: 'bg-purple-50 dark:bg-purple-950/30',
    textColor: 'text-purple-600 dark:text-purple-400',
    description: 'Elevated pitch variation, tremor, and anxious pacing.',
  },
  Surprised: {
    emoji: '😲',
    color: 'from-pink-400 to-rose-500',
    barColor: 'bg-pink-500',
    bgSoft: 'bg-pink-50 dark:bg-pink-950/30',
    textColor: 'text-pink-600 dark:text-pink-400',
    description: 'Abrupt dynamic shifts with rapid pitch excursions.',
  },
  Neutral: {
    emoji: '😐',
    color: 'from-slate-400 to-gray-600',
    barColor: 'bg-slate-400',
    bgSoft: 'bg-slate-50 dark:bg-slate-800/40',
    textColor: 'text-slate-600 dark:text-slate-300',
    description: 'Balanced, steady speech tone with minimal pitch inflection.',
  },
  Angry: {
    emoji: '😠',
    color: 'from-red-400 to-rose-600',
    barColor: 'bg-red-500',
    bgSoft: 'bg-red-50 dark:bg-red-950/30',
    textColor: 'text-red-600 dark:text-red-400',
    description: 'High vocal intensity, sharp onset energy, and aggressive harmonics.',
  },
  Disgust: {
    emoji: '🤢',
    color: 'from-emerald-400 to-green-600',
    barColor: 'bg-emerald-500',
    bgSoft: 'bg-emerald-50 dark:bg-emerald-950/30',
    textColor: 'text-emerald-600 dark:text-emerald-400',
    description: 'Guttural resonance, drawn-out syllables, and harsh lower harmonics.',
  },
  Calm: {
    emoji: '😌',
    color: 'from-teal-400 to-cyan-600',
    barColor: 'bg-teal-500',
    bgSoft: 'bg-teal-50 dark:bg-teal-950/30',
    textColor: 'text-teal-600 dark:text-teal-400',
    description: 'Gentle, steady breathing rhythm and serene acoustic stability.',
  },
};

const LEFT_EMOTIONS = ['Happy', 'Sad', 'Fearful', 'Surprised'];
const RIGHT_EMOTIONS = ['Neutral', 'Angry', 'Disgust', 'Calm'];

// Convert any browser audio Blob (WebM/Opus) into standard 16-bit PCM mono WAV at 22050Hz
async function convertBlobToWav(blob, targetSr = 22050) {
  try {
    const arrayBuffer = await blob.arrayBuffer();
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) return blob;

    const audioCtx = new AudioContextClass();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    // Resample to targetSr mono using OfflineAudioContext
    const length = Math.max(1, Math.ceil(audioBuffer.duration * targetSr));
    const offlineCtx = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(
      1,
      length,
      targetSr
    );

    const source = offlineCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineCtx.destination);
    source.start(0);

    const renderedBuffer = await offlineCtx.startRendering();
    const channelData = renderedBuffer.getChannelData(0);

    // Create 16-bit PCM RIFF WAV buffer
    const wavBuffer = new ArrayBuffer(44 + channelData.length * 2);
    const view = new DataView(wavBuffer);

    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + channelData.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // Subchunk1Size
    view.setUint16(20, 1, true); // PCM format
    view.setUint16(22, 1, true); // Mono (1 channel)
    view.setUint32(24, targetSr, true); // Sample rate (22050)
    view.setUint32(28, targetSr * 2, true); // Byte rate (targetSr * 1 * 2)
    view.setUint16(32, 2, true); // Block align (1 * 2)
    view.setUint16(34, 16, true); // 16 bits per sample
    writeString(36, 'data');
    view.setUint32(40, channelData.length * 2, true);

    // Write PCM 16-bit samples
    let offset = 44;
    for (let i = 0; i < channelData.length; i++, offset += 2) {
      let s = Math.max(-1, Math.min(1, channelData[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    try {
      await audioCtx.close();
    } catch (_) {}

    return new Blob([view], { type: 'audio/wav' });
  } catch (err) {
    console.warn('In-browser WAV conversion failed, falling back to raw recording:', err);
    return blob;
  }
}

export default function App() {
  // Navigation & layout
  const [currentTab, setCurrentTab] = useState('home'); // 'home' | 'history' | 'about'
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [apiUrl, setApiUrl] = useState(() => {
    const stored = localStorage.getItem('emosense_api');
    // Clear out hardcoded http ip from stored if present on https
    if (stored && !stored.includes('16.171.150.161')) return stored;
    if (import.meta.env.VITE_API_URL) return import.meta.env.VITE_API_URL;
    return typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
      ? 'http://127.0.0.1:5000'
      : '';
  });

  // Audio input mode
  const [inputMode, setInputMode] = useState('record'); // 'record' | 'upload'
  const [isRecording, setIsRecording] = useState(false);
  const [recordDuration, setRecordDuration] = useState(0); // in seconds
  const maxRecordDuration = 8; // 8 seconds maximum

  // Audio data
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);

  // Analysis result
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [prediction, setPrediction] = useState(null); // { emotion, emoji, confidence, probabilities }
  const [history, setHistory] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('emosense_history') || '[]');
    } catch {
      return [];
    }
  });

  // Refs
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerIntervalRef = useRef(null);
  const audioElementRef = useRef(null);
  const fileInputRef = useRef(null);

  // Ensure light mode is default
  useEffect(() => {
    document.documentElement.classList.remove('dark');
  }, []);

  // Clean up audio URL on unmount/change
  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
    };
  }, [audioUrl]);

  // Save history
  useEffect(() => {
    localStorage.setItem('emosense_history', JSON.stringify(history));
  }, [history]);

  // Handle Recording
  const startRecording = async () => {
    setError('');
    setRecordedBlob(null);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
    setRecordDuration(0);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const rawBlob = new Blob(audioChunksRef.current, { type: mediaRecorder.mimeType || 'audio/webm' });
        const wavBlob = await convertBlobToWav(rawBlob, 22050);
        setRecordedBlob(wavBlob);
        const url = URL.createObjectURL(wavBlob);
        setAudioUrl(url);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start(200);
      setIsRecording(true);

      const startTime = Date.now();
      timerIntervalRef.current = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        setRecordDuration(elapsed);
        if (elapsed >= maxRecordDuration) {
          stopRecording();
        }
      }, 250);
    } catch (err) {
      console.error('Mic access error:', err);
      setError('Microphone access denied. Please allow microphone permissions in your browser.');
    }
  };

  const stopRecording = () => {
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  };

  const handleToggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  // Handle File Upload
  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setError('');
    setSelectedFile(file);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
  };

  // Play / Pause preview audio
  const togglePlayAudio = () => {
    if (!audioElementRef.current) return;
    if (isPlayingAudio) {
      audioElementRef.current.pause();
      setIsPlayingAudio(false);
    } else {
      audioElementRef.current.play();
      setIsPlayingAudio(true);
    }
  };

  // Trigger Prediction API
  const handleAnalyze = async () => {
    setError('');
    const audioToUpload = inputMode === 'record' ? recordedBlob : selectedFile;

    if (!audioToUpload) {
      setError('Please record audio or select an audio file first.');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    const isWav = audioToUpload.type === 'audio/wav' || audioToUpload.name?.toLowerCase().endsWith('.wav');
    const uploadName = inputMode === 'record'
      ? (isWav ? 'recording.wav' : 'recording.webm')
      : audioToUpload.name;
    formData.append('audio', audioToUpload, uploadName);

    try {
      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || data.error) {
        throw new Error(data.error || 'Failed to detect emotion');
      }

      // Format response
      let emotionName = data.emotion;
      if (!emotionName && data.predicted_emotion) {
        emotionName = data.predicted_emotion.split(' ')[0];
      }

      const cleanEmotion = emotionName
        ? emotionName.charAt(0).toUpperCase() + emotionName.slice(1).toLowerCase()
        : 'Neutral';

      const emotionEmoji = EMOTIONS_CONFIG[cleanEmotion]?.emoji || '😐';
      const confidence = data.confidence || 85;

      // Ensure all 8 probabilities exist
      let probs = data.probabilities || {};
      if (Object.keys(probs).length === 0) {
        // Fallback distribution if backend returns only label
        probs = {
          Happy: cleanEmotion === 'Happy' ? confidence : 2,
          Sad: cleanEmotion === 'Sad' ? confidence : 1,
          Fearful: cleanEmotion === 'Fearful' ? confidence : 1,
          Surprised: cleanEmotion === 'Surprised' ? confidence : 3,
          Neutral: cleanEmotion === 'Neutral' ? confidence : 4,
          Angry: cleanEmotion === 'Angry' ? confidence : 2,
          Disgust: cleanEmotion === 'Disgust' ? confidence : 1,
          Calm: cleanEmotion === 'Calm' ? confidence : 3,
        };
      }

      const result = {
        emotion: cleanEmotion,
        emoji: emotionEmoji,
        confidence: Math.round(confidence),
        probabilities: probs,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        source: inputMode === 'record' ? 'Voice Recording' : selectedFile?.name || 'Uploaded File',
      };

      setPrediction(result);
      setHistory((prev) => [result, ...prev.slice(0, 19)]);
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.message || 'Cannot connect to backend server. Make sure Flask is running on port 5000.');
    } finally {
      setLoading(false);
    }
  };

  const resetAnalysis = () => {
    setPrediction(null);
    setRecordedBlob(null);
    setSelectedFile(null);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
    setRecordDuration(0);
    setError('');
  };

  // Format seconds into 00:SS
  const formatTime = (secs) => {
    const s = Math.min(secs, maxRecordDuration);
    const mm = String(Math.floor(s / 60)).padStart(2, '0');
    const ss = String(s % 60).padStart(2, '0');
    return `${mm}:${ss}`;
  };

  // Current probabilities (either from prediction or 0%)
  const currentProbs = prediction?.probabilities || {};

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#F8FAFC] dark:bg-slate-950 text-slate-800 dark:text-slate-100 font-sans transition-colors duration-200">
      {/* Hidden Audio Player for Preview */}
      {audioUrl && (
        <audio
          ref={audioElementRef}
          src={audioUrl}
          onEnded={() => setIsPlayingAudio(false)}
          className="hidden"
        />
      )}

      {/* ===================== SIDEBAR ===================== */}
      <aside
        className={`transition-all duration-300 ease-in-out border-r border-slate-200/80 dark:border-slate-800/80 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md flex flex-col justify-between ${
          isSidebarCollapsed ? 'w-20 min-w-[80px] p-3 items-center' : 'w-60 min-w-[240px] p-5'
        }`}
      >
        <div className="w-full">
          {/* Logo & Minimize Toggle */}
          <div className="flex items-center justify-between px-1 py-1">
            <div className={`flex items-center gap-2.5 overflow-hidden ${isSidebarCollapsed ? 'justify-center w-full' : ''}`}>
              <div className="flex items-center gap-1 text-blue-600 dark:text-blue-500 flex-shrink-0">
                <span className="w-1 h-3.5 bg-blue-600 dark:bg-blue-500 rounded-full"></span>
                <span className="w-1 h-6 bg-blue-600 dark:bg-blue-500 rounded-full"></span>
                <span className="w-1 h-4 bg-blue-600 dark:bg-blue-500 rounded-full"></span>
                <span className="w-1 h-2 bg-blue-600 dark:bg-blue-500 rounded-full"></span>
              </div>
              {!isSidebarCollapsed && (
                <span className="text-xl font-bold tracking-tight text-slate-900 dark:text-white transition-opacity duration-200">
                  EmoSense
                </span>
              )}
            </div>

            {/* Collapse button on expanded state */}
            {!isSidebarCollapsed && (
              <button
                onClick={() => setIsSidebarCollapsed(true)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition"
                title="Minimize sidebar"
                aria-label="Minimize sidebar"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
                </svg>
              </button>
            )}
          </div>

          {/* Expand button on collapsed state */}
          {isSidebarCollapsed && (
            <div className="mt-3 flex justify-center w-full">
              <button
                onClick={() => setIsSidebarCollapsed(false)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition"
                title="Expand sidebar"
                aria-label="Expand sidebar"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                </svg>
              </button>
            </div>
          )}

          {/* Navigation Links (Analyze removed) */}
          <nav className="mt-8 space-y-1.5 w-full">
            <button
              onClick={() => setCurrentTab('home')}
              title="Home"
              className={`w-full flex items-center ${
                isSidebarCollapsed ? 'justify-center px-2 py-2.5' : 'gap-3 px-3.5 py-2.5'
              } rounded-xl text-sm font-medium transition ${
                currentTab === 'home'
                  ? 'bg-blue-50 dark:bg-blue-950/50 text-blue-600 dark:text-blue-400 font-semibold'
                  : 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/50 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
              </svg>
              {!isSidebarCollapsed && <span>Home</span>}
            </button>

            <button
              onClick={() => setCurrentTab('history')}
              title="History"
              className={`w-full flex items-center ${
                isSidebarCollapsed ? 'justify-center px-2 py-2.5' : 'gap-3 px-3.5 py-2.5'
              } rounded-xl text-sm font-medium transition ${
                currentTab === 'history'
                  ? 'bg-blue-50 dark:bg-blue-950/50 text-blue-600 dark:text-blue-400 font-semibold'
                  : 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/50 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {!isSidebarCollapsed && (
                <>
                  <span>History</span>
                  {history.length > 0 && (
                    <span className="ml-auto text-xs px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400">
                      {history.length}
                    </span>
                  )}
                </>
              )}
            </button>

            <button
              onClick={() => setCurrentTab('about')}
              title="About"
              className={`w-full flex items-center ${
                isSidebarCollapsed ? 'justify-center px-2 py-2.5' : 'gap-3 px-3.5 py-2.5'
              } rounded-xl text-sm font-medium transition ${
                currentTab === 'about'
                  ? 'bg-blue-50 dark:bg-blue-950/50 text-blue-600 dark:text-blue-400 font-semibold'
                  : 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/50 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {!isSidebarCollapsed && <span>About</span>}
            </button>
          </nav>
        </div>

        {/* Settings button pinned to bottom */}
        <div className="border-t border-slate-100 dark:border-slate-800/70 pt-3 w-full">
          <button
            onClick={() => setShowSettings(true)}
            title="Settings"
            className={`w-full flex items-center ${
              isSidebarCollapsed ? 'justify-center px-2 py-2.5' : 'gap-3 px-3.5 py-2.5'
            } rounded-xl text-sm font-medium text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/50 hover:text-slate-900 dark:hover:text-white transition`}
          >
            <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            {!isSidebarCollapsed && <span>Settings</span>}
          </button>
        </div>
      </aside>

      {/* ===================== MAIN CONTENT ===================== */}
      <main className="flex-1 flex flex-col h-full overflow-y-auto px-10 py-7">
        {/* Top Header (Day/Night Theme Logo Removed) */}
        <header className="flex items-center justify-between pb-5">
          <div className="flex items-center gap-3">
            {isSidebarCollapsed && (
              <button
                onClick={() => setIsSidebarCollapsed(false)}
                className="p-2 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 shadow-sm hover:bg-slate-50 dark:hover:bg-slate-800 transition"
                title="Expand sidebar"
                aria-label="Expand sidebar"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
            )}
            <div>
              <h1 className="text-2xl lg:text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white">
                Speech Emotion Detection
              </h1>
              <p className="text-sm font-normal text-slate-500 dark:text-slate-400 mt-1">
                Analyze emotions from speech using AI
              </p>
            </div>
          </div>
        </header>

        {/* Global Error Banner */}
        {error && (
          <div className="mb-5 p-3.5 rounded-xl bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900/60 text-red-600 dark:text-red-300 text-sm flex items-center justify-between">
            <div className="flex items-center gap-2">
              <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>{error}</span>
            </div>
            <button onClick={() => setError('')} className="text-red-400 hover:text-red-600 text-xs font-semibold">
              Dismiss
            </button>
          </div>
        )}

        {/* Tab: History */}
        {currentTab === 'history' && (
          <div className="bg-white dark:bg-slate-900 rounded-2xl p-6 border border-slate-200/80 dark:border-slate-800 shadow-sm mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-bold text-slate-900 dark:text-white">Analysis History</h2>
              {history.length > 0 && (
                <button
                  onClick={() => setHistory([])}
                  className="text-xs text-red-500 hover:text-red-600 font-semibold"
                >
                  Clear History
                </button>
              )}
            </div>
            {history.length === 0 ? (
              <p className="text-sm text-slate-400 py-6 text-center">No past analyses recorded yet.</p>
            ) : (
              <div className="divide-y divide-slate-100 dark:divide-slate-800">
                {history.map((item, index) => (
                  <div key={index} className="py-3 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">{item.emoji}</span>
                      <div>
                        <div className="text-sm font-semibold text-slate-800 dark:text-slate-200">
                          {item.emotion}
                          <span className="ml-2 text-xs font-normal text-slate-400">({item.confidence}%)</span>
                        </div>
                        <div className="text-xs text-slate-400">{item.source} • {item.timestamp}</div>
                      </div>
                    </div>
                    <button
                      onClick={() => {
                        setPrediction(item);
                        setCurrentTab('home');
                      }}
                      className="text-xs text-blue-600 hover:text-blue-700 font-medium"
                    >
                      View
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Tab: About */}
        {currentTab === 'about' && (
          <div className="bg-white dark:bg-slate-900 rounded-2xl p-6 border border-slate-200/80 dark:border-slate-800 shadow-sm mb-6 space-y-4">
            <h2 className="text-lg font-bold text-slate-900 dark:text-white">About Speech Emotion Detection</h2>
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
              This system uses deep learning to classify human emotional states directly from speech audio. It is trained on the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) corpus featuring 24 professional vocal actors.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
              <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800">
                <div className="text-xs font-bold uppercase tracking-wider text-blue-600 dark:text-blue-400">1. Audio Features</div>
                <div className="text-sm font-medium text-slate-700 dark:text-slate-200 mt-1">MFCC (40) + Chroma (12) + Mel (128)</div>
                <p className="text-xs text-slate-400 mt-1">Total of 180 acoustic coefficients extracted per sample using Librosa.</p>
              </div>
              <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800">
                <div className="text-xs font-bold uppercase tracking-wider text-blue-600 dark:text-blue-400">2. Deep Model</div>
                <div className="text-sm font-medium text-slate-700 dark:text-slate-200 mt-1">1D Convolutional Neural Network</div>
                <p className="text-xs text-slate-400 mt-1">Stacked 1D Conv layers with dropout regularization and softmax classification.</p>
              </div>
              <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800">
                <div className="text-xs font-bold uppercase tracking-wider text-blue-600 dark:text-blue-400">3. 8 Emotion Classes</div>
                <div className="text-sm font-medium text-slate-700 dark:text-slate-200 mt-1">Neutral, Calm, Happy, Sad...</div>
                <p className="text-xs text-slate-400 mt-1">Also includes Angry, Fearful, Disgust, and Surprised with confidence scoring.</p>
              </div>
            </div>
          </div>
        )}

        {/* ===================== TOP TWO CARDS ===================== */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* CARD 1: 1. Input Audio */}
          <div className="bg-white dark:bg-slate-900 rounded-2xl p-6 border border-slate-200/80 dark:border-slate-800 shadow-sm flex flex-col justify-between">
            <div>
              {/* Header */}
              <h2 className="text-base font-bold text-slate-900 dark:text-white">1. Input Audio</h2>
              <p className="text-xs text-slate-400 dark:text-slate-500 mt-0.5">
                Record your voice or upload an audio file
              </p>

              {/* Segmented Switch: Record | Upload */}
              <div className="mt-5 mb-6 flex justify-center">
                <div className="inline-flex p-1 rounded-full bg-slate-100 dark:bg-slate-800/90 border border-slate-200/50 dark:border-slate-700/50">
                  <button
                    onClick={() => {
                      setInputMode('record');
                      setError('');
                    }}
                    className={`flex items-center gap-2 px-6 py-1.5 rounded-full text-xs font-semibold transition ${
                      inputMode === 'record'
                        ? 'bg-blue-50 dark:bg-blue-900/60 text-blue-600 dark:text-blue-300 shadow-sm'
                        : 'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-white'
                    }`}
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 100-6 3 3 0 000 6z" />
                    </svg>
                    <span>Record</span>
                  </button>

                  <button
                    onClick={() => {
                      setInputMode('upload');
                      setError('');
                    }}
                    className={`flex items-center gap-2 px-6 py-1.5 rounded-full text-xs font-semibold transition ${
                      inputMode === 'upload'
                        ? 'bg-blue-50 dark:bg-blue-900/60 text-blue-600 dark:text-blue-300 shadow-sm'
                        : 'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-white'
                    }`}
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                    </svg>
                    <span>Upload</span>
                  </button>
                </div>
              </div>

              {/* Mode: RECORD */}
              {inputMode === 'record' && (
                <div className="flex flex-col items-center justify-center py-2">
                  {/* Big Circular Microphone Button with concentric circles */}
                  <div className="relative flex items-center justify-center mb-4">
                    {/* Outer animated ripple glow */}
                    <div
                      className={`absolute w-32 h-32 rounded-full transition-all duration-700 ${
                        isRecording
                          ? 'bg-red-100 dark:bg-red-950/40 animate-ping'
                          : 'bg-blue-50 dark:bg-blue-950/30'
                      }`}
                    ></div>

                    {/* Middle soft circle */}
                    <div
                      className={`relative w-24 h-24 rounded-full flex items-center justify-center transition ${
                        isRecording
                          ? 'bg-red-50 dark:bg-red-900/30 ring-8 ring-red-100 dark:ring-red-950/50'
                          : 'bg-blue-50 dark:bg-blue-950/50 ring-8 ring-blue-50/70 dark:ring-blue-900/30'
                      }`}
                    >
                      {/* Inner clickable button */}
                      <button
                        onClick={handleToggleRecording}
                        className={`w-14 h-14 rounded-full flex items-center justify-center text-white shadow-lg transition-transform duration-200 transform hover:scale-105 active:scale-95 ${
                          isRecording
                            ? 'bg-red-600 hover:bg-red-700 shadow-red-500/40 animate-pulse'
                            : 'bg-blue-600 hover:bg-blue-700 shadow-blue-500/30'
                        }`}
                        title={isRecording ? 'Click to stop recording' : 'Click to start recording'}
                        aria-label={isRecording ? 'Stop recording' : 'Start recording'}
                      >
                        {isRecording ? (
                          <span className="w-5 h-5 bg-white rounded-sm"></span>
                        ) : (
                          <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
                            <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
                          </svg>
                        )}
                      </button>
                    </div>
                  </div>

                  {/* Title & subtitle under mic */}
                  <div className="text-center">
                    <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">
                      {isRecording ? 'Recording in progress...' : recordedBlob ? 'Recording complete' : 'Click to start recording'}
                    </p>
                    <p className="text-xs text-slate-400 dark:text-slate-500 mt-0.5">
                      {isRecording ? 'Speak clearly into your microphone' : `Record up to ${maxRecordDuration} seconds`}
                    </p>
                  </div>

                  {/* Timer display */}
                  <div className="mt-4 font-mono text-xs font-semibold text-slate-600 dark:text-slate-400 tracking-wider">
                    {formatTime(recordDuration)} / {formatTime(maxRecordDuration)}
                  </div>

                  {/* Soundwave Bars & Playback Controller */}
                  <div className="mt-4 w-full max-w-sm flex items-center justify-between gap-3 px-4 py-2.5 bg-slate-50 dark:bg-slate-800/40 rounded-xl border border-slate-100 dark:border-slate-800">
                    {/* Waveform graphic */}
                    <div className="flex-1 flex items-center justify-center gap-1 h-6">
                      {[6, 12, 18, 22, 15, 10, 24, 16, 20, 14, 8, 16, 20, 26, 18, 12, 22, 16, 10, 6].map(
                        (h, i) => (
                          <span
                            key={i}
                            className={`w-1 rounded-full transition-all duration-150 ${
                              isRecording
                                ? 'bg-blue-600 dark:bg-blue-400'
                                : recordedBlob
                                ? 'bg-slate-400 dark:bg-slate-500'
                                : 'bg-slate-200 dark:bg-slate-700'
                            }`}
                            style={{
                              height: isRecording ? `${Math.max(4, (h * (i % 3 + 1)) % 24)}px` : `${h}px`,
                            }}
                          ></span>
                        )
                      )}
                    </div>

                    {/* Play/Pause Button */}
                    <button
                      onClick={togglePlayAudio}
                      disabled={!audioUrl}
                      className={`w-7 h-7 rounded-full flex items-center justify-center transition ${
                        audioUrl
                          ? 'bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-300 dark:hover:bg-slate-600'
                          : 'bg-slate-100 dark:bg-slate-800 text-slate-300 dark:text-slate-600 cursor-not-allowed'
                      }`}
                      title={isPlayingAudio ? 'Pause preview' : 'Play preview'}
                    >
                      {isPlayingAudio ? (
                        <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
                        </svg>
                      ) : (
                        <svg className="w-3.5 h-3.5 ml-0.5" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M8 5v14l11-7z" />
                        </svg>
                      )}
                    </button>
                  </div>
                </div>
              )}

              {/* Mode: UPLOAD */}
              {inputMode === 'upload' && (
                <div className="flex flex-col items-center justify-center py-4">
                  <input
                    type="file"
                    ref={fileInputRef}
                    accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                  <div
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full max-w-sm border-2 border-dashed border-slate-200 dark:border-slate-700 hover:border-blue-500 dark:hover:border-blue-400 rounded-2xl p-6 flex flex-col items-center justify-center cursor-pointer transition bg-slate-50/50 dark:bg-slate-800/30 hover:bg-blue-50/30 dark:hover:bg-blue-950/20"
                  >
                    <div className="w-12 h-12 rounded-full bg-blue-50 dark:bg-blue-900/40 text-blue-600 dark:text-blue-400 flex items-center justify-center mb-3">
                      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                      </svg>
                    </div>
                    <p className="text-sm font-semibold text-slate-800 dark:text-slate-200 text-center">
                      {selectedFile ? selectedFile.name : 'Click to select or drag audio file'}
                    </p>
                    <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">
                      {selectedFile
                        ? `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB • Ready to analyze`
                        : 'Supports WAV, MP3, FLAC, OGG'}
                    </p>
                  </div>

                  {/* Play audio preview if selected */}
                  {selectedFile && audioUrl && (
                    <div className="mt-4 flex items-center gap-2">
                      <button
                        onClick={togglePlayAudio}
                        className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-700"
                      >
                        {isPlayingAudio ? 'Pause Audio' : 'Preview Audio'}
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Action Button: Analyze Audio */}
            <div className="mt-5 pt-3 border-t border-slate-100 dark:border-slate-800/70">
              <button
                onClick={handleAnalyze}
                disabled={loading || (inputMode === 'record' ? !recordedBlob : !selectedFile)}
                className={`w-full py-2.5 px-4 rounded-xl text-xs font-semibold uppercase tracking-wider flex items-center justify-center gap-2 shadow-sm transition ${
                  loading || (inputMode === 'record' ? !recordedBlob : !selectedFile)
                    ? 'bg-slate-100 dark:bg-slate-800 text-slate-400 dark:text-slate-600 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700 text-white shadow-blue-500/20 active:scale-[0.99]'
                }`}
              >
                {loading ? (
                  <>
                    <svg className="animate-spin w-4 h-4 text-white" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
                    </svg>
                    <span>Analyzing Speech...</span>
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    <span>Detect Emotion</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* CARD 2: 2. Emotion Result */}
          <div className="bg-white dark:bg-slate-900 rounded-2xl p-6 border border-slate-200/80 dark:border-slate-800 shadow-sm flex flex-col justify-between">
            <div>
              <h2 className="text-base font-bold text-slate-900 dark:text-white">2. Emotion Result</h2>
              <p className="text-xs text-slate-400 dark:text-slate-500 mt-0.5">
                AI analysis result will appear here
              </p>
            </div>

            {/* Content: Empty State VS Predicted State */}
            {!prediction ? (
              /* Empty State matching reference image */
              <div className="flex-1 flex flex-col items-center justify-center py-12 text-center">
                {/* Subtle vertical soundbar graphic */}
                <div className="flex items-center gap-1 text-slate-300 dark:text-slate-700 mb-4">
                  <span className="w-1.5 h-4 bg-slate-300 dark:bg-slate-700 rounded-full"></span>
                  <span className="w-1.5 h-8 bg-slate-300 dark:bg-slate-700 rounded-full"></span>
                  <span className="w-1.5 h-12 bg-slate-300 dark:bg-slate-700 rounded-full"></span>
                  <span className="w-1.5 h-6 bg-slate-300 dark:bg-slate-700 rounded-full"></span>
                  <span className="w-1.5 h-3 bg-slate-300 dark:bg-slate-700 rounded-full"></span>
                </div>
                <h3 className="text-base font-semibold text-slate-700 dark:text-slate-200">
                  No analysis yet
                </h3>
                <p className="text-xs text-slate-400 dark:text-slate-500 mt-1 max-w-xs leading-relaxed">
                  Record or upload an audio file to detect emotion
                </p>
              </div>
            ) : (
              /* Detected Emotion Display */
              <div className="flex-1 flex flex-col items-center justify-center py-6 text-center">
                {/* Emotion Badge / Avatar */}
                <div className="relative mb-3">
                  <div className="text-7xl animate-bounce duration-1000 select-none">
                    {prediction.emoji}
                  </div>
                </div>

                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-blue-50 dark:bg-blue-950/60 text-blue-600 dark:text-blue-400 mb-2">
                  <span>Confidence: {prediction.confidence}%</span>
                </div>

                <h3 className="text-2xl font-black text-slate-900 dark:text-white tracking-tight">
                  {prediction.emotion}
                </h3>

                <p className="text-xs text-slate-500 dark:text-slate-400 mt-2 max-w-xs leading-relaxed">
                  {EMOTIONS_CONFIG[prediction.emotion]?.description || 'Acoustic analysis complete.'}
                </p>

                <div className="mt-5 flex items-center gap-3">
                  <button
                    onClick={resetAnalysis}
                    className="px-4 py-1.5 rounded-lg text-xs font-medium border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 transition"
                  >
                    Analyze Another
                  </button>
                  {audioUrl && (
                    <button
                      onClick={togglePlayAudio}
                      className="px-4 py-1.5 rounded-lg text-xs font-medium bg-blue-50 dark:bg-blue-950 text-blue-600 dark:text-blue-400 hover:bg-blue-100 dark:hover:bg-blue-900 transition flex items-center gap-1.5"
                    >
                      <span>{isPlayingAudio ? 'Pause' : 'Replay Audio'}</span>
                    </button>
                  )}
                </div>
              </div>
            )}

            {/* Bottom footnote */}
            <div className="pt-3 border-t border-slate-100 dark:border-slate-800/70 text-center">
              <span className="text-[11px] text-slate-400 dark:text-slate-500">
                Classification via 1D-CNN • RAVDESS Architecture
              </span>
            </div>
          </div>
        </div>

        {/* ===================== BOTTOM CARD: EMOTION PROBABILITIES ===================== */}
        <div className="mt-6 bg-white dark:bg-slate-900 rounded-2xl p-6 border border-slate-200/80 dark:border-slate-800 shadow-sm">
          {/* Header */}
          <div className="mb-5">
            <h2 className="text-base font-bold text-slate-900 dark:text-white">Emotion Probabilities</h2>
            <p className="text-xs text-slate-400 dark:text-slate-500 mt-0.5">
              Shows confidence for each emotion
            </p>
          </div>

          {/* 2-Column Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-4">
            {/* Left Column: Happy, Sad, Fearful, Surprised */}
            <div className="space-y-4">
              {LEFT_EMOTIONS.map((emotion) => {
                const config = EMOTIONS_CONFIG[emotion];
                const pct = Math.round(currentProbs[emotion] ?? 0);

                return (
                  <div key={emotion} className="flex items-center gap-3">
                    {/* Emoji Avatar */}
                    <div className="w-7 h-7 flex-shrink-0 flex items-center justify-center text-lg select-none">
                      {config.emoji}
                    </div>

                    {/* Emotion Name */}
                    <div className="w-20 text-xs font-medium text-slate-700 dark:text-slate-300">
                      {emotion}
                    </div>

                    {/* Progress Bar Track */}
                    <div className="flex-1 h-2 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-700 ease-out ${config.barColor}`}
                        style={{ width: `${pct}%` }}
                      ></div>
                    </div>

                    {/* Percentage */}
                    <div className="w-10 text-right font-mono text-xs font-semibold text-slate-500 dark:text-slate-400">
                      {pct}%
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Right Column: Neutral, Angry, Disgust, Calm */}
            <div className="space-y-4">
              {RIGHT_EMOTIONS.map((emotion) => {
                const config = EMOTIONS_CONFIG[emotion];
                const pct = Math.round(currentProbs[emotion] ?? 0);

                return (
                  <div key={emotion} className="flex items-center gap-3">
                    {/* Emoji Avatar */}
                    <div className="w-7 h-7 flex-shrink-0 flex items-center justify-center text-lg select-none">
                      {config.emoji}
                    </div>

                    {/* Emotion Name */}
                    <div className="w-20 text-xs font-medium text-slate-700 dark:text-slate-300">
                      {emotion}
                    </div>

                    {/* Progress Bar Track */}
                    <div className="flex-1 h-2 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-700 ease-out ${config.barColor}`}
                        style={{ width: `${pct}%` }}
                      ></div>
                    </div>

                    {/* Percentage */}
                    <div className="w-10 text-right font-mono text-xs font-semibold text-slate-500 dark:text-slate-400">
                      {pct}%
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </main>

      {/* ===================== SETTINGS MODAL ===================== */}
      {showSettings && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm p-4">
          <div className="bg-white dark:bg-slate-900 rounded-2xl shadow-2xl max-w-md w-full p-6 border border-slate-200 dark:border-slate-800">
            <div className="flex items-center justify-between pb-4 border-b border-slate-100 dark:border-slate-800">
              <h3 className="text-base font-bold text-slate-900 dark:text-white">Settings</h3>
              <button
                onClick={() => setShowSettings(false)}
                className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 text-sm"
              >
                ✕
              </button>
            </div>

            <div className="py-4 space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                  Backend API URL
                </label>
                <input
                  type="text"
                  value={apiUrl}
                  onChange={(e) => {
                    setApiUrl(e.target.value);
                    localStorage.setItem('emosense_api', e.target.value);
                  }}
                  placeholder="http://127.0.0.1:5000"
                  className="w-full text-xs px-3 py-2 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <p className="text-[11px] text-slate-400 mt-1">
                  Point to local server or your deployed cloud backend (e.g. Hugging Face Space).
                </p>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                  Clear Stored Data
                </label>
                <button
                  onClick={() => {
                    localStorage.clear();
                    setHistory([]);
                    alert('Local history and preferences cleared.');
                  }}
                  className="px-3 py-1.5 text-xs rounded-lg border border-red-200 dark:border-red-900 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/30"
                >
                  Clear All History & Cache
                </button>
              </div>
            </div>

            <div className="pt-3 border-t border-slate-100 dark:border-slate-800 flex justify-end">
              <button
                onClick={() => setShowSettings(false)}
                className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold shadow-sm"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
