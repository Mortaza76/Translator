import { useState, useEffect, useRef } from 'react'
import { Mic, MicOff, Play, Loader2, Languages, Volume2, CheckCircle2, AlertCircle, MessageSquare, RotateCcw, User } from 'lucide-react'
import axios from 'axios'
import './App.css'

const API_BASE = '/api'

function App() {
  // Mode: 'single' or 'conversation'
  const [mode, setMode] = useState('single') // 'single' or 'conversation'
  
  // Conversation state
  const [currentSpeaker, setCurrentSpeaker] = useState('personA') // 'personA' (Urdu) or 'personB' (Target)
  const [conversationHistory, setConversationHistory] = useState([])
  
  // Recording state
  const [isRecording, setIsRecording] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [status, setStatus] = useState('idle')
  const [statusMessage, setStatusMessage] = useState('')
  const [recordingDuration, setRecordingDuration] = useState(0)
  
  // Results state
  const [sourceText, setSourceText] = useState('')
  const [translatedText, setTranslatedText] = useState('')
  const [targetLanguage, setTargetLanguage] = useState('English')
  const [languages, setLanguages] = useState([])
  const [audioUrl, setAudioUrl] = useState(null)

  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const streamRef = useRef(null)
  const durationIntervalRef = useRef(null)

  useEffect(() => {
    loadLanguages()
  }, [])

  const loadLanguages = async () => {
    try {
      const response = await axios.get(`${API_BASE}/languages`)
      const sorted = response.data.languages
        .filter(lang => ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'ar', 'hi'].includes(lang.code))
        .sort((a, b) => a.name.localeCompare(b.name))
      setLanguages(sorted)
    } catch (error) {
      console.error('Failed to load languages:', error)
    }
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      })
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = async () => {
        await processRecording()
      }

      mediaRecorder.start()
      setIsRecording(true)
      setStatus('recording')
      
      // Set appropriate message based on mode and speaker
      if (mode === 'conversation') {
        if (currentSpeaker === 'personA') {
          setStatusMessage('Recording... Person A speak in Urdu')
        } else {
          setStatusMessage(`Recording... Person B speak in ${targetLanguage}`)
        }
      } else {
        setStatusMessage('Recording... Speak in Urdu')
      }
      
      setRecordingDuration(0)

      durationIntervalRef.current = setInterval(() => {
        setRecordingDuration(prev => prev + 1)
      }, 1000)

    } catch (error) {
      console.error('Error starting recording:', error)
      setStatus('error')
      setStatusMessage('Failed to access microphone. Please allow microphone access.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      setStatus('processing')
      setStatusMessage('Processing audio...')

      if (durationIntervalRef.current) {
        clearInterval(durationIntervalRef.current)
        durationIntervalRef.current = null
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
        streamRef.current = null
      }
    }
  }

  const convertWebmToWav = async (webmBlob) => {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)()
    const arrayBuffer = await webmBlob.arrayBuffer()
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer)
    const wav = audioBufferToWav(audioBuffer)
    return new Blob([wav], { type: 'audio/wav' })
  }

  const audioBufferToWav = (buffer) => {
    const length = buffer.length
    const numberOfChannels = buffer.numberOfChannels
    const sampleRate = buffer.sampleRate
    const arrayBuffer = new ArrayBuffer(44 + length * numberOfChannels * 2)
    const view = new DataView(arrayBuffer)

    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i))
      }
    }

    writeString(0, 'RIFF')
    view.setUint32(4, 36 + length * numberOfChannels * 2, true)
    writeString(8, 'WAVE')
    writeString(12, 'fmt ')
    view.setUint32(16, 16, true)
    view.setUint16(20, 1, true)
    view.setUint16(22, numberOfChannels, true)
    view.setUint32(24, sampleRate, true)
    view.setUint32(28, sampleRate * numberOfChannels * 2, true)
    view.setUint16(32, numberOfChannels * 2, true)
    view.setUint16(34, 16, true)
    writeString(36, 'data')
    view.setUint32(40, length * numberOfChannels * 2, true)

    let offset = 44
    for (let i = 0; i < length; i++) {
      for (let channel = 0; channel < numberOfChannels; channel++) {
        const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]))
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true)
        offset += 2
      }
    }

    return arrayBuffer
  }

  const processRecording = async () => {
    try {
      setIsProcessing(true)
      setStatus('processing')

      const webmBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
      const wavBlob = await convertWebmToWav(webmBlob)
      const reader = new FileReader()
      reader.readAsDataURL(wavBlob)
      
      reader.onloadend = async () => {
        const base64Audio = reader.result

        try {
          if (mode === 'conversation') {
            // Use conversation endpoint
            const sourceLang = currentSpeaker === 'personA' ? 'Urdu' : targetLanguage
            const targetLang = currentSpeaker === 'personA' ? targetLanguage : 'Urdu'
            
            setStatusMessage(`Transcribing ${sourceLang}...`)
            
            const response = await axios.post(`${API_BASE}/conversation`, {
              audio_data: base64Audio,
              source_language: sourceLang,
              target_language: targetLang,
              conversation_mode: true
            })

            if (response.data.status === 'success') {
              const entry = {
                id: Date.now(),
                speaker: currentSpeaker,
                sourceText: response.data.source_text,
                sourceLanguage: response.data.source_language,
                translatedText: response.data.translated_text,
                targetLanguage: response.data.target_language,
                audioBase64: response.data.audio_base64,
                timestamp: new Date()
              }
              
              setConversationHistory(prev => [...prev, entry])
              setSourceText(response.data.source_text)
              setTranslatedText(response.data.translated_text)
              setAudioUrl(response.data.audio_base64)
              
              // Auto-play translated audio
              playAudioFromBase64(response.data.audio_base64)
              
              // Switch speaker for next turn
              const nextSpeaker = currentSpeaker === 'personA' ? 'personB' : 'personA'
              setCurrentSpeaker(nextSpeaker)
              
              setStatus('success')
              setStatusMessage(`Translation complete! ${nextSpeaker === 'personA' ? 'Person A' : 'Person B'} can speak now.`)
            } else {
              throw new Error('Processing failed')
            }
          } else {
            // Single mode - use original endpoint
            setStatusMessage('Transcribing Urdu speech...')
            
            const response = await axios.post(`${API_BASE}/process-base64`, {
              audio_data: base64Audio,
              target_language: targetLanguage
            })

            if (response.data.status === 'success') {
              setSourceText(response.data.urdu_text)
              setTranslatedText(response.data.translated_text)
              setAudioUrl(response.data.audio_base64)
              setStatus('success')
              setStatusMessage('Translation complete!')
            } else {
              throw new Error('Processing failed')
            }
          }
        } catch (error) {
          console.error('Processing error:', error)
          setStatus('error')
          setStatusMessage(error.response?.data?.detail || 'Processing failed. Please try again.')
        } finally {
          setIsProcessing(false)
        }
      }
    } catch (error) {
      console.error('Error processing recording:', error)
      setStatus('error')
      setStatusMessage('Failed to process audio. Please try again.')
      setIsProcessing(false)
    }
  }

  const playAudioFromBase64 = (base64Audio) => {
    if (base64Audio) {
      const audio = new Audio(base64Audio)
      audio.play()
    }
  }

  const playAudio = () => {
    if (audioUrl) {
      const audio = new Audio(audioUrl)
      audio.play()
    }
  }

  const clearConversation = () => {
    setConversationHistory([])
    setSourceText('')
    setTranslatedText('')
    setAudioUrl(null)
    setCurrentSpeaker('personA')
  }

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1 className="title">
            <Languages className="title-icon" />
            Urdu Speech Translation
          </h1>
          <p className="subtitle">
            {mode === 'conversation' ? 'Two-way conversation translation' : 'Speak in Urdu, hear in any language'}
          </p>
        </header>

        <div className="main-content">
          {/* Mode Toggle */}
          <div className="mode-toggle">
            <button
              className={`mode-button ${mode === 'single' ? 'active' : ''}`}
              onClick={() => {
                setMode('single')
                setCurrentSpeaker('personA')
                clearConversation()
              }}
              disabled={isRecording || isProcessing}
            >
              Single Translation
            </button>
            <button
              className={`mode-button ${mode === 'conversation' ? 'active' : ''}`}
              onClick={() => {
                setMode('conversation')
                setCurrentSpeaker('personA')
                clearConversation()
              }}
              disabled={isRecording || isProcessing}
            >
              <MessageSquare className="icon-small" />
              Conversation Mode
            </button>
          </div>

          {/* Language Selector */}
          <div className="language-selector">
            <label htmlFor="target-language">Translate to:</label>
            <select
              id="target-language"
              value={targetLanguage}
              onChange={(e) => setTargetLanguage(e.target.value)}
              disabled={isRecording || isProcessing}
              className="select"
            >
              {languages.map(lang => (
                <option key={lang.code} value={lang.name}>
                  {lang.name}
                </option>
              ))}
            </select>
          </div>

          {/* Conversation Mode Indicators */}
          {mode === 'conversation' && (
            <div className="speaker-indicator">
              <div className={`speaker-card ${currentSpeaker === 'personA' ? 'active' : ''}`}>
                <User className="icon-small" />
                <div>
                  <div className="speaker-name">Person A (Urdu)</div>
                  <div className="speaker-status">
                    {currentSpeaker === 'personA' ? 'Your turn to speak' : 'Waiting...'}
                  </div>
                </div>
              </div>
              <div className="arrow">↔</div>
              <div className={`speaker-card ${currentSpeaker === 'personB' ? 'active' : ''}`}>
                <User className="icon-small" />
                <div>
                  <div className="speaker-name">Person B ({targetLanguage})</div>
                  <div className="speaker-status">
                    {currentSpeaker === 'personB' ? 'Your turn to speak' : 'Waiting...'}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Recording Section */}
          <div className="recording-section">
            <button
              className={`record-button ${isRecording ? 'recording' : ''} ${isProcessing ? 'processing' : ''}`}
              onClick={isRecording ? stopRecording : startRecording}
              disabled={isProcessing}
            >
              {isProcessing ? (
                <Loader2 className="icon spinning" />
              ) : isRecording ? (
                <MicOff className="icon" />
              ) : (
                <Mic className="icon" />
              )}
              <span>
                {isProcessing ? 'Processing...' : isRecording ? 'Stop Recording' : 'Start Recording'}
              </span>
            </button>

            {isRecording && (
              <div className="recording-indicator">
                <div className="pulse"></div>
                <span>{formatDuration(recordingDuration)}</span>
              </div>
            )}
          </div>

          {/* Status Message */}
          {statusMessage && (
            <div className={`status-message ${status}`}>
              {status === 'success' && <CheckCircle2 className="status-icon" />}
              {status === 'error' && <AlertCircle className="status-icon" />}
              {status === 'processing' && <Loader2 className="status-icon spinning" />}
              <span>{statusMessage}</span>
            </div>
          )}

          {/* Conversation History */}
          {mode === 'conversation' && conversationHistory.length > 0 && (
            <div className="conversation-history">
              <div className="history-header">
                <h3>Conversation History</h3>
                <button className="clear-button" onClick={clearConversation}>
                  <RotateCcw className="icon-small" />
                  Clear
                </button>
              </div>
              <div className="history-list">
                {conversationHistory.map((entry) => (
                  <div key={entry.id} className="history-entry">
                    <div className="history-speaker">
                      {entry.speaker === 'personA' ? 'Person A (Urdu)' : `Person B (${entry.targetLanguage})`}
                    </div>
                    <div className="history-source">{entry.sourceText}</div>
                    <div className="history-translation">{entry.translatedText}</div>
                    <button
                      className="history-play-button"
                      onClick={() => playAudioFromBase64(entry.audioBase64)}
                    >
                      <Volume2 className="icon-small" />
                      Play
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Results Section (Single Mode) */}
          {mode === 'single' && (sourceText || translatedText) && (
            <div className="results-section">
              {sourceText && (
                <div className="result-card">
                  <div className="result-header">
                    <span className="result-label">Urdu Text</span>
                  </div>
                  <div className="result-text urdu-text">{sourceText}</div>
                </div>
              )}

              {translatedText && (
                <div className="result-card">
                  <div className="result-header">
                    <span className="result-label">Translation ({targetLanguage})</span>
                  </div>
                  <div className="result-text">{translatedText}</div>
                </div>
              )}

              {audioUrl && (
                <button className="play-button" onClick={playAudio}>
                  <Volume2 className="icon" />
                  <span>Play Translation</span>
                </button>
              )}
            </div>
          )}

          {/* Instructions */}
          <div className="instructions">
            <h3>How to use:</h3>
            {mode === 'conversation' ? (
              <ol>
                <li>Select your target language</li>
                <li>Person A clicks "Start Recording" and speaks in Urdu</li>
                <li>Person B hears the translation in {targetLanguage}</li>
                <li>Person B clicks "Start Recording" and speaks in {targetLanguage}</li>
                <li>Person A hears the translation in Urdu</li>
                <li>Continue the conversation back and forth</li>
              </ol>
            ) : (
              <ol>
                <li>Select your target language</li>
                <li>Click "Start Recording" and speak in Urdu</li>
                <li>Click "Stop Recording" when finished</li>
                <li>Wait for transcription, translation, and audio generation</li>
                <li>Play the translated audio</li>
              </ol>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
