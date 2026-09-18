import { useEffect, useRef, useState } from 'react'

const INITIAL_GRID_POSITION = {
  label: '위치 인식 중',
  grid: { cols: 14, rows: 4 },
  landmarks: [
    { key: 'room4', label: '4강의실', grid_cell: [3, 0] },
    { key: 'room3', label: '3강의실', grid_cell: [8, 0] },
    { key: 'room2', label: '2강의실', grid_cell: [14, 0] },
    { key: 'front_door', label: '앞문', grid_cell: [0, 4] },
    { key: 'rear_door', label: '뒷문', grid_cell: [14, 4] },
    { key: 'logo', label: '로고', grid_cell: [0, 0] },
  ],
}

function GridMap({ position }) {
  const mapPosition = position?.grid ? position : INITIAL_GRID_POSITION
  const grid = mapPosition.grid
  if (!grid?.cols || !grid?.rows) return null
  const pointStyle = (cell) => ({
    left: `${(cell[0] / grid.cols) * 100}%`,
    top: `${100 - (cell[1] / grid.rows) * 100}%`,
  })
  return (
    <section className="position-grid" aria-label="현재 위치 격자 지도">
      {/* <div className="position-grid-heading">
        <span>현재 위치 지도</span>
        <b>{mapPosition.label}</b>
      </div> */}
      <div
        className="grid-map"
        style={{ '--grid-cols': grid.cols, '--grid-rows': grid.rows }}
      >
        {mapPosition.landmarks?.map((landmark) => landmark.grid_cell && (
          <div className="grid-landmark" style={pointStyle(landmark.grid_cell)} key={landmark.key}>
            <i />
            <span>{landmark.label}</span>
          </div>
        ))}
        {mapPosition.current_cell && (
          <div className="grid-current" style={pointStyle(mapPosition.current_cell)}>
            <i />
            {/* <span>현재</span> */}
          </div>
        )}
      </div>
      {/* {mapPosition.current_cell} */}
    </section>
  )
}

function LivePanel({ cue, analyzing, playbackTime, processingMs, camera = false }) {
  const stability = cue?.stability
  const progressValue = stability
    ? stability.collected < stability.window
      ? stability.collected / stability.window
      : stability.count / stability.required
    : 0
  return (
    <div className="live-caption" aria-live="polite">
      <div className="live-location-row">
        <span>위치</span>
        <b>{cue?.position?.label || '재생 대기'}</b>
      </div>
      

      <div className="live-detail">
        {cue?.navigation?.target && <span>안내 단계 · {cue.navigation.target} 찾기</span>}
        {/* {cue?.location?.motion && <span>움직임 {cue.location.motion}</span>} */}
      </div>



      <div className="bottom-grid">
                  {/* {stability && !stability.confirmed && (
          <div className="stability-progress">
            <span style={{ width: `${Math.min(100, progressValue * 100)}%` }} />
          </div>
        )} */}
        <strong className="live-guidance">
          {cue?.command || cue?.guidance || '영상을 재생하면 현재 화면의 거리와 위치를 바로 알려드려요.'}
        </strong>
      </div>
      
      <GridMap position={cue?.position} />
      

    </div>
  )
}

export default function App() {
  const videoInputRef = useRef(null)
  const liveVideoRef = useRef(null)
  const cameraStreamRef = useRef(null)
  const inFlightFramesRef = useRef(0)
  const liveTimerRef = useRef(null)
  const streamTokenRef = useRef(0)
  const frameSequenceRef = useRef(0)
  const displayedSequenceRef = useRef(0)
  const lastSpokenKeyRef = useRef('')
  const lastSpokenAtRef = useRef(0)
  const [file, setFile] = useState(null)
  const [mediaType, setMediaType] = useState('video')
  const [preview, setPreview] = useState('')
  const [server, setServer] = useState(null)
  const [error, setError] = useState('')
  const [playbackTime, setPlaybackTime] = useState(0)
  const [liveCue, setLiveCue] = useState(null)
  const [liveAnalyzing, setLiveAnalyzing] = useState(false)
  const [liveProcessingMs, setLiveProcessingMs] = useState(0)
  const [cameraActive, setCameraActive] = useState(false)
  const [voiceEnabled, setVoiceEnabled] = useState(true)
  const speechSupported = typeof window !== 'undefined' && 'speechSynthesis' in window

  useEffect(() => {
    fetch('/api/health')
      .then((response) => response.json())
      .then(setServer)
      .catch(() => setServer({ status: 'offline' }))
  }, [])

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview])

  useEffect(() => () => {
    streamTokenRef.current += 1
    if (liveTimerRef.current) window.clearInterval(liveTimerRef.current)
    cameraStreamRef.current?.getTracks().forEach((track) => track.stop())
    if (speechSupported) window.speechSynthesis.cancel()
  }, [])

  useEffect(() => {
    const stability = liveCue?.stability
    if (!speechSupported || !voiceEnabled || !stability) return
    if (!stability.confirmed) {
      lastSpokenKeyRef.current = ''
      return
    }
    const landmark = liveCue.landmarks?.[0]
    const locationKey = liveCue.location?.status === 'localized'
      ? liveCue.location.label
      : liveCue.location?.status || 'unknown'
    const speechText = liveCue.command || liveCue.guidance
    const speechKey = [liveCue.detections?.[0], landmark?.state, locationKey, speechText].join('|')
    const now = Date.now()
    if (speechKey === lastSpokenKeyRef.current && now - lastSpokenAtRef.current < 5000) return

    const utterance = new SpeechSynthesisUtterance(speechText)
    utterance.lang = 'ko-KR'
    utterance.rate = 1
    utterance.pitch = 1
    const koreanVoice = window.speechSynthesis.getVoices().find((voice) => voice.lang?.toLowerCase().startsWith('ko'))
    if (koreanVoice) utterance.voice = koreanVoice
    window.speechSynthesis.cancel()
    window.speechSynthesis.speak(utterance)
    lastSpokenKeyRef.current = speechKey
    lastSpokenAtRef.current = now
  }, [liveCue, speechSupported, voiceEnabled])

  const toggleVoice = () => {
    if (!speechSupported) return
    const next = !voiceEnabled
    setVoiceEnabled(next)
    window.speechSynthesis.cancel()
    if (next) {
      const utterance = new SpeechSynthesisUtterance('음성 안내를 켰습니다.')
      utterance.lang = 'ko-KR'
      window.speechSynthesis.speak(utterance)
      lastSpokenKeyRef.current = ''
    }
  }

  const stopCamera = () => {
    cameraStreamRef.current?.getTracks().forEach((track) => track.stop())
    cameraStreamRef.current = null
    if (liveVideoRef.current) liveVideoRef.current.srcObject = null
    setCameraActive(false)
    setLiveAnalyzing(false)
    inFlightFramesRef.current = 0
    streamTokenRef.current += 1
    if (liveTimerRef.current) window.clearInterval(liveTimerRef.current)
    if (speechSupported) window.speechSynthesis.cancel()
  }

  const resetLiveTracking = () => {
    // 영상 재생 위치를 옮기면 이전 구간의 거리/다수결/격자 점은 사용할 수 없다.
    // 지도 자체는 계속 보이되, 새 구간에서 탐지될 때까지 현재 위치 점만 비운다.
    streamTokenRef.current += 1
    inFlightFramesRef.current = 0
    frameSequenceRef.current = 0
    displayedSequenceRef.current = 0
    setLiveCue(null)
    setLiveProcessingMs(0)
    return fetch('/api/reset-tracking', { method: 'POST' }).catch(() => {})
  }

  const startCamera = async () => {
    setError('')
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('웹 카메라는 HTTPS 보안 주소에서만 사용할 수 있습니다. HTTPS 주소로 다시 접속해 주세요.')
      return
    }
    try {
      stopCamera()
      if (speechSupported && voiceEnabled) {
        window.speechSynthesis.cancel()
        const utterance = new SpeechSynthesisUtterance('음성 안내를 시작합니다.')
        utterance.lang = 'ko-KR'
        window.speechSynthesis.speak(utterance)
      }
      if (preview) URL.revokeObjectURL(preview)
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 },
        },
      })
      cameraStreamRef.current = stream
      setFile(null)
      setPreview('')
      setMediaType('camera')
      setPlaybackTime(0)
      setLiveCue(null)
      setLiveProcessingMs(0)
      setCameraActive(true)
      inFlightFramesRef.current = 0
      frameSequenceRef.current = 0
      displayedSequenceRef.current = 0
      streamTokenRef.current += 1
      resetLiveTracking()
    } catch (err) {
      const message = err?.name === 'NotAllowedError'
        ? '카메라 권한이 거부되었습니다. 브라우저 주소창의 카메라 권한을 허용해 주세요.'
        : `카메라를 시작하지 못했습니다: ${err.message}`
      setError(message)
      stopCamera()
    }
  }

  const selectFile = (event) => {
    const next = event.target.files?.[0]
    if (!next) return
    if (!next.type.startsWith('video/')) {
      setError('영상 파일만 업로드할 수 있습니다.')
      return
    }
    stopCamera()
    if (preview) URL.revokeObjectURL(preview)
    setFile(next)
    setMediaType('video')
    setPreview(URL.createObjectURL(next))
    setPlaybackTime(0)
    setLiveCue(null)
    setLiveProcessingMs(0)
    setLiveAnalyzing(false)
    inFlightFramesRef.current = 0
    frameSequenceRef.current = 0
    displayedSequenceRef.current = 0
    streamTokenRef.current += 1
    if (liveTimerRef.current) window.clearInterval(liveTimerRef.current)
    setError('')
    if (next.type.startsWith('video/')) {
      resetLiveTracking()
    }
  }

  const analyzePlayingFrame = async (video) => {
    if (!video || video.paused || video.ended || inFlightFramesRef.current >= 3 || !serverReady) return
    if (!video.videoWidth || !video.videoHeight) return

    const token = streamTokenRef.current
    const sequence = ++frameSequenceRef.current
    inFlightFramesRef.current += 1
    setLiveAnalyzing(true)
    const capturedTime = video.currentTime
    try {
      const maxWidth = 960
      const scale = Math.min(1, maxWidth / video.videoWidth)
      const canvas = document.createElement('canvas')
      canvas.width = Math.max(1, Math.round(video.videoWidth * scale))
      canvas.height = Math.max(1, Math.round(video.videoHeight * scale))
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height)
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.82))
      if (!blob) throw new Error('현재 영상 화면을 가져오지 못했습니다.')

      const body = new FormData()
      body.append('frame', blob, 'frame.jpg')
      body.append('time_seconds', String(capturedTime))
      const response = await fetch('/api/analyze-frame', { method: 'POST', body })
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || '현재 화면 분석에 실패했습니다.')
      if (streamTokenRef.current === token && sequence > displayedSequenceRef.current) {
        displayedSequenceRef.current = sequence
        setLiveCue(data.cue)
        setLiveProcessingMs(data.processing_ms || 0)
        setError('')
      }
    } catch (err) {
      if (streamTokenRef.current === token) setError(err.message)
    } finally {
      if (streamTokenRef.current === token) {
        inFlightFramesRef.current = Math.max(0, inFlightFramesRef.current - 1)
        setLiveAnalyzing(inFlightFramesRef.current > 0)
      }
    }
  }

  const startLiveAnalysis = (video) => {
    if (liveTimerRef.current) window.clearInterval(liveTimerRef.current)
    analyzePlayingFrame(video)
    liveTimerRef.current = window.setInterval(() => analyzePlayingFrame(video), 200)
  }

  const reset = () => {
    stopCamera()
    if (preview) URL.revokeObjectURL(preview)
    setFile(null)
    setMediaType('video')
    setPreview('')
    setError('')
    setPlaybackTime(0)
    setLiveCue(null)
    setLiveProcessingMs(0)
    setLiveAnalyzing(false)
    inFlightFramesRef.current = 0
    frameSequenceRef.current = 0
    displayedSequenceRef.current = 0
    streamTokenRef.current += 1
    if (liveTimerRef.current) window.clearInterval(liveTimerRef.current)
    if (videoInputRef.current) videoInputRef.current.value = ''
  }

  const serverReady = server?.model_ready

  return (
    <main className={`app-shell ${cameraActive ? 'camera-running' : ''}`}>
      <div className="orientation-gate" role="dialog" aria-label="가로 화면 필요">
        <div className="rotate-phone" aria-hidden="true"><span /></div>
        <strong>휴대폰을 가로로 돌려주세요</strong>
        <p>실시간 카메라와 위치 안내는 가로 화면 전용입니다.</p>
      </div>
      <header>
        <div className="server-line">
          <span className={`server-dot ${serverReady ? 'ready' : ''}`} />
          {server === null ? '서버 확인 중' : serverReady ? `서버 준비 완료` : '서버 준비 안 됨'}
        </div>
        <h1>Rapa Navi</h1>

        {/* <p>휴대폰 영상에서 객체를 탐지하고 거리와 현재 위치를 추정합니다.</p> */}
        <div className='div-button'>

        
            {(cameraActive || preview) && (
              <button
                type="button"
                className="change-button voice-toggle"
                onClick={cameraActive ? stopCamera : startCamera}
              >
                {cameraActive ? '카메라 종료' : '실시간 카메라로 전환'}
              </button>
            )}
              
          <button
            type="button"
            className={`voice-toggle ${voiceEnabled ? 'on' : ''}`}
            aria-pressed={voiceEnabled}
            disabled={!speechSupported}
            onClick={toggleVoice}
          >
            <span aria-hidden="true">{voiceEnabled ? '🔊' : '🔇'}</span>
            {speechSupported ? `음성 안내 ${voiceEnabled ? '켜짐' : '꺼짐'}` : '음성 안내 미지원'}
          </button>

        </div>
      </header>

      <section className={`capture-card ${cameraActive ? 'live-camera-card' : ''} ${cameraActive || (preview && mediaType === 'video') ? 'live-media-card' : ''} ${!preview && !cameraActive ? 'is-empty' : ''}`}>
        <input
          ref={videoInputRef}
          id="video-upload"
          className="hidden-input"
          type="file"
          accept="video/mp4,video/quicktime,video/x-msvideo,video/webm,video/*"
          onChange={selectFile}
        />

        {!preview && !cameraActive ? (
          <div className="upload-options">
            <button className="camera-zone" type="button" onClick={startCamera}>
              <span className="camera-icon">⌾</span>
              <strong>실시간 카메라 시작</strong>
              <small>카메라 앱을 열지 않고 웹에서 바로 보기</small>
            </button>
            <label className="video-zone" htmlFor="video-upload">
              <span className="video-icon">▶</span>
              <span><strong>영상 업로드</strong><small>MP4, MOV 등 · 최대 300MB/5분</small></span>
            </label>
          </div>
        ) : (
          <div className={`preview-wrap ${cameraActive ? 'camera-preview' : ''}`}>
            {cameraActive
              ? <video
                  ref={(node) => {
                    liveVideoRef.current = node
                    if (node && cameraStreamRef.current && node.srcObject !== cameraStreamRef.current) {
                      node.srcObject = cameraStreamRef.current
                    }
                  }}
                  autoPlay
                  muted
                  playsInline
                  onPlaying={(event) => startLiveAnalysis(event.currentTarget)}
                  onPause={() => liveTimerRef.current && window.clearInterval(liveTimerRef.current)}
                />
              : mediaType === 'video'
              ? <video
                  ref={liveVideoRef}
                  src={preview}
                  controls
                  playsInline
                  onPlay={(event) => startLiveAnalysis(event.currentTarget)}
                  onPause={() => liveTimerRef.current && window.clearInterval(liveTimerRef.current)}
                  onTimeUpdate={(event) => setPlaybackTime(event.currentTarget.currentTime)}
                  onSeeking={() => resetLiveTracking()}
                  onSeeked={(event) => {
                    setPlaybackTime(event.currentTarget.currentTime)
                    if (!event.currentTarget.paused) analyzePlayingFrame(event.currentTarget)
                  }}
                />
              : null}
            {cameraActive && <span className="camera-live-badge">LIVE · 5 FPS 분석</span>}
          </div>
        )}

        {(cameraActive || (preview && mediaType === 'video')) && (
          <LivePanel
            cue={liveCue}
            analyzing={liveAnalyzing}
            playbackTime={playbackTime}
            processingMs={liveProcessingMs}
            camera={cameraActive}
          />
        )}

        {file && (
          <div className="file-row">
            <span>{file.name}</span>
            <span>{(file.size / 1024 / 1024).toFixed(1)}MB</span>
          </div>
        )}

        {error && <p className="error">{error}</p>}
        <div className="actions">
          {file && <button className="secondary" onClick={reset}>초기화</button>}
          {mediaType === 'video' && file && (
            <div className="auto-analysis-note">▶ 재생하면 현재 화면을 자동 분석합니다</div>
          )}
          {cameraActive && (
            <div className="auto-analysis-note"><span className="pulse-dot" /> 카메라 화면 자동 분석 중</div>
          )}
        </div>
      </section>

      <footer>모델은 PC 서버에서만 실행됩니다 · 휴대폰에는 설치되지 않습니다</footer>
    </main>
  )
}
