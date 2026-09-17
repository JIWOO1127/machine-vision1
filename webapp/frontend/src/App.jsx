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
      <div className="position-grid-heading">
        <span>현재 위치 지도</span>
        <b>{mapPosition.label}</b>
      </div>
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
            <span>현재</span>
          </div>
        )}
      </div>
      {mapPosition.current_cell && <small>격자 좌표 · ({mapPosition.current_cell[0]}, {mapPosition.current_cell[1]})</small>}
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
        <b>{cue?.location?.label || '재생 대기'}</b>
      </div>

      {cue?.landmarks?.length > 0 ? (
        <div className="live-landmarks">
          {cue.landmarks.map((item, index) => (
            <div className="live-landmark" key={`${item.name}-${index}`}>
              <span>{item.direction} · {item.name}</span>
              <b>{item.distance_m.toFixed(1)}m</b>
              <em className={item.state}>{item.state_text}</em>
            </div>
          ))}
        </div>
      ) : (
        <span className="live-empty">거리 정보 없음</span>
      )}

      <div className="live-detail">
        <span>객체 {cue?.detections?.length ? cue.detections.join(' · ') : '탐지 없음'}</span>
        {cue?.position?.description && <span>기준 위치 · {cue.position.description}</span>}
        {cue?.navigation?.target && <span>안내 단계 · {cue.navigation.target} 찾기</span>}
        {/* {cue?.location?.motion && <span>움직임 {cue.location.motion}</span>} */}
      </div>

      <GridMap position={cue?.position} />


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
      

    </div>
  )
}

export default function App() {
  const inputRef = useRef(null)
  const videoInputRef = useRef(null)
  const liveVideoRef = useRef(null)
  const cameraStreamRef = useRef(null)
  const inFlightFramesRef = useRef(0)
  const liveTimerRef = useRef(null)
  const streamTokenRef = useRef(0)
  const frameSequenceRef = useRef(0)
  const displayedSequenceRef = useRef(0)
  const lastSpokenKeyRef = useRef('')
  const [file, setFile] = useState(null)
  const [mediaType, setMediaType] = useState('image')
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState(null)
  const [server, setServer] = useState(null)
  const [loading, setLoading] = useState(false)
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
    if (!speechSupported || !voiceEnabled || !stability?.confirmed) return
    const speechText = liveCue.command || liveCue.guidance
    if (!speechText) return
    // stability.confirmed는 최근 5프레임 중 4프레임 다수결이라 노이즈로 프레임마다
    // true/false가 깜빡일 수 있다. confirmed가 잠깐 꺼졌다 켜질 때마다 다시 말하지
    // 않도록, "마지막으로 말한 문장"은 confirmed 여부와 무관하게 유지하고 문장이
    // 실제로 바뀔 때만 새로 안내한다.
    if (speechText === lastSpokenKeyRef.current) return
    if (window.speechSynthesis.speaking) return

    const utterance = new SpeechSynthesisUtterance(speechText)
    utterance.lang = 'ko-KR'
    utterance.rate = 1
    utterance.pitch = 1
    const koreanVoice = window.speechSynthesis.getVoices().find((voice) => voice.lang?.toLowerCase().startsWith('ko'))
    if (koreanVoice) utterance.voice = koreanVoice
    window.speechSynthesis.speak(utterance)
    lastSpokenKeyRef.current = speechText
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
    lastSpokenKeyRef.current = ''
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
    lastSpokenKeyRef.current = ''
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
      setResult(null)
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
    stopCamera()
    if (preview) URL.revokeObjectURL(preview)
    setFile(next)
    setMediaType(next.type.startsWith('video/') ? 'video' : 'image')
    setPreview(URL.createObjectURL(next))
    setResult(null)
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

  const analyze = async () => {
    if (!file) return
    setLoading(true)
    setError('')
    try {
      const body = new FormData()
      body.append(mediaType === 'video' ? 'video' : 'image', file)
      if (mediaType === 'video') body.append('frame_step', '5')
      const endpoint = mediaType === 'video' ? '/api/analyze-video' : '/api/analyze'
      const response = await fetch(endpoint, { method: 'POST', body })
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || '분석 요청에 실패했습니다.')
      setResult(data)
      setPlaybackTime(0)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    stopCamera()
    if (preview) URL.revokeObjectURL(preview)
    setFile(null)
    setMediaType('image')
    setPreview('')
    setResult(null)
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
    if (inputRef.current) inputRef.current.value = ''
    if (videoInputRef.current) videoInputRef.current.value = ''
  }

  const location = result?.location
  const serverReady = server?.model_ready
  const timeline = result?.timeline || []
  const activeCue = timeline.reduce(
    (current, cue) => cue.time_seconds <= playbackTime + 0.05 ? cue : current,
    timeline[0] || null,
  )

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
        <h1>현재 위치 확인</h1>
        {/* <p>휴대폰 영상에서 객체를 탐지하고 거리와 현재 위치를 추정합니다.</p> */}
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
      </header>

      <section className={`capture-card ${cameraActive ? 'live-camera-card' : ''} ${cameraActive || (preview && mediaType === 'video') ? 'live-media-card' : ''} ${!preview && !cameraActive ? 'is-empty' : ''}`}>
        <input
          ref={inputRef}
          id="camera"
          className="hidden-input"
          type="file"
          accept="image/*"
          capture="environment"
          onChange={selectFile}
        />
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
              : <img src={preview} alt="분석할 사진 미리보기" />}
            {cameraActive && <span className="camera-live-badge">LIVE · 5 FPS 분석</span>}
            <button className="change-button" onClick={() => cameraActive ? stopCamera() : (mediaType === 'video' ? videoInputRef : inputRef).current?.click()}>
              {cameraActive ? '카메라 종료' : `다른 ${mediaType === 'video' ? '영상' : '사진'}`}
            </button>
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
          {mediaType === 'image' && (
            <button className="primary" disabled={!file || loading || (server && !serverReady)} onClick={analyze}>
              {loading ? '서버에서 분석 중…' : '사진 분석하기'}
            </button>
          )}
          {mediaType === 'video' && file && (
            <div className="auto-analysis-note">▶ 재생하면 현재 화면을 자동 분석합니다</div>
          )}
          {cameraActive && (
            <div className="auto-analysis-note"><span className="pulse-dot" /> 카메라 화면 자동 분석 중</div>
          )}
        </div>
      </section>

      {result && (
        <section className="result-card">
          <div className="result-heading">
            <div><span className="status-dot" /> 분석 완료</div>
            <small>{(result.processing_ms / 1000).toFixed(1)}초</small>
          </div>

          {result.annotated_video_url && (
            <div className="playback-section">
              <video
                src={result.annotated_video_url}
                controls
                playsInline
                preload="metadata"
                onTimeUpdate={(event) => setPlaybackTime(event.currentTarget.currentTime)}
                onSeeked={(event) => setPlaybackTime(event.currentTarget.currentTime)}
              />
              <LivePanel cue={activeCue} analyzing={false} playbackTime={playbackTime} processingMs={0} />
            </div>
          )}

          {result.annotated_image_url && (
            <div className="playback-section image-result">
              <img src={result.annotated_image_url} alt="서버 분석 결과" />
            </div>
          )}

          <div className="location-box">
            <span>현재 위치</span>
            <strong>{location?.label || '위치 불확실'}</strong>
            <p>{location?.detail}</p>
          </div>

          {location?.candidates?.length > 0 && (
            <div className="candidate-list">
              {location.candidates.slice(0, 3).map((item) => (
                <div className="candidate" key={item.name}>
                  <span>{item.name}</span>
                  <b>{Math.round(item.score * 100)}%</b>
                </div>
              ))}
            </div>
          )}

          <h2>객체 탐지</h2>
          {result.detections.length ? result.detections.map((item, index) => (
            <div className="result-row" key={`${item.label}-${index}`}>
              <span>{item.label}</span>
              <b>{Math.round(item.confidence * 100)}%</b>
            </div>
          )) : <p className="empty">유효한 객체가 없습니다.</p>}

        </section>
      )}

      <footer>모델은 PC 서버에서만 실행됩니다 · 휴대폰에는 설치되지 않습니다</footer>
    </main>
  )
}
