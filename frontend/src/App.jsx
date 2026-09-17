import { useEffect, useRef, useState } from 'react'

export default function App() {
  const inputRef = useRef(null)
  const videoInputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [mediaType, setMediaType] = useState('image')
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState(null)
  const [server, setServer] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [playbackTime, setPlaybackTime] = useState(0)

  useEffect(() => {
    fetch('/api/health')
      .then((response) => response.json())
      .then(setServer)
      .catch(() => setServer({ status: 'offline' }))
  }, [])

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview])

  const selectFile = (event) => {
    const next = event.target.files?.[0]
    if (!next) return
    if (preview) URL.revokeObjectURL(preview)
    setFile(next)
    setMediaType(next.type.startsWith('video/') ? 'video' : 'image')
    setPreview(URL.createObjectURL(next))
    setResult(null)
    setPlaybackTime(0)
    setError('')
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
    if (preview) URL.revokeObjectURL(preview)
    setFile(null)
    setMediaType('image')
    setPreview('')
    setResult(null)
    setError('')
    setPlaybackTime(0)
    if (inputRef.current) inputRef.current.value = ''
    if (videoInputRef.current) videoInputRef.current.value = ''
  }

  const location = result?.location
  const serverReady = server?.model_ready && server?.ocr_ready
  const timeline = result?.timeline || []
  const activeCue = timeline.reduce(
    (current, cue) => cue.time_seconds <= playbackTime + 0.05 ? cue : current,
    timeline[0] || null,
  )

  return (
    <main className="app-shell">
      <header>
        <div className="server-line">
          <span className={`server-dot ${serverReady ? 'ready' : ''}`} />
          {server === null ? '서버 확인 중' : serverReady ? `서버 준비 완료 · ${server.device}` : '서버 준비 안 됨'}
        </div>
        <h1>현재 위치 확인</h1>
        <p>휴대폰으로 촬영하면 서버가 객체 탐지와 글자 인식을 처리합니다.</p>
      </header>

      <section className="capture-card">
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

        {!preview ? (
          <div className="upload-options">
            <label className="camera-zone" htmlFor="camera">
              <span className="camera-icon">⌾</span>
              <strong>사진 촬영하기</strong>
              <small>카메라로 바로 촬영</small>
            </label>
            <label className="video-zone" htmlFor="video-upload">
              <span className="video-icon">▶</span>
              <span><strong>영상 업로드</strong><small>MP4, MOV 등 · 최대 300MB/5분</small></span>
            </label>
          </div>
        ) : (
          <div className="preview-wrap">
            {mediaType === 'video'
              ? <video src={preview} controls playsInline />
              : <img src={preview} alt="분석할 사진 미리보기" />}
            <button className="change-button" onClick={() => (mediaType === 'video' ? videoInputRef : inputRef).current?.click()}>
              다른 {mediaType === 'video' ? '영상' : '사진'}
            </button>
          </div>
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
          <button className="primary" disabled={!file || loading || (server && !serverReady)} onClick={analyze}>
            {loading
              ? mediaType === 'video' ? '영상 업로드·분석 중…' : '서버에서 분석 중…'
              : mediaType === 'video' ? '영상 분석하기' : '사진 분석하기'}
          </button>
        </div>
        {loading && mediaType === 'video' && <p className="wait-note">영상 길이에 따라 몇 분 정도 걸릴 수 있어요. 화면을 닫지 말아 주세요.</p>}
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
              <div className="live-caption" aria-live="polite">
                <div className="caption-time">재생 {playbackTime.toFixed(1)}초</div>
                <strong className="live-guidance">
                  {activeCue?.guidance || '이 구간의 위치 정보를 확인하고 있습니다.'}
                </strong>

                <div className="live-location-row">
                  <span>추정 현재 위치</span>
                  <b>{activeCue?.location?.label || '위치 불확실'}</b>
                </div>

                {activeCue?.landmarks?.length > 0 ? (
                  <div className="live-landmarks">
                    {activeCue.landmarks.map((item, index) => (
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
                  <span>OCR {activeCue?.texts?.length ? activeCue.texts.join(' · ') : '인식 없음'}</span>
                  <span>객체 {activeCue?.detections?.length ? activeCue.detections.join(' · ') : '탐지 없음'}</span>
                  {activeCue?.location?.motion && <span>움직임 {activeCue.location.motion}</span>}
                </div>
              </div>
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

          <h2>{result.annotated_video_url ? '영상 전체 인식 텍스트' : '인식된 글자'}</h2>
          {result.texts.length ? result.texts.map((item, index) => (
            <div className="ocr-text" key={`${item.text}-${index}`}>
              <span>{item.text}</span>
              <small>{Math.round(item.confidence * 100)}%</small>
            </div>
          )) : <p className="empty">인식된 표지판 글자가 없습니다.</p>}

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
