// Continuous speech-to-text for the dictation fields (transparency,
// adaptability, survey, gesture setup, preference correction).
//
// A Web Speech API session ends as soon as the speaker pauses -- on the iPad
// that meant one tap of the mic captured one phrase and then went dead, so a
// user thinking mid-sentence lost the rest of their answer. Two things fix it:
// `continuous = true` (browsers that honour it keep a single session open
// across pauses) and restarting the session from `onend` for the ones that
// don't. Dictation runs until the user taps the mic again -- the button is a
// toggle, not a one-shot.
//
// createDictation() owns the recognizer, the restart loop, and the takeover-mic
// handshake; a view only supplies callbacks for the text and the status line.

const RESTART_DELAY_MS = 200
// A session ending with nothing said is normal (the user is just thinking).
// One that ends the instant it starts, again and again, means the browser is
// refusing to keep the mic open -- stop rather than spin.
const MIN_HEALTHY_SESSION_MS = 800
const MAX_IMMEDIATE_RESTARTS = 4
// No amount of restarting fixes these: report and stop.
const FATAL_ERRORS = ['not-allowed', 'service-not-allowed', 'audio-capture', 'language-not-supported']
// How much of the in-flight phrase to echo in the status line.
const INTERIM_PREVIEW_CHARS = 48
const LISTENING = 'listening…'

export function speechRecognitionCtor () {
  return window.SpeechRecognition || window.webkitSpeechRecognition || null
}

// App.vue holds a persistent mic stream for physical-button detection and the
// recognizer can't start while it's open. Ask App.vue to release it, and don't
// wait forever if nothing is listening.
export function releaseTakeoverMic () {
  return new Promise((resolve) => {
    let settled = false
    const done = () => {
      if (settled) return
      settled = true
      resolve()
    }
    window.dispatchEvent(new CustomEvent('release-takeover-mic', { detail: { done } }))
    setTimeout(done, 300)
  })
}

function previewTail (text) {
  const clean = text.trim()
  if (clean.length <= INTERIM_PREVIEW_CHARS) return clean
  return '…' + clean.slice(clean.length - INTERIM_PREVIEW_CHARS)
}

/**
 * @param {object} options
 * @param {(text: string) => void} options.onText   finalized phrase to append
 * @param {(status: string) => void} options.onStatus  status-line text ('' clears)
 * @param {(active: boolean) => void} options.onActive  drives the mic button state
 * @param {string} [options.lang]
 */
export function createDictation (options = {}) {
  const onText = options.onText || (() => {})
  const onStatus = options.onStatus || (() => {})
  const onActive = options.onActive || (() => {})
  const lang = options.lang || 'en-US'

  let recognition = null
  let active = false          // what the view has been told
  let wantActive = false      // user wants to keep dictating
  let destroyed = false
  let restartTimer = null
  let immediateRestarts = 0
  let sessionStart = 0
  let capturedAnything = false
  let interimText = ''
  let pendingError = ''
  // Set by cancel(): abort() can deliver `onend` synchronously, so the quiet
  // teardown has to win no matter which path reaches finish() first.
  let finishQuiet = false

  const statusText = () => (interimText ? LISTENING + ' ' + previewTail(interimText) : LISTENING)

  const clearRestartTimer = () => {
    if (restartTimer) {
      clearTimeout(restartTimer)
      restartTimer = null
    }
  }

  // Settle back to idle: the view's button goes dark and the status line either
  // reports why we stopped or clears. `quiet` is for teardown (new question,
  // leaving the page), where there is nothing to tell the user.
  const finish = (quiet) => {
    if (!active) return
    active = false
    interimText = ''
    const silent = quiet || finishQuiet
    finishQuiet = false
    onActive(false)
    onStatus(silent ? '' : (pendingError || (capturedAnything ? '' : 'no speech captured')))
    pendingError = ''
  }

  const build = () => {
    const Ctor = speechRecognitionCtor()
    const r = new Ctor()
    r.lang = lang
    r.continuous = true
    // Interim results are both live feedback and proof the mic is still open
    // during a long answer.
    r.interimResults = true
    r.maxAlternatives = 1

    r.onstart = () => {
      sessionStart = Date.now()
      if (active) onStatus(statusText())
    }

    r.onresult = (event) => {
      let finalText = ''
      let interim = ''
      // With continuous sessions `results` accumulates, so only walk the
      // entries this event actually changed.
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i]
        const text = result[0] ? result[0].transcript : ''
        if (result.isFinal) finalText += text
        else interim += text
      }
      if (finalText) {
        immediateRestarts = 0
        capturedAnything = true
        onText(finalText)
        // stop() flushes the phrase in flight after we've already settled;
        // don't leave "no speech captured" over text that did arrive.
        if (!active) onStatus('')
      }
      interimText = interim
      if (active) onStatus(statusText())
    }

    r.onerror = (event) => {
      const err = event.error || 'unknown'
      if (FATAL_ERRORS.indexOf(err) !== -1) {
        pendingError = 'error: ' + err + (event.message ? ' — ' + event.message : '')
        wantActive = false
        return
      }
      // 'no-speech', 'network', 'aborted': recoverable -- onend restarts us.
      // eslint-disable-next-line no-console
      console.warn('[dictation] recoverable recognition error:', err, event.message || '')
    }

    r.onend = () => {
      if (destroyed || !wantActive) {
        finish()
        return
      }
      immediateRestarts = (Date.now() - sessionStart < MIN_HEALTHY_SESSION_MS)
        ? immediateRestarts + 1
        : 0
      if (immediateRestarts > MAX_IMMEDIATE_RESTARTS) {
        pendingError = 'stopped: this browser will not keep the microphone open'
        wantActive = false
        finish()
        return
      }
      clearRestartTimer()
      restartTimer = setTimeout(() => {
        restartTimer = null
        if (wantActive && !destroyed) launch()
      }, RESTART_DELAY_MS)
    }

    return r
  }

  const launch = () => {
    if (!recognition) recognition = build()
    sessionStart = Date.now()
    try {
      recognition.start()
    } catch (e) {
      // start() throws if the previous session hasn't finished tearing down.
      // Treat it like an immediate end and back off; give up after a few tries
      // so a browser that refuses to restart doesn't spin.
      // eslint-disable-next-line no-console
      console.warn('[dictation] recognition.start() threw:', e)
      immediateRestarts += 1
      if (immediateRestarts > MAX_IMMEDIATE_RESTARTS) {
        pendingError = 'start failed: ' + (e && e.message ? e.message : e)
        wantActive = false
        finish()
        return
      }
      clearRestartTimer()
      restartTimer = setTimeout(() => {
        restartTimer = null
        if (wantActive && !destroyed) launch()
      }, RESTART_DELAY_MS)
    }
  }

  const start = async () => {
    if (destroyed || wantActive) return
    if (!speechRecognitionCtor()) {
      onStatus('speech recognition not supported in this browser')
      return
    }
    wantActive = true
    immediateRestarts = 0
    capturedAnything = false
    interimText = ''
    pendingError = ''
    active = true
    onActive(true)
    onStatus('starting…')

    await releaseTakeoverMic()
    if (destroyed || !wantActive) {
      finish()
      return
    }
    launch()
  }

  // User-requested end of dictation. stop() (not abort()) so the phrase in
  // flight is still finalized into the field.
  const stop = () => {
    wantActive = false
    clearRestartTimer()
    if (recognition) {
      try { recognition.stop() } catch (e) { /* not running */ }
    }
    finish()
  }

  // Drop dictation without keeping what is in flight -- used when the field
  // itself goes away (next survey question), where a late-arriving phrase would
  // land in the wrong answer.
  const cancel = () => {
    wantActive = false
    finishQuiet = true
    clearRestartTimer()
    pendingError = ''
    if (recognition) {
      try { recognition.abort() } catch (e) { /* not running */ }
    }
    finish(true)
    finishQuiet = false
  }

  return {
    start,
    stop,
    cancel,
    toggle () {
      if (wantActive) stop()
      else start()
    },
    isActive () {
      return wantActive
    },
    // Leaving the page / loading the next question: drop the audio immediately
    // and make sure the restart loop can't outlive the view.
    destroy () {
      destroyed = true
      wantActive = false
      clearRestartTimer()
      if (recognition) {
        try { recognition.abort() } catch (e) { /* not running */ }
        recognition.onstart = null
        recognition.onresult = null
        recognition.onerror = null
        recognition.onend = null
        recognition = null
      }
      active = false
      interimText = ''
    }
  }
}
