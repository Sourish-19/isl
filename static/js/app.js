/**
 * Indian Sign Language (ISL) Recognition & Translation Frontend Controller.
 * Handles webcam streaming, client-side frame processing, canvas annotations,
 * real-time prediction visualization, sentence construction, and text-to-speech.
 */

class ISLApp {
  constructor() {
    this.video = document.getElementById("webcam-video");
    this.canvas = document.getElementById("landmark-canvas");
    this.ctx = this.canvas.getContext("2d");
    this.toggleCamBtn = document.getElementById("toggle-camera-btn");
    this.camBtnText = document.getElementById("cam-btn-text");
    this.camStatusLabel = document.getElementById("camera-status-label");
    this.fpsLabel = document.getElementById("stream-fps-label");

    // HUD Elements
    this.signLetter = document.getElementById("hud-sign-letter");
    this.valBadge = document.getElementById("hud-validation-badge");
    this.confPct = document.getElementById("hud-conf-pct");
    this.confFill = document.getElementById("hud-conf-fill");
    this.stabilityPct = document.getElementById("hud-stability-pct");
    this.stabilityFill = document.getElementById("hud-stability-fill");
    this.cand1 = document.getElementById("candidate-1");
    this.cand2 = document.getElementById("candidate-2");
    this.cand3 = document.getElementById("candidate-3");

    // Sentence Elements
    this.sentenceTextarea = document.getElementById("sentence-textarea");
    this.activeWordBadge = document.getElementById("active-word-badge");
    this.suggestionsContainer = document.getElementById("suggestions-container");
    this.historyList = document.getElementById("history-feed-list");

    // State Variables
    this.isStreaming = false;
    this.mediaStream = null;
    this.processingInterval = null;
    this.lastFrameTime = performance.now();
    this.frameCount = 0;
    this.currentFps = 0;
    this.isBusyProcessing = false;
    this.lastDetectedLabel = "-";
    this.autoTypeToggle = document.getElementById("auto-type-toggle");

    this.initEventListeners();
    this.loadReferenceGallery();
    this.syncPhraseState();
  }

  initEventListeners() {
    this.toggleCamBtn.addEventListener("click", () => this.toggleWebcam());

    const commitBtn = document.getElementById("btn-commit");
    if (commitBtn) {
      commitBtn.addEventListener("click", () => this.commitCurrentSign());
    }

    document.getElementById("btn-space").addEventListener("click", () => this.sendPhraseAction("space"));
    document.getElementById("btn-backspace").addEventListener("click", () => this.sendPhraseAction("backspace"));
    document.getElementById("btn-clear").addEventListener("click", () => this.sendPhraseAction("clear"));
    document.getElementById("btn-copy").addEventListener("click", () => this.copySentenceText());
    document.getElementById("btn-speak").addEventListener("click", () => this.speakSentenceText());

    // Enter key in reference search
    const searchInput = document.getElementById("ref-search-input");
    if (searchInput) {
      searchInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") this.searchReference();
      });
    }

    // Global keyboard hotkeys
    window.addEventListener("keydown", (e) => {
      if (document.activeElement === this.sentenceTextarea || document.activeElement === searchInput) {
        return; // Don't intercept when user is directly typing in inputs
      }
      if (e.key === "Enter") {
        e.preventDefault();
        this.commitCurrentSign();
      } else if (e.code === "Space") {
        e.preventDefault();
        this.sendPhraseAction("space");
      } else if (e.code === "Backspace") {
        e.preventDefault();
        this.sendPhraseAction("backspace");
      } else if (e.key === "c" || e.key === "C") {
        this.sendPhraseAction("clear");
      }
    });
  }

  async toggleWebcam() {
    if (this.isStreaming) {
      this.stopWebcam();
    } else {
      await this.startWebcam();
    }
  }

  async startWebcam() {
    try {
      this.camStatusLabel.textContent = "Camera: Initializing...";
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false
      });
      this.video.srcObject = this.mediaStream;

      this.video.onloadedmetadata = () => {
        this.canvas.width = this.video.videoWidth || 640;
        this.canvas.height = this.video.videoHeight || 480;
        this.isStreaming = true;
        this.camBtnText.textContent = "Stop Camera";
        this.toggleCamBtn.classList.replace("btn-primary", "btn-danger");
        this.camStatusLabel.textContent = "Camera: Active (Streaming)";

        // Process frames periodically (every 120ms = ~8 FPS pipeline)
        this.processingInterval = setInterval(() => this.processNextFrame(), 120);
      };
    } catch (err) {
      console.error("Webcam access error:", err);
      this.camStatusLabel.textContent = "Camera: Permission Denied or Unavailable";
      alert("Unable to access webcam. Please ensure your camera is connected and browser permissions are granted.");
    }
  }

  stopWebcam() {
    if (this.processingInterval) {
      clearInterval(this.processingInterval);
      this.processingInterval = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
    this.video.srcObject = null;
    this.isStreaming = false;
    this.camBtnText.textContent = "Start Webcam";
    this.toggleCamBtn.classList.replace("btn-danger", "btn-primary");
    this.camStatusLabel.textContent = "Camera: Standby";
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.resetHUD();
  }

  async processNextFrame() {
    if (!this.isStreaming || this.isBusyProcessing) return;

    this.isBusyProcessing = true;
    try {
      // 1. Calculate FPS
      const now = performance.now();
      const delta = now - this.lastFrameTime;
      this.lastFrameTime = now;
      this.currentFps = Math.round(1000 / delta);
      this.fpsLabel.textContent = `FPS: ${this.currentFps}`;

      // 2. Capture snapshot from video onto an offscreen canvas
      const offscreen = document.createElement("canvas");
      offscreen.width = 360; // Compressed width for high throughput
      offscreen.height = Math.round((offscreen.width / this.video.videoWidth) * this.video.videoHeight);
      const offCtx = offscreen.getContext("2d");
      offCtx.drawImage(this.video, 0, 0, offscreen.width, offscreen.height);

      const base64Data = offscreen.toDataURL("image/jpeg", 0.7);

      // 3. Post to prediction API
      const response = await fetch("/api/predict/image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: base64Data, confidence_threshold: 0.50 })
      });

      if (!response.ok) {
        console.error("API Response error:", response.status, await response.text());
        return;
      }
      const data = await response.json();

      // 4. Clear and redraw canvas annotations
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

      if (data.hand_detected && data.prediction) {
        const pred = data.prediction;
        this.lastDetectedLabel = pred.label;

        // Render landmark skeleton
        if (data.landmarks && data.landmarks.length > 0) {
          this.drawLandmarks(data.landmarks);
        }

        const autoType = this.autoTypeToggle ? this.autoTypeToggle.checked : true;
        if (autoType) {
          // Send prediction to phrase buffer for debounced typing
          const bufRes = await fetch("/api/phrase/process_frame", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(pred)
          });
          const bufData = await bufRes.json();

          this.updateHUD(pred, bufData);

          if (bufData.event) {
            this.logHistory(bufData.event);
          }

          this.sentenceTextarea.value = bufData.text || "";
          this.activeWordBadge.textContent = bufData.current_word ? `Active: ${bufData.current_word}` : "Active: [None]";
          this.renderSuggestions(bufData.suggestions || []);
        } else {
          // In manual mode, simply update HUD without auto-typing
          this.updateHUD(pred, { smoothed_label: pred.label, stability_progress: 0 });
        }
      } else {
        this.lastDetectedLabel = "-";
        this.resetHUD();
      }
    } catch (err) {
      console.warn("Frame processing error:", err);
    } finally {
      this.isBusyProcessing = false;
    }
  }

  commitCurrentSign() {
    const label = this.lastDetectedLabel;
    if (label && label !== "-" && label !== "Unrecognized") {
      this.sendPhraseAction("add_char", label);
      this.logHistory(`Committed sign: '${label}'`);
    }
  }

  drawLandmarks(landmarks) {
    if (!landmarks || landmarks.length === 0) return;
    const w = this.canvas.width;
    const h = this.canvas.height;

    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4],
      [0, 5], [5, 6], [6, 7], [7, 8],
      [5, 9], [9, 10], [10, 11], [11, 12],
      [9, 13], [13, 14], [14, 15], [15, 16],
      [13, 17], [17, 18], [18, 19], [19, 20],
      [0, 17]
    ];

    const numHands = Math.floor(landmarks.length / 21);
    for (let handIdx = 0; handIdx < numHands; handIdx++) {
      const offset = handIdx * 21;

      // Draw bones
      this.ctx.strokeStyle = "rgba(56, 189, 248, 0.75)";
      this.ctx.lineWidth = 2.5;
      for (const [start, end] of connections) {
        const p1 = landmarks[offset + start];
        const p2 = landmarks[offset + end];
        if (p1 && p2) {
          this.ctx.beginPath();
          this.ctx.moveTo(p1.x * w, p1.y * h);
          this.ctx.lineTo(p2.x * w, p2.y * h);
          this.ctx.stroke();
        }
      }

      // Draw joints
      for (let i = 0; i < 21; i++) {
        const lm = landmarks[offset + i];
        if (lm) {
          this.ctx.fillStyle = i === 0 ? "#f43f5e" : (i % 4 === 0 ? "#34d399" : "#38bdf8");
          this.ctx.beginPath();
          this.ctx.arc(lm.x * w, lm.y * h, i % 4 === 0 ? 5 : 3.5, 0, 2 * Math.PI);
          this.ctx.fill();
        }
      }
    }
  }

  updateHUD(pred, bufData) {
    const label = bufData.smoothed_label || pred.label;
    this.signLetter.textContent = label;

    if (this.valBadge) {
      if (pred.is_empirically_validated) {
        this.valBadge.textContent = "🟢 Test-Validated Sign";
        this.valBadge.style.background = "rgba(16, 185, 129, 0.2)";
        this.valBadge.style.color = "#34d399";
        this.valBadge.style.border = "1px solid rgba(16, 185, 129, 0.4)";
      } else {
        this.valBadge.textContent = "🟡 Reference Architecture Sign";
        this.valBadge.style.background = "rgba(245, 158, 11, 0.15)";
        this.valBadge.style.color = "#fbbf24";
        this.valBadge.style.border = "1px solid rgba(245, 158, 11, 0.3)";
      }
    }

    const conf = Math.round(pred.confidence * 100);
    this.confPct.textContent = `${conf}%`;
    this.confFill.style.width = `${conf}%`;

    const stability = Math.round((bufData.stability_progress || 0) * 100);
    this.stabilityPct.textContent = `${stability}%`;
    this.stabilityFill.style.width = `${stability}%`;

    // Top candidates
    if (pred.top_candidates && pred.top_candidates.length >= 3) {
      this.cand1.textContent = `${pred.top_candidates[0].label} (${Math.round(pred.top_candidates[0].confidence * 100)}%)`;
      this.cand2.textContent = `${pred.top_candidates[1].label} (${Math.round(pred.top_candidates[1].confidence * 100)}%)`;
      this.cand3.textContent = `${pred.top_candidates[2].label} (${Math.round(pred.top_candidates[2].confidence * 100)}%)`;
    }
  }

  resetHUD() {
    this.signLetter.textContent = "-";
    if (this.valBadge) {
      this.valBadge.textContent = "Standby";
      this.valBadge.style.background = "rgba(56, 189, 248, 0.15)";
      this.valBadge.style.color = "var(--accent-cyan)";
      this.valBadge.style.border = "none";
    }
    this.confPct.textContent = "0%";
    this.confFill.style.width = "0%";
    this.stabilityPct.textContent = "0%";
    this.stabilityFill.style.width = "0%";
    this.cand1.textContent = "-";
    this.cand2.textContent = "-";
    this.cand3.textContent = "-";
  }

  async sendPhraseAction(action, value = null) {
    try {
      const res = await fetch("/api/phrase/action", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, value })
      });
      const data = await res.json();
      this.sentenceTextarea.value = data.text || "";
      this.activeWordBadge.textContent = data.current_word ? `Active: ${data.current_word}` : "Active: [None]";
      this.renderSuggestions(data.suggestions || []);
    } catch (err) {
      console.error("Phrase action error:", err);
    }
  }

  async syncPhraseState() {
    try {
      const res = await fetch("/api/phrase/state");
      const data = await res.json();
      this.sentenceTextarea.value = data.text || "";
      this.activeWordBadge.textContent = data.current_word ? `Active: ${data.current_word}` : "Active: [None]";
      this.renderSuggestions(data.suggestions || []);
    } catch (err) {
      console.error("State sync error:", err);
    }
  }

  insertPhrase(phrase) {
    this.sendPhraseAction("add_phrase", phrase);
    this.logHistory(`Inserted phrase: "${phrase}"`);
  }

  renderSuggestions(suggestions) {
    this.suggestionsContainer.innerHTML = "";
    suggestions.forEach(item => {
      const chip = document.createElement("span");
      chip.className = "phrase-chip";
      chip.textContent = item;
      chip.onclick = () => this.insertPhrase(item);
      this.suggestionsContainer.appendChild(chip);
    });
  }

  copySentenceText() {
    const text = this.sentenceTextarea.value;
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      const btn = document.getElementById("btn-copy");
      const originalText = btn.innerHTML;
      btn.innerHTML = "✅ Copied!";
      setTimeout(() => { btn.innerHTML = originalText; }, 1500);
    });
  }

  speakSentenceText() {
    const text = this.sentenceTextarea.value;
    if (!text) {
      alert("No sentence text to speak.");
      return;
    }
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.95;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    } else {
      alert("Text-to-Speech is not supported by your browser.");
    }
  }

  logHistory(msg) {
    const entry = document.createElement("div");
    entry.className = "history-entry";
    const now = new Date().toLocaleTimeString();
    entry.innerHTML = `<span>${msg}</span><span class="time">${now}</span>`;
    this.historyList.prepend(entry);
  }

  switchTab(tab) {
    document.getElementById("tab-btn-ref").classList.toggle("active", tab === "ref");
    document.getElementById("tab-btn-history").classList.toggle("active", tab === "history");
    document.getElementById("tab-content-ref").style.display = tab === "ref" ? "block" : "none";
    document.getElementById("tab-content-history").style.display = tab === "history" ? "block" : "none";
  }

  async loadReferenceGallery() {
    try {
      const res = await fetch("/api/reference/gallery");
      const data = await res.json();
      this.galleryData = data.gallery || [];
      this.renderGalleryGrid(this.galleryData);
    } catch (err) {
      console.error("Failed to load reference gallery:", err);
    }
  }

  renderGalleryGrid(items) {
    const grid = document.getElementById("reference-gallery-grid");
    if (!grid) return;
    grid.innerHTML = "";

    items.forEach(item => {
      const el = document.createElement("div");
      el.className = "reference-item";
      el.style.position = "relative";
      const badgeHtml = item.is_empirically_validated
        ? `<span style="position: absolute; top: 4px; right: 4px; font-size: 0.6rem; background: #059669; color: #fff; padding: 2px 5px; border-radius: 4px; font-weight: 700;">Tested</span>`
        : `<span style="position: absolute; top: 4px; right: 4px; font-size: 0.6rem; background: #1e293b; color: #94a3b8; padding: 2px 5px; border-radius: 4px; border: 1px solid #334155;">Guide</span>`;

      el.innerHTML = `
        ${badgeHtml}
        <img src="${item.image_url}" alt="${item.sign}" loading="lazy"/>
        <div class="ref-label">${item.sign}</div>
      `;
      el.onclick = () => this.showPreview(item.sign, item.image_url, item.description, item.is_empirically_validated);
      grid.appendChild(el);
    });
  }

  filterGallery(filterType) {
    if (!this.galleryData || this.galleryData.length === 0) return;
    if (filterType === 'validated') {
      const filtered = this.galleryData.filter(item => item.is_empirically_validated);
      this.renderGalleryGrid(filtered);
    } else if (filterType === 'reference') {
      const filtered = this.galleryData.filter(item => !item.is_empirically_validated);
      this.renderGalleryGrid(filtered);
    } else {
      this.renderGalleryGrid(this.galleryData);
    }
  }

  showPreview(title, imgUrl, desc, isValidated = null) {
    const box = document.getElementById("ref-preview-box");
    box.style.display = "block";
    const statusBadge = isValidated === true 
      ? " [🟢 Test-Validated Model Sign]" 
      : (isValidated === false ? " [🔵 Visual Reference Guide]" : "");
    document.getElementById("ref-preview-title").textContent = `ISL Sign: ${title}${statusBadge}`;
    document.getElementById("ref-preview-img").src = imgUrl;
    document.getElementById("ref-preview-desc").textContent = desc || "Verified Indian Sign Language gesture.";
  }

  async searchReference() {
    const query = document.getElementById("ref-search-input").value.trim().toUpperCase();
    if (!query) return;

    try {
      const res = await fetch(`/api/reference/phrase?text=${encodeURIComponent(query)}`);
      const data = await res.json();

      if (data.visual_sequence && data.visual_sequence.length > 0) {
        const first = data.visual_sequence[0];
        let desc = first.description;
        if (data.phrase_metadata) {
          desc = `${data.phrase_metadata.meaning} (${desc})`;
        }
        this.showPreview(query, first.image_url, desc);
      } else {
        alert(`No demonstration found for '${query}'. Try an individual sign (A-Z, 1-9) or common phrase.`);
      }
    } catch (err) {
      console.error("Reference search error:", err);
    }
  }
}

// Instantiate application on window load
window.addEventListener("DOMContentLoaded", () => {
  window.app = new ISLApp();
});
