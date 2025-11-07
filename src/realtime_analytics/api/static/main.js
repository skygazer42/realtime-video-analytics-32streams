// ==================== DOM Elements ====================
const statusEl = document.getElementById("status");
const streamsBody = document.getElementById("streams-body");
const tracksBody = document.getElementById("tracks-body");
const previewImg = document.getElementById("stream-preview");
const previewPlaceholder = document.getElementById("preview-placeholder");
const previewInfo = document.getElementById("preview-info");
const previewResolution = document.getElementById("preview-resolution");
const previewTimestamp = document.getElementById("preview-timestamp");
const selectedStreamName = document.getElementById("selected-stream-name");
const trackCount = document.getElementById("track-count");

// Statistics elements
const statTotalStreams = document.getElementById("stat-total-streams");
const statTotalTracks = document.getElementById("stat-total-tracks");
const statTotalDetections = document.getElementById("stat-total-detections");
const statUptime = document.getElementById("stat-uptime");

// Search and filter elements
const streamSearchInput = document.getElementById("stream-search");
const streamFilterSelect = document.getElementById("stream-filter");

// ==================== State ====================
let selectedStream = null;
let latestEvents = {};
let startTime = Date.now();
let totalDetectionsCount = 0;
let lastDetectionTime = Date.now();
let searchQuery = "";
let filterMode = "all";

// ==================== Status Management ====================
function setStatus(connected) {
  if (connected) {
    statusEl.innerHTML = '<span class="status-dot"></span>Connected';
    statusEl.classList.remove("status--disconnected");
    statusEl.classList.add("status--connected");
  } else {
    statusEl.innerHTML = '<span class="status-dot"></span>Disconnected';
    statusEl.classList.remove("status--connected");
    statusEl.classList.add("status--disconnected");
  }
}

// ==================== Statistics Update ====================
function updateStatistics() {
  const events = Object.values(latestEvents);
  const totalStreams = events.length;
  const totalTracks = events.reduce((sum, evt) => sum + evt.tracks.length, 0);

  statTotalStreams.textContent = totalStreams;
  statTotalTracks.textContent = totalTracks;

  // Calculate detections per second (simplified)
  const now = Date.now();
  const timeDiff = (now - lastDetectionTime) / 1000;
  if (timeDiff > 0) {
    const detectionsPerSec = totalDetectionsCount > 0 ? (totalDetectionsCount / timeDiff).toFixed(1) : 0;
    statTotalDetections.textContent = detectionsPerSec;
  }
}

// ==================== Uptime Timer ====================
function updateUptime() {
  const now = Date.now();
  const elapsed = Math.floor((now - startTime) / 1000);
  const hours = Math.floor(elapsed / 3600).toString().padStart(2, '0');
  const minutes = Math.floor((elapsed % 3600) / 60).toString().padStart(2, '0');
  const seconds = (elapsed % 60).toString().padStart(2, '0');
  statUptime.textContent = `${hours}:${minutes}:${seconds}`;
}

// Update uptime every second
setInterval(updateUptime, 1000);

// ==================== Search and Filter ====================
function applyFilters(events) {
  return events.filter(event => {
    // Apply search filter
    const matchesSearch = !searchQuery ||
      event.stream.toLowerCase().includes(searchQuery.toLowerCase());

    // Apply status filter
    const isActive = event.tracks.length > 0;
    const matchesFilter =
      filterMode === "all" ||
      (filterMode === "active" && isActive) ||
      (filterMode === "inactive" && !isActive);

    return matchesSearch && matchesFilter;
  });
}

streamSearchInput.addEventListener("input", (e) => {
  searchQuery = e.target.value;
  renderStreams();
});

streamFilterSelect.addEventListener("change", (e) => {
  filterMode = e.target.value;
  renderStreams();
});

// ==================== Render Streams Table ====================
function renderStreams() {
  streamsBody.innerHTML = "";
  const allEvents = Object.values(latestEvents).sort(
    (a, b) => new Date(b.received_at) - new Date(a.received_at)
  );

  const events = applyFilters(allEvents);

  if (!events.length) {
    const message = searchQuery || filterMode !== "all"
      ? "No streams match the current filters"
      : "Waiting for detections...";
    streamsBody.innerHTML = `<tr class="empty"><td colspan="6">${message}</td></tr>`;
    return;
  }

  for (const event of events) {
    const row = document.createElement("tr");
    if (selectedStream === event.stream) {
      row.classList.add("active");
    }
    row.dataset.stream = event.stream;

    // Calculate FPS (estimated based on frame updates)
    const fps = event.fps || "—";

    // Determine status
    const isActive = event.tracks.length > 0;
    const statusBadge = isActive
      ? '<span style="color: #4ade80;">● Active</span>'
      : '<span style="color: #94a3b8;">○ Idle</span>';

    row.innerHTML = `
      <td><strong>${event.stream}</strong></td>
      <td>${event.frame_id}</td>
      <td><span style="color: ${event.tracks.length > 0 ? '#3b82f6' : '#94a3b8'}; font-weight: 600;">${event.tracks.length}</span></td>
      <td>${fps}</td>
      <td>${statusBadge}</td>
      <td>${new Date(event.received_at).toLocaleTimeString()}</td>
    `;

    row.addEventListener("click", () => {
      selectedStream = event.stream;
      renderStreams();
      renderTracks();
      renderPreview();
    });

    streamsBody.appendChild(row);
  }

  if (!selectedStream && events.length) {
    selectedStream = events[0].stream;
    renderTracks();
    renderPreview();
  }

  updateStatistics();
}

// ==================== Render Tracks Table ====================
function renderTracks() {
  tracksBody.innerHTML = "";
  const event = latestEvents[selectedStream];

  // Update selected stream name
  if (selectedStream) {
    selectedStreamName.textContent = selectedStream;
  } else {
    selectedStreamName.textContent = "No stream selected";
  }

  if (!event) {
    tracksBody.innerHTML =
      '<tr class="empty"><td colspan="5">Select a stream to view track details</td></tr>';
    trackCount.textContent = "0 tracks";
    return;
  }

  const tracksLength = event.tracks.length;
  trackCount.textContent = `${tracksLength} track${tracksLength !== 1 ? 's' : ''}`;

  if (!tracksLength) {
    tracksBody.innerHTML =
      '<tr class="empty"><td colspan="5">No active tracks for this stream</td></tr>';
    return;
  }

  for (const track of event.tracks) {
    const row = document.createElement("tr");

    // Calculate bounding box size
    const bbox = track.bbox_xyxy;
    const width = bbox[2] - bbox[0];
    const height = bbox[3] - bbox[1];
    const size = `${width.toFixed(0)}×${height.toFixed(0)}`;

    // Format confidence as percentage with color
    const confidence = (track.confidence * 100).toFixed(1);
    const confidenceColor = confidence >= 80 ? '#4ade80' : confidence >= 60 ? '#fbbf24' : '#f87171';

    row.innerHTML = `
      <td><strong>#${track.track_id}</strong></td>
      <td>${track.class_id}</td>
      <td><span style="color: ${confidenceColor}; font-weight: 600;">${confidence}%</span></td>
      <td style="font-family: monospace; font-size: 0.85rem;">${bbox.map((v) => v.toFixed(1)).join(", ")}</td>
      <td>${size}</td>
    `;
    tracksBody.appendChild(row);
  }
}

// ==================== Render Preview ====================
function renderPreview() {
  const event = latestEvents[selectedStream];
  if (!event) {
    previewImg.hidden = true;
    previewInfo.hidden = true;
    previewPlaceholder.hidden = false;
    previewPlaceholder.textContent = "Select a stream to view the latest annotated frame with bounding boxes.";
    return;
  }

  if (event.frame_jpeg) {
    previewImg.src = event.frame_jpeg;
    previewImg.hidden = false;
    previewPlaceholder.hidden = true;
    previewInfo.hidden = false;

    // Update preview info
    previewImg.onload = () => {
      const imgWidth = previewImg.naturalWidth;
      const imgHeight = previewImg.naturalHeight;
      previewResolution.textContent = `${imgWidth}×${imgHeight}`;
    };
    previewTimestamp.textContent = new Date(event.received_at).toLocaleString();
  } else {
    previewImg.hidden = true;
    previewInfo.hidden = true;
    previewPlaceholder.hidden = false;
    previewPlaceholder.textContent = "No frame with detections is available for this stream yet.";
  }
}

async function fetchInitialSnapshot() {
  try {
    const response = await fetch("/api/snapshot");
    if (!response.ok) {
      throw new Error("Snapshot request failed"); // 抛出异常供 catch 处理
    }
    const data = await response.json();
    latestEvents = {};
    for (const event of data.streams || []) {
      latestEvents[event.stream] = event;
    }
    renderStreams();
    renderTracks();
    renderPreview();
  } catch (error) {
    console.error("Failed to fetch snapshot:", error);
  }
}

function connectWebsocket() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${location.host}/ws`); // 根据协议选择 ws/wss

  ws.onopen = () => {
    setStatus(true);
  };

  ws.onclose = () => {
    setStatus(false);
    setTimeout(connectWebsocket, 2000); // 简单重连策略
  };

  ws.onerror = () => {
    ws.close(); // 触发 onclose 执行重连
  };

  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "snapshot") {
      latestEvents = {};
      for (const evt of message.payload.streams || []) {
        latestEvents[evt.stream] = evt;
      }
      renderStreams();
      renderTracks();
      renderPreview();
    } else if (message.type === "event") {
      const evt = message.payload;
      latestEvents[evt.stream] = evt;

      // Update detection count for statistics
      if (evt.tracks && evt.tracks.length > 0) {
        totalDetectionsCount += evt.tracks.length;
      }

      renderStreams();
      if (selectedStream === evt.stream) {
        renderTracks();
        renderPreview();
      }
    }
  };
}

setStatus(false); // 初始展示断开状态
fetchInitialSnapshot(); // 获取初始快照
connectWebsocket(); // 建立 WebSocket 通道
