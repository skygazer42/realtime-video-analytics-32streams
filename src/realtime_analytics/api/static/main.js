// ==================== Initialize Managers ====================
let chartsManager = null;
let notificationManager = null;
let alertsManager = null;

// Initialize after DOM loaded
document.addEventListener('DOMContentLoaded', () => {
  if (typeof NotificationManager !== 'undefined') {
    notificationManager = new NotificationManager();
  }
  if (typeof StreamChartsManager !== 'undefined') {
    chartsManager = new StreamChartsManager();
    chartsManager.initCharts();
  }
  if (typeof StreamAlertsManager !== 'undefined' && notificationManager) {
    alertsManager = new StreamAlertsManager(notificationManager);
  }
});

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
const emptyState = document.getElementById("empty-state");
const livePill = document.getElementById("live-pill");
const liveLabel = document.getElementById("live-label");
const liveStreamCount = document.getElementById("live-stream-count");
const liveTrackCount = document.getElementById("live-track-count");
const lastUpdateEl = document.getElementById("last-update");
const chipKafka = document.getElementById("chip-kafka");
const chipMetrics = document.getElementById("chip-metrics");
const chipLatency = document.getElementById("chip-latency");
const timelineBody = document.getElementById("timeline-body");
const timelineClearBtn = document.getElementById("timeline-clear");

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
let lastDetectionRate = 0;
let searchQuery = "";
let filterMode = "all";
let isPaused = false;
let currentView = "grid"; // default to rich card view
let sortColumn = null;
let sortDirection = "asc";
let currentStreamIndex = 0;
let chartsVisible = true;
let previousEvents = {}; // Track previous state for change detection
let lastUpdateTs = null;
let timeline = [];
const TIMELINE_LIMIT = 60;
let carouselTimer = null;

// ==================== Status Management ====================
function setStatus(connected) {
  if (connected) {
    statusEl.innerHTML = '<span class="status-dot"></span>实时连接';
    statusEl.classList.remove("status--disconnected");
    statusEl.classList.add("status--connected");
  } else {
    statusEl.innerHTML = '<span class="status-dot"></span>已断开';
    statusEl.classList.remove("status--connected");
    statusEl.classList.add("status--disconnected");
  }
  if (livePill) {
    livePill.className = `live-pill ${connected ? "status--connected" : "status--disconnected"}`;
  }
  if (liveLabel) {
    liveLabel.textContent = connected ? "已连接" : "已断开";
  }
}

// ==================== Statistics Update ====================
function updateStatistics() {
  const events = Object.values(latestEvents);
  const totalStreams = events.length;
  const totalTracks = events.reduce((sum, evt) => sum + evt.tracks.length, 0);
  const latestTs = events.length ? Math.max(...events.map(evt => new Date(evt.received_at).getTime())) : null;
  const classCounts = {};
  const streamActivity = [];
  const nowMs = Date.now();
  const latencies = [];

  statTotalStreams.textContent = totalStreams;
  statTotalTracks.textContent = totalTracks;
  if (liveStreamCount) liveStreamCount.textContent = totalStreams;
  if (liveTrackCount) liveTrackCount.textContent = totalTracks;
  if (latestTs) {
    lastUpdateTs = latestTs;
    if (lastUpdateEl) lastUpdateEl.textContent = new Date(latestTs).toLocaleTimeString();
  }

  // Calculate detections per second
  const now = Date.now();
  const timeDiff = (now - lastDetectionTime) / 1000;
  if (timeDiff > 0 && timeDiff < 60) { // Only update if less than 60 seconds
    const detectionsPerSec = totalDetectionsCount > 0 ? (totalDetectionsCount / timeDiff).toFixed(1) : 0;
    statTotalDetections.textContent = detectionsPerSec;
    lastDetectionRate = parseFloat(detectionsPerSec);
  }

  // Aggregate classes and latency
  events.forEach(evt => {
    evt.tracks.forEach(track => {
      const key = `cls-${track.class_id}`;
      classCounts[key] = (classCounts[key] || 0) + 1;
    });
    streamActivity.push({ name: evt.stream, tracks: evt.tracks.length });
    const latency = nowMs - new Date(evt.received_at).getTime();
    if (latency > 0) latencies.push(latency);
  });

  // Update charts if available
  if (chartsManager && !isPaused) {
    // Update detection chart
    chartsManager.updateDetectionChart(lastDetectionRate);

    // Update FPS chart with current stream data
    chartsManager.updateFPSChart(latestEvents);

    // Calculate average health score
    let totalHealth = 0;
    let healthCount = 0;
    events.forEach(event => {
      if (event.health !== undefined) {
        totalHealth += event.health;
        healthCount++;
      }
    });
    const averageHealth = healthCount > 0 ? totalHealth / healthCount : 0.8; // Default to 0.8 if no data
    chartsManager.updateHealthChart(averageHealth);

    chartsManager.updateClassChart(classCounts);
    chartsManager.updateTopStreams(streamActivity);
  }

  updateServiceChips(events, latencies);
}

// ==================== Service Chips ====================
function updateServiceChips(events, latencies) {
  if (chipKafka) {
    const dot = chipKafka.querySelector(".chip-dot");
    const active = events.length > 0;
    chipKafka.textContent = active ? " Kafka streaming" : " Kafka waiting";
    chipKafka.prepend(dot);
    dot.style.background = active ? "#22c55e" : "#f59e0b";
  }

  if (chipMetrics) {
    const dot = chipMetrics.querySelector(".chip-dot");
    const hasHealth = events.some(evt => evt.health !== undefined);
    chipMetrics.textContent = hasHealth ? " Metrics live" : " Metrics pending";
    chipMetrics.prepend(dot);
    dot.style.background = hasHealth ? "#2563eb" : "#94a3b8";
  }

  if (chipLatency) {
    const dot = chipLatency.querySelector(".chip-dot");
    if (latencies.length) {
      const avg = Math.round(
        latencies.reduce((a, b) => a + b, 0) / latencies.length
      );
      chipLatency.textContent = ` 延迟 ${avg} ms`;
      dot.style.background = avg < 300 ? "#22c55e" : avg < 800 ? "#f59e0b" : "#ef4444";
    } else {
      chipLatency.textContent = " 延迟 --";
      dot.style.background = "#94a3b8";
    }
    chipLatency.prepend(dot);
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

// ==================== Timeline ====================
function addTimelineEntry(evt) {
  const prev = previousEvents[evt.stream];
  if (prev && prev.frame_id === evt.frame_id) return;

  const topTrack = evt.tracks.reduce((best, t) => {
    if (!best) return t;
    return t.confidence > best.confidence ? t : best;
  }, null);

  timeline.unshift({
    stream: evt.stream,
    frame: evt.frame_id,
    tracks: evt.tracks.length,
    topClass: topTrack ? topTrack.class_id : null,
    confidence: topTrack ? topTrack.confidence : null,
    ts: evt.received_at,
  });

  if (timeline.length > TIMELINE_LIMIT) {
    timeline = timeline.slice(0, TIMELINE_LIMIT);
  }
}

function renderTimeline() {
  if (!timelineBody) return;
  if (!timeline.length) {
    timelineBody.innerHTML = '<div class="timeline-empty">等待检测事件...</div>';
    return;
  }

  timelineBody.innerHTML = timeline
    .map(
      (item) => `
      <div class="timeline-item">
        <div>
          <div class="timeline-stream">${item.stream} · 帧 ${item.frame}</div>
          <div class="timeline-meta">
            <span>${item.tracks} 个轨迹</span>
            <span>${item.topClass !== null ? `Top 类别 ${item.topClass}` : "无类别"}</span>
            ${
              item.confidence !== null
                ? `<span>置信度 ${(item.confidence * 100).toFixed(1)}%</span>`
                : ""
            }
          </div>
        </div>
        <div class="timeline-meta">
          <span>${new Date(item.ts).toLocaleTimeString()}</span>
        </div>
      </div>
    `
    )
    .join("");
}

if (timelineClearBtn) {
  timelineClearBtn.addEventListener("click", () => {
    timeline = [];
    renderTimeline();
  });
}

// ==================== Utility Functions ====================
// Debounce function for performance optimization
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

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

// Debounced search for better performance
const debouncedSearch = debounce((value) => {
  searchQuery = value;
  renderStreams();
}, 300);

streamSearchInput.addEventListener("input", (e) => {
  debouncedSearch(e.target.value);
});

streamFilterSelect.addEventListener("change", (e) => {
  filterMode = e.target.value;
  renderStreams();
  if (notificationManager) {
    const filterTexts = {
      'all': '全部视频流',
      'active': '仅显示活跃流',
      'inactive': '仅显示非活跃流'
    };
    notificationManager.info(`过滤器已切换为：${filterTexts[filterMode] || filterMode}`, 2000);
  }
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
      ? "暂无符合当前过滤条件的视频流"
      : "等待检测数据...";
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
      ? '<span style="color: #4ade80;">● 活跃</span>'
      : '<span style="color: #94a3b8;">○ 空闲</span>';

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
    selectedStreamName.textContent = "未选择视频流";
  }

  if (!event) {
    tracksBody.innerHTML =
      '<tr class="empty"><td colspan="5">选择一个视频流以查看跟踪详情</td></tr>';
    trackCount.textContent = "0 个目标";
    return;
  }

  const tracksLength = event.tracks.length;
  trackCount.textContent = `${tracksLength} 个目标`;

  if (!tracksLength) {
    tracksBody.innerHTML =
      '<tr class="empty"><td colspan="5">此视频流没有活跃跟踪目标</td></tr>';
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
    previewPlaceholder.textContent = "选择一个视频流以查看带有边界框的最新标注帧。";
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
    previewPlaceholder.textContent = "此视频流还没有检测到的帧可用。";
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
      addTimelineEntry(event);
    }
    renderStreams();
    renderTracks();
    renderPreview();
    renderTimeline();
    startCarousel();
  } catch (error) {
    console.error("Failed to fetch snapshot:", error);
  }
}

function connectWebsocket() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${location.host}/ws`); // 根据协议选择 ws/wss

  ws.onopen = () => {
    setStatus(true);
    if (notificationManager) {
      notificationManager.success('实时连接到服务器', 3000);
    }
    startCarousel();
  };

  ws.onclose = () => {
    setStatus(false);
    if (notificationManager) {
      notificationManager.warning('已断开连接。正在重新连接...', 3000);
    }
    setTimeout(connectWebsocket, 2000); // 简单重连策略
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    ws.close(); // 触发 onclose 执行重连
  };

  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "snapshot") {
      latestEvents = {};
      for (const evt of message.payload.streams || []) {
        latestEvents[evt.stream] = evt;

        // Check for alerts
        if (alertsManager && previousEvents[evt.stream]) {
          alertsManager.checkStream(evt.stream, evt, previousEvents[evt.stream]);
        }

        // Update previous state
        previousEvents[evt.stream] = { ...evt };
        addTimelineEntry(evt);
      }
      renderStreams();
      renderTracks();
      renderPreview();
      renderTimeline();
    } else if (message.type === "event") {
      const evt = message.payload;
      const previousEvt = latestEvents[evt.stream];

      latestEvents[evt.stream] = evt;

      // Update detection count for statistics
      if (evt.tracks && evt.tracks.length > 0) {
        totalDetectionsCount += evt.tracks.length;

        // Show notification for new detections (throttled)
        if (notificationManager && evt.tracks.length > 5) {
          const lastNotificationKey = `detection_${evt.stream}`;
          const now = Date.now();
          if (!window.lastNotifications) window.lastNotifications = {};
          if (!window.lastNotifications[lastNotificationKey] ||
              now - window.lastNotifications[lastNotificationKey] > 30000) {
            notificationManager.info(
              `在视频流 "${evt.stream}" 中检测到 ${evt.tracks.length} 个对象`,
              3000
            );
            window.lastNotifications[lastNotificationKey] = now;
          }
        }
      }

      // Check for alerts
      if (alertsManager && previousEvt) {
        alertsManager.checkStream(evt.stream, evt, previousEvt);
      }

      // Update previous state
      previousEvents[evt.stream] = { ...evt };
      addTimelineEntry(evt);

      renderStreams();
      if (selectedStream === evt.stream) {
        renderTracks();
        renderPreview();
      }
      renderTimeline();
    }
  };
}

// ==================== Table Sorting ====================
let sortedEvents = [];

function sortEvents(events, column, direction) {
  return [...events].sort((a, b) => {
    let valA, valB;

    switch (column) {
      case 'stream':
        valA = a.stream.toLowerCase();
        valB = b.stream.toLowerCase();
        break;
      case 'frame_id':
        valA = a.frame_id;
        valB = b.frame_id;
        break;
      case 'tracks':
        valA = a.tracks.length;
        valB = b.tracks.length;
        break;
      case 'fps':
        valA = a.fps || 0;
        valB = b.fps || 0;
        break;
      case 'status':
        valA = a.tracks.length > 0 ? 1 : 0;
        valB = b.tracks.length > 0 ? 1 : 0;
        break;
      case 'updated':
        valA = new Date(a.received_at);
        valB = new Date(b.received_at);
        break;
      default:
        return 0;
    }

    if (valA < valB) return direction === 'asc' ? -1 : 1;
    if (valA > valB) return direction === 'asc' ? 1 : -1;
    return 0;
  });
}

// Add click handlers to sortable headers
document.querySelectorAll('.sortable').forEach(th => {
  th.addEventListener('click', () => {
    const column = th.dataset.sort;

    if (sortColumn === column) {
      sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
      sortColumn = column;
      sortDirection = 'asc';
    }

    // Update UI
    document.querySelectorAll('.sortable').forEach(header => {
      header.classList.remove('sorted-asc', 'sorted-desc');
    });
    th.classList.add(`sorted-${sortDirection}`);

    renderStreams();
  });
});

// ==================== View Toggle ====================
const viewToggleBtn = document.getElementById('view-toggle');
const streamsTableView = document.getElementById('streams-table-view');
const streamsGridView = document.getElementById('streams-grid-view');

if (currentView === 'grid') {
  streamsTableView.hidden = true;
  streamsGridView.hidden = false;
}

viewToggleBtn.addEventListener('click', () => {
  currentView = currentView === 'table' ? 'grid' : 'table';
  if (currentView === 'grid') {
    streamsTableView.hidden = true;
    streamsGridView.hidden = false;
  } else {
    streamsTableView.hidden = false;
    streamsGridView.hidden = true;
  }
  renderStreams();
});

function renderGridView(events) {
  streamsGridView.innerHTML = '';

  if (!events.length) {
    const message = searchQuery || filterMode !== "all"
      ? "没有符合当前过滤条件的视频流"
      : "等待检测数据...";
    streamsGridView.innerHTML = `<div style="grid-column: 1/-1; text-align: center; color: #94a3b8; padding: 40px;">${message}</div>`;
    return;
  }

  for (const event of events) {
    const card = document.createElement('div');
    card.className = 'stream-card';
    if (selectedStream === event.stream) {
      card.classList.add('active');
    }

    const isActive = event.tracks.length > 0;
    const health = event.health !== undefined ? event.health : 0.82;
    const healthPct = Math.round(health * 100);
    const updatedAt = new Date(event.received_at).toLocaleTimeString();
    const statusBadge = isActive
      ? '<span style="color: #22c55e;">● 活跃</span>'
      : '<span style="color: #94a3b8;">● 空闲</span>';

    card.innerHTML = `
      <div class="stream-card-header">
        <div class="stream-card-title">${event.stream}</div>
        <div class="stream-card-status">${statusBadge}</div>
      </div>
      <div class="stream-card-stats">
        <div class="stream-card-stat">
          <div class="stream-card-stat-label">Frame</div>
          <div class="stream-card-stat-value">${event.frame_id}</div>
        </div>
        <div class="stream-card-stat">
          <div class="stream-card-stat-label">Tracks</div>
          <div class="stream-card-stat-value" style="color: ${event.tracks.length > 0 ? '#2563eb' : '#94a3b8'};">${event.tracks.length}</div>
        </div>
        <div class="stream-card-stat">
          <div class="stream-card-stat-label">FPS</div>
          <div class="stream-card-stat-value">${event.fps || '–'}</div>
        </div>
        <div class="stream-card-stat">
          <div class="stream-card-stat-label">Updated</div>
          <div class="stream-card-stat-value" style="font-size: 0.9rem;">${updatedAt}</div>
        </div>
      </div>
      <div class="stream-card-health">
        <div class="health-bar"><span style="width:${healthPct}%;"></span></div>
        <div class="health-meta">健康 ${healthPct}% · 更新时间 ${updatedAt}</div>
      </div>
    `;

    card.addEventListener('click', () => {
      selectedStream = event.stream;
      renderStreams();
      renderTracks();
      renderPreview();
    });

    streamsGridView.appendChild(card);
  }

  if (!selectedStream && events.length) {
    selectedStream = events[0].stream;
    renderTracks();
    renderPreview();
  }

  updateStatistics();
}
// ==================== Pause/Resume ====================
const pauseToggleBtn = document.getElementById('pause-toggle');

pauseToggleBtn.addEventListener('click', togglePause);

function togglePause() {
  isPaused = !isPaused;

  if (isPaused) {
    pauseToggleBtn.classList.add('active');
    pauseToggleBtn.innerHTML = `
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polygon points="5 3 19 12 5 21 5 3"></polygon>
      </svg>
    `;
    pauseToggleBtn.title = '恢复更新 (空格键)';
  } else {
    pauseToggleBtn.classList.remove('active');
    pauseToggleBtn.innerHTML = `
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="6" y="4" width="4" height="16"></rect>
        <rect x="14" y="4" width="4" height="16"></rect>
      </svg>
    `;
    pauseToggleBtn.title = '暂停更新 (空格键)';
  }
}

// ==================== Theme Toggle ====================
const themeToggleBtn = document.getElementById('theme-toggle');
let currentTheme = localStorage.getItem('theme') || 'light';

// Apply saved theme
if (currentTheme === 'dark') {
  document.body.classList.add('dark-theme');
}

themeToggleBtn.addEventListener('click', toggleTheme);

function toggleTheme() {
  currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
  document.body.classList.toggle('dark-theme');
  localStorage.setItem('theme', currentTheme);
}

// ==================== Fullscreen Preview ====================
const fullscreenBtn = document.getElementById('fullscreen-btn');
const fullscreenModal = document.getElementById('fullscreen-modal');
const fullscreenImg = document.getElementById('fullscreen-img');
const closeFullscreen = document.getElementById('close-fullscreen');
const fullscreenStreamEl = document.getElementById('fullscreen-stream');
const fullscreenResolutionEl = document.getElementById('fullscreen-resolution');
const fullscreenTimestampEl = document.getElementById('fullscreen-timestamp');

fullscreenBtn.addEventListener('click', openFullscreen);
closeFullscreen.addEventListener('click', closeFullscreenModal);
fullscreenModal.addEventListener('click', (e) => {
  if (e.target === fullscreenModal) {
    closeFullscreenModal();
  }
});

function openFullscreen() {
  const event = latestEvents[selectedStream];
  if (!event || !event.frame_jpeg) return;

  fullscreenImg.src = event.frame_jpeg;
  fullscreenStreamEl.textContent = event.stream;
  fullscreenTimestampEl.textContent = new Date(event.received_at).toLocaleString();

  fullscreenImg.onload = () => {
    const imgWidth = fullscreenImg.naturalWidth;
    const imgHeight = fullscreenImg.naturalHeight;
    fullscreenResolutionEl.textContent = `${imgWidth}×${imgHeight}`;
  };

  fullscreenModal.hidden = false;
}

function closeFullscreenModal() {
  fullscreenModal.hidden = true;
}

// ==================== Help Modal ====================
const helpBtn = document.getElementById('help-btn');
const helpModal = document.getElementById('help-modal');
const closeHelp = document.getElementById('close-help');

helpBtn.addEventListener('click', () => {
  helpModal.hidden = false;
});

closeHelp.addEventListener('click', () => {
  helpModal.hidden = true;
});

helpModal.addEventListener('click', (e) => {
  if (e.target === helpModal) {
    helpModal.hidden = true;
  }
});

// ==================== Export Modal ====================
const exportBtn = document.getElementById('export-btn');
const exportModal = document.getElementById('export-modal');
const closeExport = document.getElementById('close-export');
const exportJsonBtn = document.getElementById('export-json');
const exportCsvBtn = document.getElementById('export-csv');

function showExportModal() {
  exportModal.hidden = false;
}

function hideExportModal() {
  exportModal.hidden = true;
}

exportBtn.addEventListener('click', showExportModal);
closeExport.addEventListener('click', hideExportModal);

exportModal.addEventListener('click', (e) => {
  if (e.target === exportModal) {
    hideExportModal();
  }
});

window.addEventListener('keydown', (e) => {
  if (!exportModal.hidden && e.key === 'Escape') {
    hideExportModal();
  }
});

exportJsonBtn.addEventListener('click', exportAsJSON);
exportCsvBtn.addEventListener('click', exportAsCSV);

function exportAsJSON() {
  const allEvents = Object.values(latestEvents);
  const filteredEvents = applyFilters(allEvents);

  const exportData = {
    exported_at: new Date().toISOString(),
    stream_count: filteredEvents.length,
    streams: filteredEvents.map(event => ({
      stream: event.stream,
      frame_id: event.frame_id,
      timestamp: event.received_at,
      fps: event.fps || null,
      tracks: event.tracks.map(track => ({
        track_id: track.track_id,
        class_id: track.class_id,
        confidence: track.confidence,
        bbox_xyxy: track.bbox_xyxy
      }))
    }))
  };

  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `video-analytics-${new Date().toISOString().split('T')[0]}.json`;
  a.click();
  URL.revokeObjectURL(url);

  exportModal.hidden = true;
}

function exportAsCSV() {
  const allEvents = Object.values(latestEvents);
  const filteredEvents = applyFilters(allEvents);

  const rows = [['Stream', 'Frame ID', 'Timestamp', 'FPS', 'Track ID', 'Class', 'Confidence', 'BBox X1', 'BBox Y1', 'BBox X2', 'BBox Y2']];

  filteredEvents.forEach(event => {
    if (event.tracks.length === 0) {
      rows.push([
        event.stream,
        event.frame_id,
        event.received_at,
        event.fps || '',
        '',
        '',
        '',
        '',
        '',
        '',
        ''
      ]);
    } else {
      event.tracks.forEach(track => {
        rows.push([
          event.stream,
          event.frame_id,
          event.received_at,
          event.fps || '',
          track.track_id,
          track.class_id,
          track.confidence,
          track.bbox_xyxy[0],
          track.bbox_xyxy[1],
          track.bbox_xyxy[2],
          track.bbox_xyxy[3]
        ]);
      });
    }
  });

  const csv = rows.map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `video-analytics-${new Date().toISOString().split('T')[0]}.csv`;
  a.click();
  URL.revokeObjectURL(url);

  exportModal.hidden = true;
}

// ==================== Keyboard Shortcuts ====================
document.addEventListener('keydown', (e) => {
  // Ignore if typing in input
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
    if (e.key === 'Escape') {
      e.target.blur();
    }
    return;
  }

  switch (e.key.toLowerCase()) {
    case ' ':
      e.preventDefault();
      togglePause();
      break;
    case 'e':
      exportModal.hidden = false;
      break;
    case 'f':
      if (!fullscreenModal.hidden) {
        closeFullscreenModal();
      } else {
        openFullscreen();
      }
      break;
    case 'v':
      viewToggleBtn.click();
      break;
    case 't':
      toggleTheme();
      break;
    case '/':
      e.preventDefault();
      streamSearchInput.focus();
      break;
    case '?':
      helpModal.hidden = !helpModal.hidden;
      break;
    case 'escape':
      // Close all modals
      helpModal.hidden = true;
      exportModal.hidden = true;
      closeFullscreenModal();
      break;
    case 'arrowup':
    case 'arrowdown':
      e.preventDefault();
      navigateStreams(e.key === 'arrowup' ? -1 : 1);
      break;
    case 'enter':
      e.preventDefault();
      selectCurrentStream();
      break;
  }
});

function startCarousel() {
  if (carouselTimer) clearInterval(carouselTimer);
  carouselTimer = setInterval(() => {
    if (isPaused) return;
    navigateStreams(1);
  }, 8000);
}

function navigateStreams(direction) {
  const allEvents = Object.values(latestEvents);
  const filteredEvents = applyFilters(allEvents);

  if (!filteredEvents.length) return;

  currentStreamIndex += direction;
  if (currentStreamIndex < 0) currentStreamIndex = filteredEvents.length - 1;
  if (currentStreamIndex >= filteredEvents.length) currentStreamIndex = 0;

  selectedStream = filteredEvents[currentStreamIndex].stream;
  renderStreams();
  renderTracks();
  renderPreview();
}

function selectCurrentStream() {
  const allEvents = Object.values(latestEvents);
  const filteredEvents = applyFilters(allEvents);

  if (filteredEvents.length > currentStreamIndex) {
    selectedStream = filteredEvents[currentStreamIndex].stream;
    renderStreams();
    renderTracks();
    renderPreview();
  }
}

// ==================== Update Render Functions ====================
// Override renderStreams to support sorting and grid view
const originalRenderStreams = renderStreams;

function renderStreams() {
  if (isPaused) return; // Don't update if paused

  streamsBody.innerHTML = "";
  let allEvents = Object.values(latestEvents).sort(
    (a, b) => new Date(b.received_at) - new Date(a.received_at)
  );

  // Apply sorting if active
  if (sortColumn) {
    allEvents = sortEvents(allEvents, sortColumn, sortDirection);
  }

  const events = applyFilters(allEvents);

  if (currentView === 'grid') {
    renderGridView(events);
    updateStatistics();
    return;
  }

  // Table view rendering
  if (!events.length) {
    const message = searchQuery || filterMode !== "all"
      ? "没有符合当前过滤条件的视频流"
      : "等待检测数据...";
    streamsBody.innerHTML = `<tr class="empty"><td colspan="6">${message}</td></tr>`;
    return;
  }

  for (const event of events) {
    const row = document.createElement("tr");
    if (selectedStream === event.stream) {
      row.classList.add("active");
    }
    row.dataset.stream = event.stream;

    const fps = event.fps || "–";
    const isActive = event.tracks.length > 0;
    const statusBadge = isActive
      ? '<span style="color: #22c55e;">● 活跃</span>'
      : '<span style="color: #94a3b8;">○ 空闲</span>';

    row.innerHTML = `
      <td><strong>${event.stream}</strong></td>
      <td>${event.frame_id}</td>
      <td><span style="color: ${event.tracks.length > 0 ? '#2563eb' : '#94a3b8'}; font-weight: 600;">${event.tracks.length}</span></td>
      <td>${fps}</td>
      <td>${statusBadge}</td>
      <td>${new Date(event.received_at).toLocaleTimeString()}</td>
    `;

    row.addEventListener("click", () => {
      selectedStream = event.stream;
      currentStreamIndex = events.findIndex(e => e.stream === event.stream);
      renderStreams();
      renderTracks();
      renderPreview();
    });

    streamsBody.appendChild(row);
  }

  if (!selectedStream && events.length) {
    selectedStream = events[0].stream;
    currentStreamIndex = 0;
    renderTracks();
    renderPreview();
  }

  updateStatistics();
}
// Update renderPreview to show/hide fullscreen button
const originalRenderPreview = renderPreview;

function renderPreview() {
  const event = latestEvents[selectedStream];
  if (!event) {
    previewImg.hidden = true;
    previewInfo.hidden = true;
    fullscreenBtn.hidden = true;
    previewPlaceholder.hidden = false;
    previewPlaceholder.textContent = "Select a stream to view the latest annotated frame with bounding boxes.";
    return;
  }

  if (event.frame_jpeg) {
    previewImg.src = event.frame_jpeg;
    previewImg.hidden = false;
    previewPlaceholder.hidden = true;
    previewInfo.hidden = false;
    fullscreenBtn.hidden = false;

    previewImg.onload = () => {
      const imgWidth = previewImg.naturalWidth;
      const imgHeight = previewImg.naturalHeight;
      previewResolution.textContent = `${imgWidth}×${imgHeight}`;
    };
    previewTimestamp.textContent = new Date(event.received_at).toLocaleString();
  } else {
    previewImg.hidden = true;
    previewInfo.hidden = true;
    fullscreenBtn.hidden = true;
    previewPlaceholder.hidden = false;
    previewPlaceholder.textContent = "No frame with detections is available for this stream yet.";
  }
}

// ==================== Live overlay helpers ====================
function updateLiveOverlay() {
  const events = applyFilters(Object.values(latestEvents));
  const streamCount = events.length;
  const trackCountTotal = events.reduce((sum, evt) => sum + evt.tracks.length, 0);
  if (liveStreamCount) liveStreamCount.textContent = streamCount;
  if (liveTrackCount) liveTrackCount.textContent = trackCountTotal;
  if (emptyState) emptyState.hidden = streamCount !== 0;
  const latestTs = events.length ? Math.max(...events.map(evt => new Date(evt.received_at).getTime())) : null;
  if (latestTs && lastUpdateEl) {
    lastUpdateEl.textContent = new Date(latestTs).toLocaleTimeString();
  }
}

setInterval(updateLiveOverlay, 1500);

// ==================== Charts Toggle ====================
const chartsToggleBtn = document.getElementById('charts-toggle');
const chartsContainer = document.getElementById('charts-container');

if (chartsToggleBtn && chartsContainer) {
  chartsToggleBtn.addEventListener('click', () => {
    chartsVisible = !chartsVisible;
    chartsContainer.hidden = !chartsVisible;

    if (chartsVisible && chartsManager) {
      // Reinitialize charts if they were hidden
      setTimeout(() => {
        chartsManager.initCharts();
      }, 100);
    }
  });
}

// ==================== Additional Keyboard Shortcuts ====================
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
    return;
  }

  // 'C' - Toggle charts
  if (e.key.toLowerCase() === 'c') {
    if (chartsToggleBtn) {
      chartsToggleBtn.click();
    }
  }

  // 'R' - Refresh data
  if (e.key.toLowerCase() === 'r') {
    if (notificationManager) {
      notificationManager.info('正在刷新数据...', 2000);
    }
    fetchInitialSnapshot();
  }

  // 'A' - Select all streams (show info)
  if (e.key.toLowerCase() === 'a' && e.ctrlKey) {
    e.preventDefault();
    if (notificationManager) {
      const streamCount = Object.keys(latestEvents).length;
      notificationManager.info(`总计：${streamCount} 个活跃视频流`, 3000);
    }
  }
});

// ==================== Periodic Chart Updates ====================
// Update charts every 5 seconds
setInterval(() => {
  if (chartsManager && !isPaused && chartsVisible) {
    updateStatistics();
  }
}, 5000);

// Enhance connection indicator to同步 live bar
const __origSetStatus = setStatus;
setStatus = function(connected) {
  __origSetStatus(connected);
  if (livePill) {
    livePill.className = `live-pill ${connected ? "status--connected" : "status--disconnected"}`;
  }
  if (liveLabel) {
    liveLabel.textContent = connected ? "Connected" : "Disconnected";
  }
};

setStatus(false); // 初始展示断开状态
fetchInitialSnapshot(); // 获取初始快照
connectWebsocket(); // 建立 WebSocket 通道
