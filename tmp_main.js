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
let currentView = "table"; // "table" or "grid"
let sortColumn = null;
let sortDirection = "asc";
let currentStreamIndex = 0;
let chartsVisible = true;
let previousEvents = {}; // Track previous state for change detection

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
}

// ==================== Statistics Update ====================
function updateStatistics() {
  const events = Object.values(latestEvents);
  const totalStreams = events.length;
  const totalTracks = events.reduce((sum, evt) => sum + evt.tracks.length, 0);

  statTotalStreams.textContent = totalStreams;
  statTotalTracks.textContent = totalTracks;

  // Calculate detections per second
  const now = Date.now();
  const timeDiff = (now - lastDetectionTime) / 1000;
  if (timeDiff > 0 && timeDiff < 60) { // Only update if less than 60 seconds
    const detectionsPerSec = totalDetectionsCount > 0 ? (totalDetectionsCount / timeDiff).toFixed(1) : 0;
    statTotalDetections.textContent = detectionsPerSec;
    lastDetectionRate = parseFloat(detectionsPerSec);
  }

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
      'all': '全部视频�?,
      'active': '仅显示活跃流',
      'inactive': '仅显示非活跃�?
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
      : "等待检测数�?..";
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
    const fps = event.fps || "�?;

    // Determine status
    const isActive = event.tracks.length > 0;
    const statusBadge = isActive
      ? '<span style="color: #4ade80;">�?活跃</span>'
      : '<span style="color: #94a3b8;">�?空闲</span>';

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
    selectedStreamName.textContent = "未选择视频�?;
  }

  if (!event) {
    tracksBody.innerHTML =
      '<tr class="empty"><td colspan="5">选择一个视频流以查看跟踪详�?/td></tr>';
    trackCount.textContent = "0 个目�?;
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
    previewPlaceholder.textContent = "选择一个视频流以查看带有边界框的最新标注帧�?;
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
    previewPlaceholder.textContent = "此视频流还没有检测到的帧可用�?;
  }
}

async function fetchInitialSnapshot() {
  try {
    const response = await fetch("/api/snapshot");
    if (!response.ok) {
      throw new Error("Snapshot request failed"); // 抛出异常�?catch 处理
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
    if (notificationManager) {
      notificationManager.success('实时连接到服务器', 3000);
    }
  };

  ws.onclose = () => {
    setStatus(false);
    if (notificationManager) {
      notificationManager.warning('已断开连接。正在重新连�?..', 3000);
    }
    setTimeout(connectWebsocket, 2000); // 简单重连策�?
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
      }
      renderStreams();
      renderTracks();
      renderPreview();
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

      renderStreams();
      if (selectedStream === evt.stream) {
        renderTracks();
        renderPreview();
      }
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
      ? "暂无符合当前过滤条件的视频流"
      : "等待检测数�?..";
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
    const statusBadge = isActive
      ? '<span style="color: #4ade80;">�?Active</span>'
      : '<span style="color: #94a3b8;">�?Idle</span>';

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
          <div class="stream-card-stat-value" style="color: ${event.tracks.length > 0 ? '#3b82f6' : '#94a3b8'};">${event.tracks.length}</div>
        </div>
        <div class="stream-card-stat">
          <div class="stream-card-stat-label">FPS</div>
          <div class="stream-card-stat-value">${event.fps || '�?}</div>
        </div>
        <div class="stream-card-stat">
          <div class="stream-card-stat-label">Updated</div>
          <div class="stream-card-stat-value" style="font-size: 0.9rem;">${new Date(event.received_at).toLocaleTimeString()}</div>
        </div>
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
    pauseToggleBtn.title = '恢复更新 (空格�?';
  } else {
    pauseToggleBtn.classList.remove('active');
    pauseToggleBtn.innerHTML = `
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="6" y="4" width="4" height="16"></rect>
        <rect x="14" y="4" width="4" height="16"></rect>
      </svg>
    `;
    pauseToggleBtn.title = '暂停更新 (空格�?';
  }
}

// ==================== Theme Toggle ====================
const themeToggleBtn = document.getElementById('theme-toggle');
let currentTheme = localStorage.getItem('theme') || 'dark';

// Apply saved theme
if (currentTheme === 'light') {
  document.body.classList.add('light-theme');
}

themeToggleBtn.addEventListener('click', toggleTheme);

function toggleTheme() {
  currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
  document.body.classList.toggle('light-theme');
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
      ? "暂无符合当前过滤条件的视频流"
      : "等待检测数�?..";
    streamsBody.innerHTML = `<tr class="empty"><td colspan="6">${message}</td></tr>`;
    return;
  }

  for (const event of events) {
    const row = document.createElement("tr");
    if (selectedStream === event.stream) {
      row.classList.add("active");
    }
    row.dataset.stream = event.stream;

    const fps = event.fps || "�?;
    const isActive = event.tracks.length > 0;
    const statusBadge = isActive
      ? '<span style="color: #4ade80;">�?Active</span>'
      : '<span style="color: #94a3b8;">�?Idle</span>';

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
      notificationManager.info(`总计�?{streamCount} 个活跃视频流`, 3000);
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

setStatus(false); // 初始展示断开状�?
fetchInitialSnapshot(); // 获取初始快照
connectWebsocket(); // 建立 WebSocket 通道

