const statusEl = document.getElementById("status"); // 状态标签
const streamsBody = document.getElementById("streams-body"); // 摄像头表格主体
const tracksBody = document.getElementById("tracks-body"); // 轨迹表格主体
const previewImg = document.getElementById("stream-preview"); // 预览图像元素
const previewPlaceholder = document.getElementById("preview-placeholder"); // 预览占位符

let selectedStream = null; // 当前选中的流名称
let latestEvents = {}; // 缓存每路流的最新事件

function setStatus(connected) {
  if (connected) {
    statusEl.textContent = "Connected";
    statusEl.classList.remove("status--disconnected");
    statusEl.classList.add("status--connected");
  } else {
    statusEl.textContent = "Disconnected";
    statusEl.classList.remove("status--connected");
    statusEl.classList.add("status--disconnected");
  }
}

function renderStreams() {
  streamsBody.innerHTML = "";
  const events = Object.values(latestEvents).sort(
    (a, b) => new Date(b.received_at) - new Date(a.received_at), // 按最新时间排序
  );

  if (!events.length) {
    streamsBody.innerHTML =
      '<tr class="empty"><td colspan="4">Waiting for detections...</td></tr>'; // 无数据时展示提示
    return;
  }

  for (const event of events) {
    const row = document.createElement("tr");
    if (selectedStream === event.stream) {
      row.classList.add("active");
    }
    row.dataset.stream = event.stream;
    row.innerHTML = `
      <td>${event.stream}</td>
      <td>${event.frame_id}</td>
      <td>${event.tracks.length}</td>
      <td>${new Date(event.received_at).toLocaleTimeString()}</td>
    `;
    row.addEventListener("click", () => {
      selectedStream = event.stream;
      renderStreams();
      renderTracks();
      renderPreview(); // 点击行后刷新右侧信息
    });
    streamsBody.appendChild(row);
  }

  if (!selectedStream && events.length) {
    selectedStream = events[0].stream;
    renderTracks();
    renderPreview();
  }
}

function renderTracks() {
  tracksBody.innerHTML = "";
  const event = latestEvents[selectedStream];
  if (!event) {
    tracksBody.innerHTML =
      '<tr class="empty"><td colspan="4">Select a stream to view track details</td></tr>'; // 尚未选择流时的提示
    return;
  }

  if (!event.tracks.length) {
    tracksBody.innerHTML =
      '<tr class="empty"><td colspan="4">No active tracks for this stream</td></tr>'; // 当前无轨迹
    return;
  }

  for (const track of event.tracks) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${track.track_id}</td>
      <td>${track.class_id}</td>
      <td>${(track.confidence * 100).toFixed(1)}%</td>
      <td>${track.bbox_xyxy.map((v) => v.toFixed(1)).join(", ")}</td>
    `;
    tracksBody.appendChild(row);
  }
}

function renderPreview() {
  const event = latestEvents[selectedStream];
  if (!event) {
    previewImg.hidden = true;
    previewPlaceholder.hidden = false;
    previewPlaceholder.textContent = "Select a stream to view the latest annotated frame."; // 默认提示
    return;
  }

  if (event.frame_jpeg) {
    previewImg.src = event.frame_jpeg;
    previewImg.hidden = false;
    previewPlaceholder.hidden = true;
  } else {
    previewImg.hidden = true;
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
        latestEvents[evt.stream] = evt; // 更新全量快照
      }
      renderStreams();
      renderTracks();
      renderPreview();
    } else if (message.type === "event") {
      const evt = message.payload;
      latestEvents[evt.stream] = evt; // 单条增量事件
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
