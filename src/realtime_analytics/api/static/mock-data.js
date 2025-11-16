// 模拟数据生成器 - 用于演示界面
// 将此脚本添加到 main.js 末尾或在控制台中运行

function generateMockData() {
  // 生成32个模拟视频流
  const mockStreams = [];
  const classNames = ['person', 'car', 'truck', 'bicycle', 'motorcycle', 'bus', 'dog', 'cat'];

  for (let i = 1; i <= 32; i++) {
    const streamName = `摄像头_${i.toString().padStart(2, '0')}`;
    const numTracks = Math.random() > 0.3 ? Math.floor(Math.random() * 8) : 0;

    const tracks = [];
    for (let j = 0; j < numTracks; j++) {
      tracks.push({
        track_id: j + 1,
        class_id: classNames[Math.floor(Math.random() * classNames.length)],
        confidence: 0.6 + Math.random() * 0.39,
        bbox_xyxy: [
          Math.floor(Math.random() * 800),
          Math.floor(Math.random() * 600),
          Math.floor(Math.random() * 800) + 100,
          Math.floor(Math.random() * 600) + 100
        ]
      });
    }

    mockStreams.push({
      stream: streamName,
      frame_id: Math.floor(Math.random() * 90000) + 10000,
      tracks: tracks,
      fps: Math.floor(Math.random() * 10) + 20,
      received_at: new Date().toISOString(),
      health: 0.7 + Math.random() * 0.3
    });
  }

  // 更新全局数据
  latestEvents = {};
  mockStreams.forEach(stream => {
    latestEvents[stream.stream] = stream;
  });

  // 选择第一个流作为默认选中
  if (mockStreams.length > 0) {
    selectedStream = mockStreams[0].stream;
  }

  // 渲染界面
  renderStreams();
  renderTracks();
  renderPreview();

  // 更新统计
  updateStatistics();

  console.log('✅ 已生成32个模拟视频流数据');

  // 每2秒更新一次数据
  setInterval(() => {
    // 随机更新5-10个流
    const numUpdates = Math.floor(Math.random() * 5) + 5;
    const streamsToUpdate = Object.keys(latestEvents)
      .sort(() => Math.random() - 0.5)
      .slice(0, numUpdates);

    streamsToUpdate.forEach(streamName => {
      const stream = latestEvents[streamName];

      // 更新帧ID
      stream.frame_id++;

      // 随机更新跟踪目标
      if (Math.random() > 0.7) {
        const numTracks = Math.floor(Math.random() * 8);
        stream.tracks = [];
        for (let j = 0; j < numTracks; j++) {
          stream.tracks.push({
            track_id: j + 1,
            class_id: classNames[Math.floor(Math.random() * classNames.length)],
            confidence: 0.6 + Math.random() * 0.39,
            bbox_xyxy: [
              Math.floor(Math.random() * 800),
              Math.floor(Math.random() * 600),
              Math.floor(Math.random() * 800) + 100,
              Math.floor(Math.random() * 600) + 100
            ]
          });
        }
      }

      // 更新FPS
      stream.fps = Math.floor(Math.random() * 10) + 20;

      // 更新时间戳
      stream.received_at = new Date().toISOString();

      // 更新健康度
      stream.health = 0.7 + Math.random() * 0.3;

      // 累计检测数
      if (stream.tracks.length > 0) {
        totalDetectionsCount += stream.tracks.length;
      }
    });

    // 如果不是暂停状态，刷新界面
    if (!isPaused) {
      renderStreams();
      if (selectedStream && latestEvents[selectedStream]) {
        renderTracks();
        renderPreview();
      }
      updateStatistics();
    }
  }, 2000);
}

// 自动启动模拟数据
setTimeout(() => {
  console.log('🚀 启动模拟数据生成器...');
  generateMockData();
}, 2000);