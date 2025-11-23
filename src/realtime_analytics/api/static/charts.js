// ==================== Charts Module ====================
// Real-time data visualization for stream analytics

class StreamChartsManager {
  constructor() {
    this.detectionChart = null;
    this.fpsChart = null;
    this.healthChart = null;
    this.classChart = null;
    this.topStreamsChart = null;
    this.maxDataPoints = 60; // Show last 60 data points

    this.detectionData = {
      labels: [],
      datasets: [{
        label: 'Detections/sec',
        data: [],
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true,
        tension: 0.4
      }]
    };

    this.fpsData = {
      labels: [],
      datasets: []
    };

    this.healthData = {
      labels: [],
      datasets: [{
        label: 'Average Health Score',
        data: [],
        borderColor: '#4ade80',
        backgroundColor: 'rgba(74, 222, 128, 0.1)',
        fill: true,
        tension: 0.4
      }]
    };

    this.classData = {
      labels: [],
      datasets: [{
        data: [],
        backgroundColor: ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#0ea5e9', '#14b8a6']
      }]
    };

    this.topStreamsData = {
      labels: [],
      datasets: [{
        label: 'Tracks',
        data: [],
        backgroundColor: '#2563eb',
        borderRadius: 6
      }]
    };
  }

  initCharts() {
    // Detection rate chart
    const detectionCtx = document.getElementById('detection-chart');
    if (detectionCtx) {
      this.detectionChart = new Chart(detectionCtx, {
        type: 'line',
        data: this.detectionData,
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: true,
              position: 'top',
              labels: {
                color: getComputedStyle(document.documentElement)
                  .getPropertyValue('--text-primary').trim()
              }
            },
            tooltip: {
              mode: 'index',
              intersect: false,
            }
          },
          scales: {
            x: {
              display: true,
              title: {
                display: true,
                text: 'Time',
                color: getComputedStyle(document.documentElement)
                  .getPropertyValue('--text-secondary').trim()
              },
              ticks: {
                color: getComputedStyle(document.documentElement)
                  .getPropertyValue('--text-secondary').trim()
              },
              grid: {
                color: 'rgba(148, 163, 184, 0.1)'
              }
            },
            y: {
              display: true,
              title: {
                display: true,
                text: 'Detections per Second',
                color: getComputedStyle(document.documentElement)
                  .getPropertyValue('--text-secondary').trim()
              },
              ticks: {
                color: getComputedStyle(document.documentElement)
                  .getPropertyValue('--text-secondary').trim()
              },
              grid: {
                color: 'rgba(148, 163, 184, 0.1)'
              }
            }
          },
          interaction: {
            mode: 'nearest',
            axis: 'x',
            intersect: false
          }
        }
      });
    }

    // FPS chart (multi-stream)
    const fpsCtx = document.getElementById('fps-chart');
    if (fpsCtx) {
      this.fpsChart = new Chart(fpsCtx, {
        type: 'line',
        data: this.fpsData,
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: true,
              position: 'top',
              labels: {
                color: getComputedStyle(document.documentElement)
                  .getPropertyValue('--text-primary').trim()
              }
            },
            tooltip: {
              mode: 'index',
              intersect: false,
            }
          },
          scales: {
            x: {
              display: true,
              title: {
                display: true,
                text: 'Time',
                color: getComputedStyle(document.documentElement)
                  .getPropertyValue('--text-secondary').trim()
              },
              ticks: {
                color: getComputedStyle(document.documentElement)
                  .getPropertyValue('--text-secondary').trim()
              },
              grid: {
                color: 'rgba(148, 163, 184, 0.1)'
              }
            },
            y: {
              display: true,
              title: {
                display: true,
                text: 'FPS',
                color: getComputedStyle(document.documentElement)
                  .getPropertyValue('--text-secondary').trim()
              },
              ticks: {
                color: getComputedStyle(document.documentElement)
                  .getPropertyValue('--text-secondary').trim()
              },
              grid: {
                color: 'rgba(148, 163, 184, 0.1)'
              },
              beginAtZero: true
            }
          }
        }
      });
    }

    // Health score chart
    const healthCtx = document.getElementById('health-chart');
    if (healthCtx) {
      this.healthChart = new Chart(healthCtx, {
        type: 'line',
        data: this.healthData,
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: true,
              position: 'top',
              labels: {
                color: getComputedStyle(document.documentElement)
                  .getPropertyValue('--text-primary').trim()
              }
            },
            tooltip: {
              mode: 'index',
              intersect: false,
            }
          },
          scales: {
            x: {
              display: true,
              title: {
                display: true,
                text: 'Time',
                color: getComputedStyle(document.documentElement)
                  .getPropertyValue('--text-secondary').trim()
              },
              ticks: {
                color: getComputedStyle(document.documentElement)
                  .getPropertyValue('--text-secondary').trim()
              },
              grid: {
                color: 'rgba(148, 163, 184, 0.1)'
              }
            },
            y: {
              display: true,
              title: {
                display: true,
                text: 'Health Score',
                color: getComputedStyle(document.documentElement)
                  .getPropertyValue('--text-secondary').trim()
              },
              min: 0,
              max: 1,
              ticks: {
                color: getComputedStyle(document.documentElement)
                  .getPropertyValue('--text-secondary').trim(),
                callback: function(value) {
                  return (value * 100).toFixed(0) + '%';
                }
              },
              grid: {
                color: 'rgba(148, 163, 184, 0.1)'
              }
            }
          }
        }
      });
    }

    const classCtx = document.getElementById('class-chart');
    if (classCtx) {
      this.classChart = new Chart(classCtx, {
        type: 'doughnut',
        data: this.classData,
        options: {
          responsive: true,
          plugins: {
            legend: {
              position: 'right',
              labels: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary').trim() }
            }
          }
        }
      });
    }

    const topStreamsCtx = document.getElementById('top-streams-chart');
    if (topStreamsCtx) {
      this.topStreamsChart = new Chart(topStreamsCtx, {
        type: 'bar',
        data: this.topStreamsData,
        options: {
          indexAxis: 'y',
          responsive: true,
          plugins: {
            legend: { display: false },
            tooltip: { mode: 'index', intersect: false }
          },
          scales: {
            x: {
              beginAtZero: true,
              ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary').trim() }
            },
            y: {
              ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary').trim() }
            }
          }
        }
      });
    }
  }

  updateDetectionChart(detectionsPerSec) {
    if (!this.detectionChart) return;

    const now = new Date();
    const timeLabel = now.toLocaleTimeString();

    this.detectionData.labels.push(timeLabel);
    this.detectionData.datasets[0].data.push(detectionsPerSec);

    // Keep only last N data points
    if (this.detectionData.labels.length > this.maxDataPoints) {
      this.detectionData.labels.shift();
      this.detectionData.datasets[0].data.shift();
    }

    this.detectionChart.update('none'); // Update without animation for performance
  }

  updateFPSChart(streams) {
    if (!this.fpsChart) return;

    const now = new Date();
    const timeLabel = now.toLocaleTimeString();

    // Update time labels
    if (this.fpsData.labels.length === 0 || this.fpsData.labels[this.fpsData.labels.length - 1] !== timeLabel) {
      this.fpsData.labels.push(timeLabel);

      // Keep only last N data points
      if (this.fpsData.labels.length > this.maxDataPoints) {
        this.fpsData.labels.shift();
        // Also shift all dataset data
        this.fpsData.datasets.forEach(dataset => {
          if (dataset.data.length > 0) {
            dataset.data.shift();
          }
        });
      }
    }

    // Update or create datasets for each stream
    Object.entries(streams).forEach(([streamName, event]) => {
      let dataset = this.fpsData.datasets.find(d => d.label === streamName);

      if (!dataset) {
        // Create new dataset for this stream
        const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];
        const colorIndex = this.fpsData.datasets.length % colors.length;

        dataset = {
          label: streamName,
          data: new Array(this.fpsData.labels.length - 1).fill(null),
          borderColor: colors[colorIndex],
          backgroundColor: colors[colorIndex] + '20',
          fill: false,
          tension: 0.4
        };
        this.fpsData.datasets.push(dataset);
      }

      // Add current FPS value
      dataset.data.push(event.fps || 0);

      // Ensure data length matches labels length
      while (dataset.data.length < this.fpsData.labels.length) {
        dataset.data.unshift(null);
      }
      while (dataset.data.length > this.fpsData.labels.length) {
        dataset.data.shift();
      }
    });

    this.fpsChart.update('none');
  }

  updateHealthChart(averageHealth) {
    if (!this.healthChart) return;

    const now = new Date();
    const timeLabel = now.toLocaleTimeString();

    this.healthData.labels.push(timeLabel);
    this.healthData.datasets[0].data.push(averageHealth);

    // Keep only last N data points
    if (this.healthData.labels.length > this.maxDataPoints) {
      this.healthData.labels.shift();
      this.healthData.datasets[0].data.shift();
    }

    this.healthChart.update('none');
  }

  updateClassChart(classCounts) {
    if (!this.classChart) return;
    const entries = Object.entries(classCounts);
    if (!entries.length) {
      this.classData.labels = ['无数据'];
      this.classData.datasets[0].data = [1];
    } else {
      this.classData.labels = entries.map(([k]) => k.replace('cls-', '类 '));
      this.classData.datasets[0].data = entries.map(([, v]) => v);
    }
    this.classChart.update('none');
  }

  updateTopStreams(streamActivity) {
    if (!this.topStreamsChart) return;
    const top = [...streamActivity].sort((a, b) => b.tracks - a.tracks).slice(0, 5);
    if (!top.length) {
      this.topStreamsData.labels = ['无数据'];
      this.topStreamsData.datasets[0].data = [0];
    } else {
      this.topStreamsData.labels = top.map(t => t.name);
      this.topStreamsData.datasets[0].data = top.map(t => t.tracks);
    }
    this.topStreamsChart.update('none');
  }

  destroy() {
    if (this.detectionChart) this.detectionChart.destroy();
    if (this.fpsChart) this.fpsChart.destroy();
    if (this.healthChart) this.healthChart.destroy();
    if (this.classChart) this.classChart.destroy();
    if (this.topStreamsChart) this.topStreamsChart.destroy();
  }
}

// Export for use in main.js
window.StreamChartsManager = StreamChartsManager;
