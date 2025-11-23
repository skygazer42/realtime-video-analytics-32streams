// ==================== Notification System ====================
// Toast notifications and alerts for user feedback

class NotificationManager {
  constructor() {
    this.container = null;
    this.init();
  }

  init() {
    // Create notification container if it doesn't exist
    if (!document.getElementById('notification-container')) {
      this.container = document.createElement('div');
      this.container.id = 'notification-container';
      this.container.className = 'notification-container';
      document.body.appendChild(this.container);
    } else {
      this.container = document.getElementById('notification-container');
    }
  }

  show(message, type = 'info', duration = 3000) {
    const notification = document.createElement('div');
    notification.className = `notification notification--${type}`;

    const icon = this.getIcon(type);
    notification.innerHTML = `
      <div class="notification-icon">${icon}</div>
      <div class="notification-content">
        <div class="notification-message">${message}</div>
      </div>
      <button class="notification-close" onclick="this.parentElement.remove()">×</button>
    `;

    this.container.appendChild(notification);

    // Trigger animation
    setTimeout(() => {
      notification.classList.add('notification--show');
    }, 10);

    // Auto-remove after duration
    if (duration > 0) {
      setTimeout(() => {
        this.remove(notification);
      }, duration);
    }

    return notification;
  }

  remove(notification) {
    notification.classList.remove('notification--show');
    notification.classList.add('notification--hide');

    setTimeout(() => {
      if (notification.parentElement) {
        notification.remove();
      }
    }, 300);
  }

  getIcon(type) {
    const icons = {
      success: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
        <polyline points="22 4 12 14.01 9 11.01"></polyline>
      </svg>`,
      error: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="15" y1="9" x2="9" y2="15"></line>
        <line x1="9" y1="9" x2="15" y2="15"></line>
      </svg>`,
      warning: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
        <line x1="12" y1="9" x2="12" y2="13"></line>
        <line x1="12" y1="17" x2="12.01" y2="17"></line>
      </svg>`,
      info: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="12" y1="16" x2="12" y2="12"></line>
        <line x1="12" y1="8" x2="12.01" y2="8"></line>
      </svg>`
    };
    return icons[type] || icons.info;
  }

  success(message, duration) {
    return this.show(message, 'success', duration);
  }

  error(message, duration) {
    return this.show(message, 'error', duration);
  }

  warning(message, duration) {
    return this.show(message, 'warning', duration);
  }

  info(message, duration) {
    return this.show(message, 'info', duration);
  }
}

// Stream Alerts Manager - monitors streams for issues
class StreamAlertsManager {
  constructor(notificationManager) {
    this.notifications = notificationManager;
    this.alertThresholds = {
      fpsDropThreshold: 0.5, // Alert if FPS drops below 50% of normal
      inactiveTimeout: 30000, // Alert if no updates for 30 seconds
      errorCountThreshold: 5, // Alert after 5 consecutive errors
    };
    this.streamStates = {};
    this.alertCooldowns = {}; // Prevent alert spam
  }

  checkStream(streamName, event, previousEvent) {
    const now = Date.now();

    // Initialize state if needed
    if (!this.streamStates[streamName]) {
      this.streamStates[streamName] = {
        lastUpdateTime: now,
        baselineFPS: event.fps || 30,
        consecutiveErrors: 0,
        alertSent: {}
      };
    }

    const state = this.streamStates[streamName];

    // Check 1: FPS drop
    if (event.fps && state.baselineFPS) {
      const fpsRatio = event.fps / state.baselineFPS;
      if (fpsRatio < this.alertThresholds.fpsDropThreshold &&
          !this.isInCooldown(streamName, 'fps_drop')) {
        this.notifications.warning(
          `Stream "${streamName}" FPS dropped to ${event.fps.toFixed(1)} (baseline: ${state.baselineFPS.toFixed(1)})`,
          5000
        );
        this.setCooldown(streamName, 'fps_drop', 60000); // 1 minute cooldown
      }
    }

    // Check 2: Inactive stream
    const timeSinceUpdate = now - state.lastUpdateTime;
    if (timeSinceUpdate > this.alertThresholds.inactiveTimeout &&
        !this.isInCooldown(streamName, 'inactive')) {
      this.notifications.warning(
        `Stream "${streamName}" has been inactive for ${Math.floor(timeSinceUpdate / 1000)}s`,
        5000
      );
      this.setCooldown(streamName, 'inactive', 120000); // 2 minute cooldown
    }

    // Update state
    state.lastUpdateTime = now;
    if (event.fps) {
      // Update baseline FPS (exponential moving average)
      state.baselineFPS = state.baselineFPS * 0.9 + event.fps * 0.1;
    }
  }

  isInCooldown(streamName, alertType) {
    const key = `${streamName}:${alertType}`;
    if (this.alertCooldowns[key]) {
      return Date.now() < this.alertCooldowns[key];
    }
    return false;
  }

  setCooldown(streamName, alertType, duration) {
    const key = `${streamName}:${alertType}`;
    this.alertCooldowns[key] = Date.now() + duration;
  }

  reportError(streamName, error) {
    if (!this.streamStates[streamName]) {
      this.streamStates[streamName] = {
        consecutiveErrors: 0,
        alertSent: {}
      };
    }

    this.streamStates[streamName].consecutiveErrors++;

    if (this.streamStates[streamName].consecutiveErrors >= this.alertThresholds.errorCountThreshold &&
        !this.isInCooldown(streamName, 'errors')) {
      this.notifications.error(
        `Stream "${streamName}" has ${this.streamStates[streamName].consecutiveErrors} consecutive errors`,
        5000
      );
      this.setCooldown(streamName, 'errors', 300000); // 5 minute cooldown
    }
  }

  reportSuccess(streamName) {
    if (this.streamStates[streamName]) {
      this.streamStates[streamName].consecutiveErrors = 0;
    }
  }
}

// Export
window.NotificationManager = NotificationManager;
window.StreamAlertsManager = StreamAlertsManager;
