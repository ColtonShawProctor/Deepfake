/**
 * Request throttling utility to prevent API request flooding
 */

class RequestThrottle {
  constructor() {
    this.pendingRequests = new Map();
    this.requestCounts = new Map();
    this.lastRequestTime = new Map();
    this.MAX_REQUESTS_PER_SECOND = 10;
    this.MIN_REQUEST_INTERVAL = 100; // milliseconds
  }

  /**
   * Check if a request should be throttled
   * @param {string} key - Unique identifier for the request type
   * @returns {boolean} - True if request should proceed, false if throttled
   */
  shouldAllowRequest(key) {
    const now = Date.now();
    const lastTime = this.lastRequestTime.get(key) || 0;
    const timeDiff = now - lastTime;

    // Check minimum interval
    if (timeDiff < this.MIN_REQUEST_INTERVAL) {
      console.warn(`Request throttled: ${key} (too frequent, ${timeDiff}ms since last)`);
      return false;
    }

    // Reset counter every second
    if (timeDiff > 1000) {
      this.requestCounts.set(key, 0);
    }

    const currentCount = this.requestCounts.get(key) || 0;
    if (currentCount >= this.MAX_REQUESTS_PER_SECOND) {
      console.warn(`Request throttled: ${key} (rate limit exceeded)`);
      return false;
    }

    // Update tracking
    this.requestCounts.set(key, currentCount + 1);
    this.lastRequestTime.set(key, now);
    return true;
  }

  /**
   * Wrap an async function with throttling
   * @param {string} key - Unique identifier for the request type
   * @param {Function} fn - Async function to wrap
   * @returns {Promise} - Result of the function or rejection if throttled
   */
  async throttle(key, fn) {
    // Check if there's already a pending request for this key
    if (this.pendingRequests.has(key)) {
      console.log(`Returning pending request for: ${key}`);
      return this.pendingRequests.get(key);
    }

    if (!this.shouldAllowRequest(key)) {
      throw new Error(`Request throttled: ${key}`);
    }

    // Create and store the promise
    const promise = fn().finally(() => {
      // Clean up pending request after completion
      this.pendingRequests.delete(key);
    });

    this.pendingRequests.set(key, promise);
    return promise;
  }

  /**
   * Clear all throttling state
   */
  reset() {
    this.pendingRequests.clear();
    this.requestCounts.clear();
    this.lastRequestTime.clear();
  }
}

// Create singleton instance
const requestThrottle = new RequestThrottle();

// Expose globally for debugging in development
if (process.env.NODE_ENV === 'development') {
  window.requestThrottle = requestThrottle;
}

export default requestThrottle;