(() => {
  class DeepyConnection {
    constructor(base, handlers) {
      this.base = new URL(base, location.href);
      this.handlers = handlers;
      this.source = null;
      this.pending = null;
      this.timer = null;
      this.closed = false;
      this.polling = false;
      this.pollAbort = null;
      this.cursor = 0;
    }
    async request(path, body, signal) {
      const options = body === undefined ? {} : body instanceof FormData ? {method: 'POST', body} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)};
      const response = await fetch(new URL('deepy_api/' + path, this.base), {...options, signal});
      const result = await response.json();
      if (!response.ok) {
        const error = new Error(typeof result.detail === 'string' ? result.detail : 'Request failed.');
        error.notification = result.error;
        error.status = response.status;
        if (error.status === 401) this.handlers.unauthorized?.();
        throw error;
      }
      return result;
    }
    close() {
      this.closed = true;
      this.disconnect();
      clearTimeout(this.timer);
    }
    disconnect() {
      if (this.source) {
        this.source.onopen = this.source.onmessage = this.source.onclose = this.source.onerror = null;
        this.source.close();
        this.source = null;
      }
      if (this.pollAbort) {
        this.pollAbort.abort();
        this.pollAbort = null;
      }
    }
    async poll(after) {
      if (this.closed) return;
      const abort = new AbortController();
      this.pollAbort = abort;
      try {
        const events = await this.request('events/poll?after=' + after, undefined, abort.signal);
        if (this.closed || this.pollAbort !== abort) return;
        this.handlers.connection?.(true);
        for (const event of events) {
          this.cursor = event.id;
          this.handlers.event(event);
        }
        this.pollAbort = null;
        this.timer = setTimeout(() => this.poll(this.cursor), 1000);
      } catch (error) {
        if (this.closed || this.pollAbort !== abort) return;
        this.pollAbort = null;
        this.handlers.connection?.(false);
        if (error.status !== 401) this.timer = setTimeout(() => this.connect(), 1500);
      }
    }
    fallback() {
      if (this.closed || this.polling) return;
      this.polling = true;
      this.disconnect();
      this.handlers.connection?.(false);
      this.poll(this.cursor);
    }
    connect() {
      this.closed = false;
      if (this.pending) return this.pending;
      if (this.polling && this.pollAbort) return Promise.resolve();
      // Mobile focus/visibility events must not replace a healthy live stream.
      if (this.source && (this.source.readyState === WebSocket.CONNECTING || this.source.readyState === WebSocket.OPEN)) return Promise.resolve();
      clearTimeout(this.timer);
      this.disconnect();
      this.pending = this.request('state').then(state => {
        if (this.closed) return;
        this.handlers.snapshot(state);
        this.cursor = state.cursor;
        if (this.polling) {
          this.poll(this.cursor);
          return state;
        }
        const url = new URL('deepy_api/events?after=' + state.cursor, this.base);
        url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
        // Gradio already holds heartbeat/queue SSE connections per tab. A third
        // HTTP stream exhausts the browser's six-connection pool with two tabs.
        this.source = new WebSocket(url);
        this.source.onopen = () => this.handlers.connection?.(true);
        this.source.onmessage = event => {
          const data = JSON.parse(event.data);
          this.cursor = data.id;
          this.handlers.event(data);
        };
        this.source.onerror = () => this.fallback();
        this.source.onclose = () => this.fallback();
        return state;
      }).catch(error => {
        if (this.closed) return;
        this.handlers.connection?.(false);
        if (error.status !== 401) this.retry();
        this.handlers.error?.(error);
      }).finally(() => {this.pending = null;});
      return this.pending;
    }
    retry() {
      this.disconnect();
      this.handlers.connection?.(false);
      clearTimeout(this.timer);
      if (!this.closed) this.timer = setTimeout(() => this.connect(), 1500);
    }
  }
  window.DeepyConnection = DeepyConnection;
})();
