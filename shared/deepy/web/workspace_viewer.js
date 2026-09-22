/* Full workspace management is available only in the native Gradio interface. */
(() => {
  const icons = {search:'<circle cx="10" cy="10" r="6"/><path d="m15 15 6 6"/>', close:'<path d="m6 6 12 12M6 18 18 6"/>', import:'<path d="M12 16V3m-5 5 5-5 5 5M4 15v6h16v-6"/>', eject:'<path d="m12 3 8 11H4l8-11ZM4 20h16"/>', delete:'<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/>', copy:'<rect x="8" y="8" width="13" height="13" rx="2"/><path d="M16 8V3H3v13h5"/>', archive:'<path d="M3 3h18v5H3zM5 8v13h14V8M10 12h4"/>'};
  const svg = key => `<svg viewBox="0 0 24 24" aria-hidden="true">${icons[key]}</svg>`;
  icons.broom = '<path d="m20 3-8 10m-3-2 6 5-5 6-8-7 7-4Zm-3 5 4 4"/>';
  icons.lock = '<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V6a4 4 0 0 1 8 0v4M12 14v3"/>';
  icons.unlock = '<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V6a4 4 0 0 1 8 0M12 14v3"/>';
  icons.create = '<path d="M12 5v14M5 12h14"/>';
  icons.move = '<path d="M14 3H3v18h11M9 12h12m-5-5 5 5-5 5"/>';
  class WorkspaceViewer {
    constructor(picker, transport) {
      this.picker = picker; this.transport = transport; this.source = 'video'; this.selections = {video: new Set(), audio: new Set()}; this.requestId = 0;
      const button = document.createElement('button'); button.type = 'button'; button.title = button.ariaLabel = 'Explore workspace'; button.innerHTML = svg('search');
      picker.node.append(button); button.onclick = event => {event.stopPropagation(); this.open();};
      this.dialog = document.createElement('dialog'); this.dialog.className = 'wangp-workspace-viewer'; this.dialog.ariaLabel = 'Workspace media viewer';
      this.dialog.innerHTML = `<header><div class="wv-heading"><div class="wv-workspace-picker"><select data-workspace aria-label="Workspace"></select><button data-create aria-label="Add workspace" title="Add workspace">${svg('create')}</button></div><span data-summary></span></div><div class="wv-header-actions"><button data-retention aria-label="Automatic workspace archiving" title="Automatic workspace archiving">${svg('broom')}</button><button data-close aria-label="Close workspace viewer" title="Close (Esc)">${svg('close')}</button></div></header>
        <div class="wv-toolbar"><div role="tablist" aria-label="Workspace media"><button role="tab" data-source="video">Images / Videos</button><button role="tab" data-source="audio">Audio</button></div><span class="wv-spacer"></span><span data-activity></span><button data-protect aria-label="Protect workspace from automatic archiving" aria-pressed="false">${svg('unlock')}</button><button data-import>${svg('import')}Import</button><button data-refresh>Refresh</button><input data-files type="file" accept="image/*,video/*,audio/*" multiple hidden></div>
        <div class="wv-selection"><span data-count></span><button data-page-select>Select page</button><button data-clear>Clear selection</button><div data-actions hidden><button data-action="eject">${svg('eject')}Eject</button><button data-action="delete">${svg('delete')}Delete files</button><button data-action="copy">${svg('copy')}Copy to workspace</button><button data-action="move">${svg('move')}Move to workspace</button><button data-action="archive">${svg('archive')}ZIP</button><button data-action="first" title="Move selected media to the oldest end">Move to start</button><button data-action="last" title="Move selected media to the newest end">Move to end</button></div></div>
        <p class="wv-notice" role="status" hidden></p><div class="wv-content"><section class="wv-browser" aria-label="Workspace media grid"><div class="wv-grid" role="listbox" aria-multiselectable="true" aria-label="Media"></div><div class="wv-empty" hidden>No media in this gallery. Import files to get started.</div><div class="wv-rubber" hidden></div></section><aside><h3>Media details</h3><div class="wv-preview"></div><iframe title="Generation properties" sandbox=""></iframe><button data-extract-settings disabled title="Load the selected media's generation settings and close the workspace manager">Extract Settings</button></aside></div>
        <footer><span>Oldest → newest · Ctrl/⌘ click or drag on empty space to select · Drag tiles to reorder</span><button data-prev aria-label="Older page">←</button><label>Page <input data-page type="number" min="1" aria-label="Page number"></label><span data-pages></span><button data-next aria-label="Newer page">→</button></footer>`;
      document.body.append(this.dialog);
      this.grid = this.dialog.querySelector('.wv-grid'); this.browser = this.dialog.querySelector('.wv-browser');
      this.info = this.dialog.querySelector('iframe'); this.preview = this.dialog.querySelector('.wv-preview');
      this.modal = document.createElement('dialog'); this.modal.className = 'wangp-workspace-dialog';
      this.modal.innerHTML = '<form><h2></h2><div class="wangp-workspace-body"><p></p><label data-name-label>Archive name<input maxlength="120" required></label><label data-target-label><span data-select-label>Target workspace</span><select required></select></label><div class="wangp-workspace-error" role="alert" hidden></div><div class="wangp-workspace-actions"><button type="button" data-cancel>Cancel</button><button type="submit">Confirm</button></div></div></form>';
      document.body.append(this.modal);
      this.modal.querySelector('[data-cancel]').onclick = () => this.modal.close();
      this.modal.querySelector('form').onsubmit = event => {event.preventDefault(); this.confirm();};
      this.dialog.querySelector('[data-close]').onclick = () => this.dialog.close();
      this.dialog.querySelector('[data-extract-settings]').onclick = () => {
        if (this.busy || this.loading || this.selected.size !== 1) return;
        window.__wangpAssistantChatNS.setBridgeValue('#wangp-workspace-extract-settings textarea', JSON.stringify({workspace: this.workspace, source: this.source, revision: this.state.revision, keys: [...this.selected], request: Date.now()}));
        this.dialog.close();
      };
      this.dialog.querySelector('[data-workspace]').onchange = event => this.changeWorkspace(event.target.value);
      this.dialog.querySelector('[data-create]').onclick = () => this.picker.open('create');
      this.picker.dialog.addEventListener('close', () => this.invalidate());
      this.dialog.querySelector('[data-protect]').onclick = () => this.protect();
      this.dialog.querySelector('[data-retention]').onclick = () => this.action('retention').catch(error => this.notice(error.message));
      this.dialog.onclose = () => {this.requestId++; this.detailRequest?.abort(); this.grid.replaceChildren(); this.preview.replaceChildren(); this.modal.close(); clearTimeout(this.refreshTimer); clearTimeout(this.pageHover); cancelAnimationFrame(this.rubberFrame);};
      this.dialog.querySelectorAll('[data-source]').forEach(tab => tab.onclick = () => {if (this.source !== tab.dataset.source) {this.source = tab.dataset.source; this.detailKey = null; this.preview.replaceChildren(); this.info.srcdoc = WanGPMediaView.documentHtml('Select media to view its properties.'); this.load(this.pages[this.source]);}});
      this.dialog.querySelector('[data-import]').onclick = () => this.dialog.querySelector('[data-files]').click();
      this.dialog.querySelector('[data-files]').onchange = event => this.importFiles([...event.target.files]);
      this.dialog.querySelector('[data-refresh]').onclick = () => this.load(this.state?.page || 0);
      this.dialog.querySelector('[data-prev]').onclick = () => this.load(this.state.page - 1);
      this.dialog.querySelector('[data-next]').onclick = () => this.load(this.state.page + 1);
      this.dialog.querySelector('[data-page]').onchange = event => this.load(Number(event.target.value) - 1);
      this.dialog.querySelector('[data-page-select]').onclick = () => {this.state.items.forEach(item => this.selected.add(item.key)); this.selectionChanged();};
      this.dialog.querySelector('[data-clear]').onclick = () => {this.selected.clear(); this.selectionChanged();};
      this.dialog.querySelectorAll('[data-action]').forEach(button => button.onclick = () => this.action(button.dataset.action).catch(error => this.notice(error.message)));
      this.grid.onclick = event => {
        if (this.suppressClick) {this.suppressClick = false; return;}
        const tile = event.target.closest('[data-key]');
        if (tile) this.choose(tile.dataset.key, event);
      };
      this.grid.onkeydown = event => {
        if (event.target.closest('button')) return;
        if (event.key === ' ' || event.key === 'Enter') {const tile = event.target.closest('[data-key]'); if (tile) {event.preventDefault(); this.choose(tile.dataset.key, event);}}
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'a') {event.preventDefault(); this.state.items.forEach(item => this.selected.add(item.key)); this.selectionChanged();}
      };
      this.installDrag(); this.installRubber();
    }
    get selected() {return this.selections[this.source];}
    query(path, extra = {}) {return path + '?' + new URLSearchParams({workspace: this.workspace, source: this.source, ...extra});}
    notice(text) {const node = this.dialog.querySelector('.wv-notice'); node.textContent = text; node.hidden = !text;}
    navigation() {
      this.dialog.querySelector('[data-prev]').disabled = this.busy || this.loading || !this.state || this.state.page === 0;
      this.dialog.querySelector('[data-next]').disabled = this.busy || this.loading || !this.state || this.state.page + 1 === this.state.pages;
      this.dialog.querySelector('[data-page]').disabled = this.busy || this.loading || !this.state;
      this.dialog.querySelector('[data-protect]').disabled = this.busy || this.loading || !this.state;
      this.dialog.querySelector('[data-extract-settings]').disabled = this.busy || this.loading || !this.state || this.selected.size !== 1;
    }
    setBusy(value) {
      this.busy = value;
      this.dialog.querySelectorAll('button:not([data-close])').forEach(button => button.disabled = value);
      this.dialog.querySelector('[data-workspace]').disabled = value;
      if (value) this.dialog.setAttribute('aria-busy', 'true'); else this.dialog.removeAttribute('aria-busy');
      this.navigation();
    }
    open() {
      this.dialog.showModal(); this.resetWorkspace();
    }
    renderWorkspaces() {
      const select = this.dialog.querySelector('[data-workspace]'), choices = JSON.stringify(this.picker.state.items);
      if (this.choices !== choices) {select.replaceChildren(...this.picker.state.items.map(item => new Option(WanGPWorkspacePicker.label(item), item.id))); this.choices = choices;}
      select.value = this.picker.state.selected;
      select.title = select.selectedOptions[0].textContent;
    }
    renderActivity(state) {
      const date = new Date(state.last_activity * 1000), activity = this.dialog.querySelector('[data-activity]');
      activity.textContent = 'Last Activity: ' + date.toLocaleString([], {dateStyle:'medium', timeStyle:'short'}); activity.title = date.toLocaleString();
      const button = this.dialog.querySelector('[data-protect]');
      button.innerHTML = svg(state.archive_protected ? 'lock' : 'unlock');
      button.setAttribute('aria-pressed', String(state.archive_protected));
      button.title = state.archive_protected ? 'Protected from automatic archiving — click to unlock' : 'Allow automatic archiving — click to protect';
    }
    resetWorkspace() {
      clearTimeout(this.refreshTimer); this.modal.close();
      this.workspace = this.picker.state.selected; this.selections = {video: new Set(), audio: new Set()}; this.state = null; this.detailKey = null;
      this.pages = {video: 0, audio: 0};
      this.grid.replaceChildren(); this.selectionChanged();
      this.renderWorkspaces(); this.dialog.querySelector('[data-activity]').textContent = '';
      this.dialog.querySelector('[data-summary]').textContent = '';
      this.dialog.querySelector('[data-prev]').disabled = this.dialog.querySelector('[data-next]').disabled = true;
      this.notice('Loading workspace…'); this.load(0, false, true);
    }
    async changeWorkspace(id) {
      this.renderWorkspaces(); this.setBusy(true);
      try {this.picker.render(await this.transport.request('workspaces/select', {id, context:'gallery'})); this.invalidate();}
      catch (error) {this.notice(error.message);}
      finally {this.setBusy(false);}
    }
    async protect() {
      this.setBusy(true);
      try {await this.transport.request('workspace_viewer/protection', {workspace:this.workspace, protected:!this.state.archive_protected}); await this.load(this.page, true);}
      catch (error) {this.notice(error.message);}
      finally {this.setBusy(false);}
    }
    invalidate() {
      if (!this.dialog.open) return;
      this.renderWorkspaces();
      if (this.workspace !== this.picker.state.selected) {this.resetWorkspace(); return;}
      if (!this.state) return;
      clearTimeout(this.refreshTimer); this.refreshTimer = setTimeout(() => this.load(this.page || 0, true), 150);
    }
    async load(page, keepScroll = false, initial = false) {
      clearTimeout(this.refreshTimer);
      const request = ++this.requestId;
      this.page = this.pages[this.source] = page; this.loading = true; this.navigation();
      const requestedSelection = new Set(this.selected);
      try {
        const state = await this.transport.request('workspace_viewer', {workspace: this.workspace, source: this.source, page: Math.max(0, page), selected: [...this.selected], initial});
        if (request !== this.requestId || !this.dialog.open) return;
        state.items.forEach(item => {if (item.thumbnail) item.thumbnail = new URL(item.thumbnail, this.transport.base).href;});
        this.pages[state.source] = state.page;
        this.renderActivity(state);
        if (keepScroll && this.state?.revision === state.revision && this.state.source === state.source && this.state.page === state.page) {this.state = state; this.selectionChanged(); return;}
        const retained = new Set(state.selected);
        if (initial) {this.source = state.source; this.selections[this.source] = retained; this.anchor = state.selected[0];}
        else requestedSelection.forEach(key => {if (!retained.has(key)) this.selected.delete(key);});
        this.state = state; this.page = state.page;
        const range = state.items.length ? `${state.items[0].index + 1}–${state.items[state.items.length - 1].index + 1} of ${state.total} media` : '0 media';
        this.dialog.querySelector('[data-summary]').textContent = `${range} · ${state.visible} visible in the main gallery`;
        this.dialog.querySelectorAll('[data-source]').forEach(tab => tab.setAttribute('aria-selected', String(tab.dataset.source === state.source)));
        this.grid.replaceChildren(...state.items.map(item => this.tile(item)));
        this.dialog.querySelector('.wv-empty').hidden = state.total > 0;
        this.dialog.querySelector('[data-page]').value = state.page + 1; this.dialog.querySelector('[data-page]').max = state.pages;
        this.dialog.querySelector('[data-pages]').textContent = '/ ' + state.pages;
        this.navigation();
        if (!keepScroll) this.browser.scrollTop = 0;
        this.notice(''); this.selectionChanged();
        if (initial && this.anchor) {
          this.grid.querySelector('[aria-selected="true"]').scrollIntoView({block: 'nearest'});
          this.showDetails(this.anchor);
        }
      } catch (error) {if (request === this.requestId) this.notice(error.message);}
      finally {if (request === this.requestId) {this.loading = false; this.navigation();}}
    }
    tile(item) {
      const tile = document.createElement('article'); tile.className = 'wv-tile'; tile.dataset.key = item.key; tile.dataset.visible = String(item.visible); tile.draggable = true; tile.tabIndex = 0; tile.setAttribute('role', 'option'); tile.setAttribute('aria-label', item.name);
      const thumb = document.createElement('div'); thumb.className = 'wv-thumb';
      if (!item.url) thumb.textContent = 'File missing';
      else if (item.kind === 'audio') {thumb.innerHTML = '<svg viewBox="0 0 120 70" aria-hidden="true"><path d="M12 30v10m12-24v38m12-44v50m12-37v24m12-42v60m12-50v40m12-30v20m12-38v56m12-39v22"/></svg>';}
      else {const image = WanGPMediaView.thumbnail(item); image.draggable = false; image.onerror = () => {image.remove(); thumb.prepend(document.createTextNode('Preview unavailable'));}; thumb.append(image);}
      if (item.url && item.kind === 'video') thumb.append(WanGPMediaView.playButton(() => {
        this.choose(item.key, {});
        this.play(item);
      }));
      const name = document.createElement('div'); name.className = 'wv-name'; name.textContent = item.name; name.title = item.name;
      const badge = document.createElement('span'); badge.className = 'wv-badge'; badge.textContent = item.visible ? 'In gallery' : 'Stored';
      const index = document.createElement('span'); index.className = 'wv-index'; index.textContent = `${item.index + 1} · ${item.kind}`;
      tile.append(thumb, name, index, badge); return tile;
    }
    choose(key, event) {
      const keys = this.state.items.map(item => item.key), multi = event.ctrlKey || event.metaKey;
      if (event.shiftKey && keys.includes(this.anchor)) {
        if (!multi) this.selected.clear();
        const a = keys.indexOf(this.anchor), b = keys.indexOf(key); keys.slice(Math.min(a,b), Math.max(a,b)+1).forEach(id => this.selected.add(id));
      } else if (multi) {if (this.selected.has(key)) this.selected.delete(key); else this.selected.add(key);}
      else {this.selected.clear(); this.selected.add(key);}
      this.anchor = key; this.selectionChanged(); this.showDetails(key);
    }
    selectionChanged() {
      this.navigation();
      this.grid.querySelectorAll('[data-key]').forEach(tile => tile.setAttribute('aria-selected', String(this.selected.has(tile.dataset.key))));
      this.dialog.querySelector('[data-count]').textContent = `${this.selected.size} selected`;
      this.dialog.querySelector('[data-actions]').hidden = !this.selected.size;
      this.dialog.querySelector('[data-clear]').hidden = !this.selected.size;
      if (!this.selected.size) {this.detailRequest?.abort(); this.detailKey = null; this.preview.replaceChildren(); this.info.srcdoc = WanGPMediaView.documentHtml('Select media to view its properties.');}
    }
    async showDetails(key) {
      const item = this.state.items.find(item => item.key === key); if (!item) return;
      const identity = this.workspace + this.source + key; this.detailKey = identity;
      this.detailRequest?.abort();
      const request = this.detailRequest = new AbortController();
      this.preview.replaceChildren();
      if (item.url) {
        const open = document.createElement('button'); open.type = 'button'; open.className = 'wv-open';
        open.append(WanGPMediaView.thumbnail(item));
        const label = document.createElement('span'); label.textContent = item.kind === 'image' ? 'Open image' : '▶ Play'; open.append(label);
        open.onclick = () => this.play(item);
        this.preview.append(open);
      }
      this.info.srcdoc = WanGPMediaView.documentHtml('Loading media information…');
      try {const result = await this.transport.request(this.query('workspace_viewer/info', {key}), undefined, request.signal); if (!request.signal.aborted && this.detailKey === identity) this.info.srcdoc = WanGPMediaView.documentHtml(result.html);}
      catch (error) {if (!request.signal.aborted && this.detailKey === identity) this.notice(error.message);}
    }
    play(item) {
      const media = WanGPMediaView.media(item); this.preview.replaceChildren(media);
      if (item.kind !== 'image') media.play().catch(() => {});
    }
    async run(action, extra = {}) {
      if (this.busy) return false;
      this.setBusy(true);
      try {
        const result = await this.transport.request('workspace_viewer/' + action, {workspace: this.workspace, source: this.source, revision: this.state.revision, keys: [...this.selected], ...extra});
        if (action === 'archive') {const link = document.createElement('a'); link.href = result.url; link.download = result.filename; document.body.append(link); link.click(); link.remove();}
        if (['eject', 'delete', 'move'].includes(action)) this.selected.clear();
        await this.load(this.page, true);
        this.notice(result.errors?.length ? result.errors.join(' · ') : action === 'copy' ? `${result.count} media added to the target workspace.` : action === 'move' ? `${result.count} media moved to the target workspace.` : '');
        return true;
      } catch (error) {this.notice(error.message); if (this.modal.open) {const node = this.modal.querySelector('[role=alert]'); node.hidden = false; node.textContent = error.message;} await this.load(this.state.page, true); this.notice(error.message); return false;}
      finally {this.setBusy(false);}
    }
    async action(action) {
      if (this.busy || (action !== 'retention' && !this.selected.size)) return;
      let retention;
      if (action === 'retention') {
        this.setBusy(true);
        try {retention = await this.transport.request('workspace_viewer/retention');}
        finally {this.setBusy(false);}
        if (!this.dialog.open) return;
      }
      if (action === 'eject') return this.run(action);
      if (action === 'last') return this.run('reorder', {before: null});
      if (action === 'first') return this.run('reorder', {edge: 'start'});
      this.modal.dataset.action = action;
      this.modal.querySelector('h2').textContent = {delete:'Delete files permanently',copy:'Copy to workspace',move:'Move to workspace',archive:'Download ZIP archive',retention:'Automatic workspace archiving'}[action];
      this.modal.querySelector('p').textContent = action === 'retention' ? 'At the next WanGP startup, archive unlocked workspaces with no media additions, removals or reordering during this period. Their dedicated Deepy sessions are archived too. Media files stay in place. Opening or renaming a workspace does not reset its activity date.' : action === 'delete' ? `Permanently delete ${this.selected.size} selected media files from disk? They will also be removed from every workspace that references them. This cannot be undone.` : action === 'copy' ? 'Append the selected media to another workspace. The original files are shared, without copying them on disk.' : 'Choose the name of the ZIP to download.';
      if (action === 'move') this.modal.querySelector('p').textContent = 'Move the selected media to another workspace and remove them from this one. The files stay in place on disk; other workspaces keep their references.';
      this.modal.querySelector('[data-name-label]').hidden = action !== 'archive'; this.modal.querySelector('input').disabled = action !== 'archive';
      this.modal.querySelector('input').value = action === 'archive' ? this.state.name + '.zip' : '';
      const target = this.modal.querySelector('select'); target.disabled = !['copy', 'move', 'retention'].includes(action); target.closest('label').hidden = target.disabled;
      this.modal.querySelector('[data-select-label]').textContent = action === 'retention' ? 'Archive after inactivity of' : 'Target workspace';
      target.replaceChildren(...(retention ? retention.choices.map(([label, days]) => new Option(label, days)) : this.picker.state.items.filter(item => item.id !== this.workspace).map(item => new Option(WanGPWorkspacePicker.label(item), item.id))));
      if (retention) target.value = retention.days;
      this.modal.querySelector('[role=alert]').hidden = true;
      this.modal.querySelector('[type=submit]').textContent = action === 'retention' ? 'Save' : action === 'delete' ? 'Delete permanently' : action === 'copy' ? 'Copy' : action === 'move' ? 'Move' : 'Download';
      this.modal.querySelector('[type=submit]').disabled = ['copy', 'move'].includes(action) && !target.options.length;
      this.modal.showModal(); this.modal.querySelector(action === 'delete' ? '[data-cancel]' : action === 'archive' ? 'input' : 'select').focus();
    }
    async confirm() {
      const button = this.modal.querySelector('[type=submit]'); button.disabled = true;
      if (this.modal.dataset.action === 'retention') {
        this.setBusy(true);
        try {
          await this.transport.request('workspace_viewer/retention', {days:Number(this.modal.querySelector('select').value)});
          this.modal.close(); this.notice('Automatic archiving preference saved. It applies at the next WanGP startup.');
        } catch (error) {const node = this.modal.querySelector('[role=alert]'); node.hidden = false; node.textContent = error.message;}
        finally {button.disabled = false; this.setBusy(false);}
        return;
      }
      const ok = await this.run(this.modal.dataset.action, {confirmed: this.modal.dataset.action === 'delete', name: this.modal.querySelector('input').value, target: this.modal.querySelector('select').value});
      button.disabled = false; if (ok) this.modal.close();
    }
    async importFiles(files) {
      if (this.busy) return;
      this.setBusy(true);
      const workspace = this.workspace;
      let done = 0; const failures = [], reused = [];
      for (const file of files) {
        this.notice(`Importing ${++done} / ${files.length}: ${file.name}`);
        const data = new FormData(); data.append('file', file);
        try {
          const result = await this.transport.request('workspace_viewer/import?' + new URLSearchParams({workspace}), data);
          if (result.reused) reused.push(`Existing file reused: ${result.filename}`);
        }
        catch (error) {failures.push(`${file.name}: ${error.message}`);}
      }
      this.setBusy(false); this.dialog.querySelector('[data-files]').value = '';
      await this.load(this.state.page); this.notice([...reused, ...failures].join(' · '));
    }
    installDrag() {
      this.grid.ondragstart = event => {
        if (event.target.closest('button')) {event.preventDefault(); return;}
        const tile = event.target.closest('[data-key]'); if (!tile) return;
        if (!this.selected.has(tile.dataset.key)) {this.selected.clear(); this.selected.add(tile.dataset.key); this.selectionChanged();}
        this.dragging = true; event.dataTransfer.effectAllowed = 'move'; event.dataTransfer.setData('text/plain', `${this.selected.size} media`);
      };
      this.dialog.ondragend = () => {this.dragging = false; clearTimeout(this.pageHover); this.grid.querySelectorAll('.wv-drop').forEach(tile => tile.classList.remove('wv-drop'));};
      this.grid.ondragover = event => {
        if (!this.dragging) return; event.preventDefault(); event.dataTransfer.dropEffect = 'move';
        this.grid.querySelectorAll('.wv-drop').forEach(tile => tile.classList.remove('wv-drop'));
        event.target.closest('[data-key]')?.classList.add('wv-drop');
      };
      this.grid.ondrop = async event => {if (!this.dragging) return; event.preventDefault(); const tile = event.target.closest('[data-key]'); this.dragging = false; clearTimeout(this.pageHover); await this.run('reorder', {before: tile?.dataset.key || null});};
      for (const [selector, delta] of [['[data-prev]', -1], ['[data-next]', 1]]) {
        const button = this.dialog.querySelector(selector);
        button.ondragover = event => {if (!this.dragging || button.disabled) return; event.preventDefault(); if (!this.pageHover) this.pageHover = setTimeout(async () => {await this.load(this.state.page + delta); this.pageHover = null;}, 650);};
        button.ondragleave = () => {clearTimeout(this.pageHover); this.pageHover = null;};
      }
    }
    installRubber() {
      this.browser.onpointerdown = event => {
        if (event.button !== 0 || event.pointerType !== 'mouse' || event.target.closest('[data-key]')) return;
        const box = this.browser.getBoundingClientRect(), origin = {x: event.clientX - box.left + this.browser.scrollLeft, y: event.clientY - box.top + this.browser.scrollTop};
        const base = event.ctrlKey || event.metaKey ? new Set(this.selected) : new Set();
        let client = {x:event.clientX, y:event.clientY}, moved = false;
        const rubber = this.dialog.querySelector('.wv-rubber');
        this.browser.setPointerCapture(event.pointerId);
        const update = () => {
          const bounds = this.browser.getBoundingClientRect();
          if (client.y > bounds.bottom - 35) this.browser.scrollTop += 12;
          if (client.y < bounds.top + 35) this.browser.scrollTop -= 12;
          const x = client.x - bounds.left + this.browser.scrollLeft, y = client.y - bounds.top + this.browser.scrollTop;
          const rect = {left:Math.min(x,origin.x), top:Math.min(y,origin.y), right:Math.max(x,origin.x), bottom:Math.max(y,origin.y)};
          moved ||= Math.abs(x-origin.x)+Math.abs(y-origin.y)>5;
          if (moved) {
            rubber.hidden = false; Object.assign(rubber.style, {left:rect.left+'px', top:rect.top+'px', width:rect.right-rect.left+'px', height:rect.bottom-rect.top+'px'});
            this.selected.clear(); base.forEach(key => this.selected.add(key));
            this.grid.querySelectorAll('[data-key]').forEach(tile => {const r=tile.getBoundingClientRect(), left=r.left-bounds.left+this.browser.scrollLeft, top=r.top-bounds.top+this.browser.scrollTop; if(left<rect.right && left+r.width>rect.left && top<rect.bottom && top+r.height>rect.top) this.selected.add(tile.dataset.key);});
            this.selectionChanged();
          }
          this.rubberFrame = requestAnimationFrame(update);
        };
        this.browser.onpointermove = move => {client={x:move.clientX,y:move.clientY};};
        const end = () => {cancelAnimationFrame(this.rubberFrame); rubber.hidden=true; this.suppressClick=moved; setTimeout(() => {this.suppressClick=false;}, 0); this.browser.onpointermove=null; this.browser.onpointerup=null; this.browser.onpointercancel=null;};
        this.browser.onpointerup=end; this.browser.onpointercancel=end; update();
      };
    }
  }
  window.WanGPWorkspaceViewer = WorkspaceViewer;
})();
