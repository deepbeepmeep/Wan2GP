// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
/* One workspace picker and dialog implementation for both browser interfaces. */
(() => {
  const icons = {
    create: '<path d="M12 5v14M5 12h14"/>',
    rename: '<path d="m15 5 4 4M4 20l4-1L20 7a2.8 2.8 0 0 0-4-4L4 15v5Z"/>',
    delete: '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/>',
  };
  function closeTools(tools, restoreFocus = false) {
    tools.classList.remove('is-open');
    const toggle = tools.querySelector('.wangp-workspace-toggle');
    toggle.setAttribute('aria-expanded', 'false');
    if (restoreFocus) toggle.focus({preventScroll: true});
  }
  document.addEventListener('pointerdown', event => {
    document.querySelectorAll('.wangp-workspace-tools.is-open').forEach(tools => {
      if (!tools.contains(event.target)) closeTools(tools);
    });
  });
  document.addEventListener('keydown', event => {
    if (event.key !== 'Escape') return;
    document.querySelectorAll('.wangp-workspace-tools.is-open').forEach(tools => {
      closeTools(tools, true); event.preventDefault();
    });
  });
  function layoutTools(tools) {
    const header = tools.parentElement;
    if (!header.clientWidth || tools.hidden) return;
    // Measure the native tab labels and inline tools, even while collapsed.
    // This switches as soon as they fit, including after a window resize.
    const tabs = header.querySelector('.tab-container[aria-hidden=true]');
    const open = tools.classList.contains('is-open');
    tools.classList.remove('is-compact', 'is-open');
    const tabsWidth = tabs ? tabs.scrollWidth : header.querySelector('[role=tablist]').scrollWidth;
    const compact = header.clientWidth < tabsWidth + tools.getBoundingClientRect().width + 4;
    tools.classList.toggle('is-compact', compact);
    tools.classList.toggle('is-open', compact && open);
    tools.querySelector('.wangp-workspace-toggle').setAttribute('aria-expanded', String(compact && open));
  }
  window.addEventListener('resize', () => document.querySelectorAll('.wangp-workspace-tools').forEach(tools => {closeTools(tools); layoutTools(tools);}));
  document.addEventListener('animationstart', event => {
    if (event.animationName === 'wangp-workspace-ready') layoutTools(event.target);
  });
  class WorkspacePicker {
    constructor(host, request, notice, {deepy = false, management = true, info = null} = {}) {
      this.request = request; this.notice = notice; this.info = info; this.state = null;
      this.deepy = deepy; this.management = management; this.busy = false;
      this.node = document.createElement('div'); this.node.className = 'wangp-workspaces'; this.node.setAttribute('role', 'group'); this.node.setAttribute('aria-label', 'Workspace');
      this.node.innerHTML = '<select aria-label="Workspace" title="Workspace"></select>' + Object.entries(icons).map(([action, path]) => `<button type="button" data-workspace-action="${action}" aria-label="${action === 'create' ? 'Add' : action === 'rename' ? 'Rename' : 'Delete'} workspace" title="${action === 'create' ? 'Add' : action === 'rename' ? 'Rename' : 'Delete'} workspace"><svg viewBox="0 0 24 24" aria-hidden="true">${path}</svg></button>`).join('');
      if (host.matches('#wangp-gallery-tabs [role=tablist]')) {
        this.tools = document.createElement('div'); this.tools.className = 'wangp-workspace-tools';
        const toggle = document.createElement('button'); toggle.type = 'button'; toggle.className = 'wangp-workspace-toggle';
        toggle.title = toggle.ariaLabel = 'Workspace & Actions';
        toggle.setAttribute('aria-expanded', 'false'); toggle.setAttribute('aria-controls', 'wangp-gallery-workspace-controls');
        toggle.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 7h7l2-3h9v16H3V7Z"/><path d="M8 12h8M8 16h8"/></svg>';
        this.node.id = 'wangp-gallery-workspace-controls';
        this.tools.append(toggle, this.node);
        host.parentElement.append(this.tools);
        layoutTools(this.tools);
        toggle.onclick = () => {
          const open = this.tools.classList.toggle('is-open');
          toggle.setAttribute('aria-expanded', String(open));
        };
      } else host.append(this.node);
      this.select = this.node.querySelector('select');
      this.select.onchange = async () => {
        const id = this.select.value;
        this.select.value = this.deepy ? this.state.deepy_selected || '' : this.state.selected;
        await this.commit('select', {id});
      };
      this.node.addEventListener('click', event => {
        event.stopPropagation();
        const action = event.target.closest('[data-workspace-action]')?.dataset.workspaceAction;
        if (action) this.open(action);
      });
      this.dialog = document.createElement('dialog'); this.dialog.className = 'wangp-workspace-dialog';
      this.dialog.innerHTML = '<form><h2></h2><div class="wangp-workspace-body"><p></p><label>Workspace name<input maxlength="120" autocomplete="off" required></label><div class="wangp-workspace-error" role="alert" hidden></div><div class="wangp-workspace-actions"><button type="button" data-cancel>Cancel</button><button type="submit">Save</button></div></div></form>';
      document.body.append(this.dialog);
      this.dialog.querySelector('[data-cancel]').onclick = () => this.dialog.close();
      this.dialog.querySelector('form').onsubmit = async event => {
        event.preventDefault();
        const payload = {id: this.dialog.dataset.workspaceId, name: this.dialog.querySelector('input').value};
        if (await this.commit(this.dialog.dataset.action, payload)) this.dialog.close();
      };
    }
    static label(item) { return `${item.deepy_session_id || item.mcp_agent_id ? '🤖' : '📁'} ${item.name}${item.storage_mode === 'dedicated' ? ' · Folder' : ''}`; }
    availability() {
      const locked = this.deepy && !!this.state?.deepy_locked;
      this.select.disabled = this.busy || locked;
      this.node.querySelectorAll('[data-workspace-action]').forEach(node => {node.hidden = locked || !this.management; node.disabled = this.busy;});
      if (locked && this.dialog.open) this.dialog.close();
    }
    render(state, reset = false) {
      if (!reset && state && this.state && state.revision < this.state.revision) return;
      this.node.hidden = !state;
      if (this.tools) this.tools.hidden = !state;
      if (this.tools) layoutTools(this.tools);
      if (!state) return;
      this.state = state;
      const selected = this.deepy ? state.deepy_selected : state.selected;
      const choices = JSON.stringify([state.items, selected === null]);
      if (choices !== this.choices) {
        this.select.replaceChildren(...state.items.map(item => new Option(WorkspacePicker.label(item), item.id)));
        if (selected === null) this.select.prepend(new Option('🤖 New Deepy session', ''));
        this.choices = choices;
      }
      this.select.value = selected || '';
      this.select.title = this.select.selectedOptions[0].textContent;
      this.availability();
    }
    open(action) {
      if (this.tools) closeTools(this.tools);
      const current = this.state.items.find(item => item.id === this.state.selected);
      if (current.deepy_session_id && action !== 'create') {
        (this.info || this.notice)(`${action === 'rename' ? 'Rename' : 'Delete'} this workspace through its Deepy session.`);
        return;
      }
      const removing = action === 'delete';
      this.dialog.dataset.action = action; this.dialog.dataset.workspaceId = current.id;
      this.dialog.querySelector('h2').textContent = action === 'create' ? 'Add workspace' : removing ? 'Delete workspace' : 'Rename workspace';
      this.dialog.querySelector('p').textContent = removing ? `Delete “${current.name}”? The media files will stay on disk.` : action === 'create' ? 'Create an empty gallery workspace.' : 'Choose a new name for this workspace.';
      const input = this.dialog.querySelector('input');
      input.disabled = removing; input.closest('label').hidden = removing;
      input.value = action === 'create' ? '' : current.name;
      this.dialog.querySelector('button[type=submit]').textContent = removing ? 'Delete' : action === 'create' ? 'Create' : 'Save';
      this.dialog.querySelector('[role=alert]').hidden = true;
      this.dialog.showModal();
      if (!removing) {input.focus(); input.select();}
      else this.dialog.querySelector('[data-cancel]').focus();
    }
    async commit(action, payload) {
      this.busy = true; this.availability();
      this.dialog.querySelector('button[type=submit]').disabled = true;
      try {
        this.render(await this.request('workspaces/' + action, {...payload, context: this.deepy ? 'deepy' : 'gallery'}));
        return true;
      } catch (error) {
        if (this.info && (action === 'rename' || action === 'delete')) this.info(error.message);
        else if (this.dialog.open) {
          const target = this.dialog.querySelector('[role=alert]'); target.textContent = error.message; target.hidden = false;
        } else this.notice(error.message);
        return false;
      } finally {
        this.busy = false; this.availability();
        this.dialog.querySelector('button[type=submit]').disabled = false;
      }
    }
  }
  window.WanGPWorkspacePicker = WorkspacePicker;
})();
