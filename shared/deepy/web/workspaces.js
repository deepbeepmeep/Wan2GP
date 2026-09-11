/* One workspace picker and dialog implementation for both browser interfaces. */
(() => {
  const icons = {
    create: '<path d="M12 5v14M5 12h14"/>',
    rename: '<path d="m15 5 4 4M4 20l4-1L20 7a2.8 2.8 0 0 0-4-4L4 15v5Z"/>',
    delete: '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/>',
  };
  class WorkspacePicker {
    constructor(host, request, notice, {deepy = false, management = true} = {}) {
      this.request = request; this.notice = notice; this.state = null;
      this.deepy = deepy; this.management = management; this.busy = false;
      this.node = document.createElement('div'); this.node.className = 'wangp-workspaces'; this.node.setAttribute('role', 'group'); this.node.setAttribute('aria-label', 'Workspace');
      this.node.innerHTML = '<select aria-label="Workspace" title="Workspace"></select>' + Object.entries(icons).map(([action, path]) => `<button type="button" data-workspace-action="${action}" aria-label="${action === 'create' ? 'Add' : action === 'rename' ? 'Rename' : 'Delete'} workspace" title="${action === 'create' ? 'Add' : action === 'rename' ? 'Rename' : 'Delete'} workspace"><svg viewBox="0 0 24 24" aria-hidden="true">${path}</svg></button>`).join('');
      host.append(this.node);
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
    static label(item) { return `${item.deepy_session_id ? '🤖' : '📁'} ${item.name}`; }
    availability() {
      const locked = this.deepy && !!this.state?.deepy_locked;
      this.select.disabled = this.busy || locked;
      this.node.querySelectorAll('[data-workspace-action]').forEach(node => {node.hidden = locked || !this.management; node.disabled = this.busy;});
      if (locked && this.dialog.open) this.dialog.close();
    }
    render(state, reset = false) {
      if (!reset && state && this.state && state.revision < this.state.revision) return;
      this.node.hidden = !state;
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
      const current = this.state.items.find(item => item.id === this.state.selected);
      if (current.deepy_session_id && action !== 'create') {
        this.notice(`${action === 'rename' ? 'Rename' : 'Delete'} this workspace through its Deepy session.`);
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
        if (this.dialog.open) {
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
