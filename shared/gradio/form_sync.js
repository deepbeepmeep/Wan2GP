(() => {
  function equal(a, b) {
    if (a === b) return true;
    if (a === null || b === null || typeof a !== 'object' || typeof b !== 'object' || Array.isArray(a) !== Array.isArray(b)) return false;
    const keys = Object.keys(a);
    return keys.length === Object.keys(b).length && keys.every(key => Object.prototype.hasOwnProperty.call(b, key) && equal(a[key], b[key]));
  }
  function merge(base, local, remote) {
    const values = {}, conflicts = [], dirty = [];
    for (const key of Object.keys(remote)) {
      if (equal(local[key], base[key]) || equal(local[key], remote[key])) values[key] = remote[key];
      else {
        values[key] = local[key]; dirty.push(key);
        if (!equal(remote[key], base[key])) conflicts.push(key);
      }
    }
    return {values, conflicts, dirty};
  }
  const forms = new Map();
  function register(name, selector) {
    if (!forms.has(name)) forms.set(name, {selector, revision: -1});
  }
  function refresh(revisions) {
    for (const [name, revision] of Object.entries(revisions || {})) {
      const form = forms.get(name);
      if (!form || revision === form.revision) continue;
      const button = document.querySelector(form.selector);
      if (button) {form.revision = revision; button.click();}
    }
  }
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function apply(keys, labels, raw, previous, inputs, reload = false) {
    const remote = JSON.parse(raw), base = JSON.parse(previous);
    if (remote.instance === base.instance && remote.revision < base.revision) return [previous, ...Array.from({length:keys.length+(remote.view || []).length+2}, () => ({__type__:'update'}))];
    if (remote.ack) base.values = {...base.values, ...remote.ack};
    const local = Object.fromEntries(keys.map((key, i) => [key, inputs[i]]));
    const switched = reload || base.scope !== remote.scope;
    const result = switched ? {values: remote.values, conflicts: [], dirty: []} : merge(base.values, local, remote.values);
    const baseline = {...remote, values: {...remote.values}};
    for (const key of result.conflicts) baseline.values[key] = base.values[key];
    const note = result.conflicts.length ? 'Changed in another window: ' + result.conflicts.map(key => labels[key]).join(', ') + '. Your edits have been kept.' : '';
    const updates = keys.map(key => {
      const field = remote.fields?.[key];
      const update = field && !field.dependencies.some(k => result.dirty.includes(k)) && !result.dirty.includes(key) ? {...field.update} : {__type__:'update'};
      if (!equal(local[key], result.values[key])) update.value = result.values[key];
      return update;
    });
    const view = (remote.view || []).map(item => item.dependencies.some(key => result.dirty.includes(key)) ? {__type__:'update'} : item.update);
    return [JSON.stringify(baseline), ...updates, ...view, note ? '<div role="status">'+escape(note)+'</div>' : '', {__type__:'update',visible:!!note}];
  }
  window.WanGPFormSync = {merge, register, refresh, apply};
  if (typeof module !== 'undefined') module.exports = {merge};
})();
