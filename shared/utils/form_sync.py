"""Versioned application forms; no dependency on Gradio or the LLM runtime."""
from copy import deepcopy
from dataclasses import dataclass
from threading import RLock
from uuid import uuid4


class FormConflict(ValueError):
    pass


@dataclass
class Saved:
    output: object


def merge_values(base, local, remote):
    merged, conflicts = {}, []
    for key, current in remote.items():
        original, edited = base[key], local[key]
        if edited == original or edited == current:
            merged[key] = current
        else:
            merged[key] = edited
            if current != original:
                conflicts.append(key)
    return merged, conflicts


class SharedForm:
    def __init__(self, name, values, *, publish, read=None, scope=lambda: 'application', labels=None, lock=None):
        self.name, self._values = name, deepcopy(values)
        self._read, self._scope, self._publish = read, scope, publish
        self.labels = labels or {key: key for key in values}
        self.lock = lock if lock is not None else RLock()
        self.revision = 0
        self.instance = uuid4().hex
        self._scope_id = scope()

    def _refresh(self):
        values = self._read() if self._read else self._values
        scope = self._scope()
        if values != self._values or scope != self._scope_id:
            self._values, self._scope_id = deepcopy(values), scope
            self.revision += 1
            self._publish('forms', {self.name: self.revision})

    def snapshot(self):
        with self.lock:
            self._refresh()
            return {'instance': self.instance, 'scope': self._scope_id, 'revision': self.revision, 'values': deepcopy(self._values)}

    def submit(self, baseline, local, save):
        with self.lock:
            current = self.snapshot()
            if not isinstance(baseline, dict) or not isinstance(baseline.get('values'), dict) or not isinstance(local, dict):
                raise ValueError('Invalid form submission.')
            if baseline.get('scope') != current['scope']:
                raise FormConflict('The conversation changed. Reload these settings before saving.')
            if set(local) != set(current['values']) or set(baseline['values']) != set(local):
                raise ValueError('The form fields changed. Reload this page before saving.')
            merged, conflicts = merge_values(baseline['values'], local, current['values'])
            if conflicts:
                labels = ', '.join(self.labels[key] for key in conflicts)
                raise FormConflict(f'Changed in another window: {labels}. Choose whether to load the saved values or keep your edits.')
            result = save(merged)
            if isinstance(result, Saved):
                if self._read is None and merged != self._values:
                    self._values = deepcopy(merged)
                    self.revision += 1
                    self._publish('forms', {self.name: self.revision})
                self._refresh()
            return result


class FormRegistry:
    def __init__(self, publish):
        self.publish, self.forms = publish, {}

    def register(self, name, values, **kwargs):
        if name in self.forms:
            raise ValueError(f'Form already registered: {name}')
        form = SharedForm(name, values, publish=self.publish, **kwargs)
        self.forms[name] = form
        return form

    def revisions(self):
        return {name: form.revision for name, form in self.forms.items()}

    def refresh(self, name):
        if name in self.forms:
            self.forms[name].snapshot()
