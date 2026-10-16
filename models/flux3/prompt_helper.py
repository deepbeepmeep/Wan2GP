# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Flux3 caption and bounding-box editor, hosted by the shared prompt helper."""
import html
import json
from pathlib import Path

from shared.gradio import prompt_layout


def render_prompt_helper(model_type, model_def, prompt_id, popup_id, prompt_elem_id, resolution_elem_id):
    config = html.escape(json.dumps({'promptTarget': prompt_elem_id, 'resolutionTarget': resolution_elem_id}), quote=True)
    popup_id = html.escape(popup_id, quote=True)
    return f'''
<div id="{popup_id}" class="wangp-model-info-popup wangp-prompt-helper-popup flux3-prompt-helper-popup" role="dialog" aria-label="Flux 3 Layout Editor" data-wangp-model-info-popup data-wangp-prompt-helper-popup="1" hidden style="width:min(92vw,calc(100vw - 24px));height:min(94vh,calc(100vh - 12px));">
 <div class="wangp-model-info-card" data-flux3-helper data-flux3-config="{config}">
  <div class="wangp-model-info-titlebar" data-wangp-model-info-drag>
   <div class="wangp-model-info-heading">Flux 3 Layout Editor</div>
   <button type="button" class="wangp-model-info-close" aria-label="Close prompt helper" data-wangp-model-info-close>&times;</button>
  </div>
  <div class="wangp-model-info-content flux3-helper-content">
   <div class="flux3-helper-toolbar">
    <button type="button" data-action="add">Add Element</button>
    <button type="button" data-action="delete">Delete Element</button>
    <button type="button" data-action="undo">Undo</button>
    <button type="button" data-action="redo">Redo</button>
    <button type="button" data-action="references">Refresh References</button>
    <button type="button" data-action="raw">Edit Prompt Text</button>
    <button type="button" data-action="save" data-flux3-action="save">Save Prompt</button>
   </div>
   <div class="flux3-helper-status" role="status" data-status></div>
   <div class="flux3-helper-raw" hidden data-raw>
    <label>Original Caption and Layout — correct any errors, then load the layout.
     <textarea data-raw-input spellcheck="false" aria-label="Caption and Layout Text"></textarea>
    </label>
    <button type="button" data-action="retry">Load Layout</button>
   </div>
   <div class="flux3-helper-grid">
    <div class="flux3-helper-visual">
     <div class="flux3-view-controls">
      <label>Canvas<select data-field="view"><option value="target">Output Layout</option><option value="source">Selected Element's Source</option></select></label>
      <label>Output Preview Shape<select data-field="aspect"><option value="resolution">Selected Resolution</option><option value="reference">First Reference (KI)</option></select></label>
     </div>
     <div class="flux3-helper-resolution" data-resolution></div>
     <div class="flux3-helper-canvas-wrap"><canvas width="1000" height="1000" aria-label="Flux 3 layout canvas" tabindex="0"></canvas></div>
     <div class="flux3-helper-hint">Drag empty output space or Alt-drag to add an element. Drag boxes to move; drag edges or corners to resize. In Source view, drag to set the selected element's source region. Boxes guide placement, not pixel masks.</div>
     <div class="flux3-helper-list" aria-label="Layout elements"></div>
    </div>
    <div class="flux3-helper-fields">
     <label>Scene Caption<textarea data-field="caption" placeholder="Describe the complete finished image, using <id> to identify each element."></textarea></label>
     <h3>Selected Element</h3>
     <label>Element ID<input data-field="id" placeholder="subject_1" autocomplete="off"></label>
     <label>Description<textarea data-field="desc" placeholder="Describe this element, quoting exact visible text."></textarea></label>
     <label>Content Source<select data-field="from"></select></label>
     <label>Output Box — Top, Left, Bottom, Right<input data-field="target" placeholder="[100, 100, 900, 900]" spellcheck="false"></label>
     <label class="flux3-check"><input type="checkbox" data-field="removed">Remove This Element from the Output</label>
     <label>Source Box — Top, Left, Bottom, Right<input data-field="source" placeholder="[0, 0, 1000, 1000]" spellcheck="false"></label>
     <div class="flux3-helper-hint">Coordinates use a 0–1000 grid. Reference IDs follow the main Reference Images gallery order. Upload and reorder references there, then refresh here. Preview shape does not change generation settings.</div>
     <div class="flux3-reference-status" data-reference-status></div>
    </div>
   </div>
  </div>
 </div>
</div>'''


def get_prompt_helper_css():
    return prompt_layout.get_css('flux3') + '''
.flux3-helper-grid { grid-template-columns: minmax(300px, 1.3fr) minmax(280px, 1fr); }
.flux3-view-controls { display:flex; flex-wrap:wrap; gap:8px; }
.flux3-view-controls label { display:flex; flex-direction:column; gap:4px; font-size:.85rem; flex:1; }
.flux3-helper-fields select, .flux3-view-controls select { padding:6px; border:1px solid var(--border-color-primary); border-radius:5px; background:var(--input-background-fill); color:var(--body-text-color); max-width:100%; }
.flux3-helper-fields textarea[data-field="caption"] { min-height:110px; }
.flux3-helper-fields textarea[data-field="desc"] { min-height:80px; }
.flux3-helper-fields .flux3-check { flex-direction:row; align-items:center; }
.flux3-check input { appearance:auto; accent-color:var(--button-primary-background-fill, #176b92); width:16px; height:16px; flex-shrink:0; }
.flux3-helper-fields :disabled { opacity:.55; }
.flux3-helper-fields [aria-invalid="true"] { border-color:var(--error-text-color, #b32438); }
.flux3-helper-raw label { display:flex; flex-direction:column; gap:8px; flex:1; }
.flux3-helper-canvas-wrap { flex:1; min-height:220px; }
.flux3-helper-list { max-height:160px; flex-shrink:0; }
.flux3-reference-status { font-size:.85rem; color:var(--body-text-color-subdued); }
@media (max-width:800px) { .flux3-helper-grid { display:flex; flex-direction:column; overflow:auto; } .flux3-helper-visual, .flux3-helper-fields { overflow:visible; flex:none; } .flux3-helper-canvas-wrap { flex:none; height:38vh; } }
'''


def get_prompt_helper_javascript():
    return prompt_layout.get_javascript() + Path(__file__).with_suffix('.js').read_text(encoding='utf-8')
