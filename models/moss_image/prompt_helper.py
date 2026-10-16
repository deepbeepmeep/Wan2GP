# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""MOSS centered-pixel layout editor hosted by WanGP's shared prompt helper."""
import html
import json
from pathlib import Path

from shared.gradio import prompt_layout


def render_prompt_helper(model_type, model_def, prompt_id, popup_id, prompt_elem_id, resolution_elem_id):
    config = html.escape(json.dumps({'promptTarget': prompt_elem_id, 'resolutionTarget': resolution_elem_id}), quote=True)
    popup_id = html.escape(popup_id, quote=True)
    return f'''
<div id="{popup_id}" class="wangp-model-info-popup wangp-prompt-helper-popup moss-prompt-helper-popup" role="dialog" aria-label="MOSS Image Layout Editor" data-wangp-model-info-popup data-wangp-prompt-helper-popup="1" hidden style="width:min(92vw,calc(100vw - 24px));height:min(94vh,calc(100vh - 12px));">
 <div class="wangp-model-info-card" data-moss-helper data-moss-config="{config}">
  <div class="wangp-model-info-titlebar" data-wangp-model-info-drag>
   <div class="wangp-model-info-heading">MOSS Image Layout Editor</div>
   <button type="button" class="wangp-model-info-close" aria-label="Close prompt helper" data-wangp-model-info-close>&times;</button>
  </div>
  <div class="wangp-model-info-content moss-helper-content">
   <div class="moss-helper-toolbar">
    <button type="button" data-action="add">Add Item</button><button type="button" data-action="delete">Delete Item</button>
    <button type="button" data-action="back">Move Back</button><button type="button" data-action="front">Move Forward</button>
    <button type="button" data-action="undo">Undo</button><button type="button" data-action="redo">Redo</button>
    <button type="button" data-action="fit">Fit Boxes to Canvas</button><button type="button" data-action="raw">Edit Prompt Text</button>
    <button type="button" data-action="save" data-moss-action="save">Save Prompt</button>
   </div>
   <div class="moss-helper-status" role="status" data-status></div>
   <div class="moss-helper-raw" hidden data-raw>
    <label>Original Prompt — correct the reported syntax errors, then load the layout.
     <textarea data-raw-input aria-label="MOSS Layout Text" spellcheck="false"></textarea>
    </label><button type="button" data-action="retry">Load Layout</button>
   </div>
   <div class="moss-helper-grid">
    <div class="moss-helper-visual">
     <div class="moss-helper-resolution" data-resolution></div>
     <div class="moss-helper-canvas-wrap"><canvas tabindex="0" aria-label="MOSS layout canvas"></canvas></div>
     <div class="moss-helper-hint">Drag empty space or Alt-drag to add an item. Drag boxes to move; drag edges or corners to resize. Boxes guide composition, not pixel masks.</div>
     <div class="moss-helper-list" aria-label="Layout items"></div>
    </div>
    <div class="moss-helper-fields">
     <label>Scene Caption<textarea data-field="scene"></textarea></label>
     <h3>Selected Item</h3>
     <label>Label<input data-field="label" placeholder="text or object"></label>
     <label>Appearance and Placement<textarea data-field="caption"></textarea></label>
     <label>Exact Visible Text<textarea data-field="text" placeholder="Leave empty for an object without writing"></textarea></label>
     <label>Box — Top, Left, Bottom, Right<input data-field="bbox" spellcheck="false" placeholder="[-256, -256, 256, 256]"></label>
     <div class="moss-helper-hint" data-bounds></div>
     <div class="moss-helper-hint">Fit Boxes to Canvas moves boxes inside, preserving their size unless larger than the canvas. It may change overlaps. The editor preserves other JSON fields and comment lines. Repairs stay here until Save Prompt.</div>
    </div>
   </div>
  </div>
 </div>
</div>'''


def get_prompt_helper_css():
    return prompt_layout.get_css('moss') + '''
.moss-helper-grid {grid-template-columns:minmax(300px,1.3fr) minmax(260px,1fr)}
.moss-helper-canvas-wrap {flex:1;min-height:200px}
.moss-helper-list {max-height:160px;flex-shrink:0}
.moss-helper-fields textarea {min-height:72px}
.moss-helper-fields [aria-invalid="true"] {border-color:var(--error-text-color,#b32438)}
.moss-helper-raw label {display:flex;flex-direction:column;gap:8px;flex:1}
.moss-helper-raw textarea {min-height:280px}
.moss-helper-fields :disabled {opacity:.55}
@media (max-width:800px) {.moss-helper-grid {display:flex;flex-direction:column;overflow:auto}.moss-helper-visual,.moss-helper-fields {overflow:visible;flex:none}.moss-helper-canvas-wrap {flex:none;height:38vh}}
'''


def get_prompt_helper_javascript():
    return prompt_layout.get_javascript() + Path(__file__).with_suffix('.js').read_text(encoding='utf-8')
