// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
(() => {
    if (window.wangpFlux3Layout) return;
    const helper = window.wangpFlux3Layout = {};
    const shared = window.wangpPromptLayout;
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    const copy = value => JSON.parse(JSON.stringify(value));
    const isObject = value => value !== null && typeof value === 'object' && !Array.isArray(value);
    const validId = value => typeof value === 'string' && /^[A-Za-z][A-Za-z0-9_]*$/.test(value) && !/^ref_image_\d+$/.test(value);
    const isEdit = row => Object.hasOwn(row, 'from');
    const targetBox = row => isEdit(row) ? row.tgt_bbox : row.bbox;
    const setTarget = (row, box) => { row[isEdit(row) ? 'tgt_bbox' : 'bbox'] = box; };
    helper.validBox = box => Array.isArray(box) && box.length === 4 && box.every(v => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1000) && box[2] > box[0] && box[3] > box[1];
    helper.validateRows = rows => {
        const ids = new Set();
        rows.forEach((row, i) => {
            const prefix = `Element ${i + 1}`;
            if (!isObject(row) || !validId(row.id)) throw new Error(`${prefix} needs a valid, non-reference id (letters, digits and underscores).`);
            if (ids.has(row.id)) throw new Error(`Duplicate element id: ${row.id}.`);
            ids.add(row.id);
            if (typeof row.desc !== 'string') throw new Error(`${prefix} needs a text description.`);
            if (isEdit(row)) {
                if (Object.hasOwn(row, 'bbox')) throw new Error(`${row.id} mixes bbox with editing fields. Use src_bbox and tgt_bbox for an edit.`);
                if (!/^(target|ref_image_[0-9])$/.test(row.from)) throw new Error(`${row.id}: from must be target or ref_image_0 through ref_image_9.`);
                if (row.from === 'target' ? row.src_bbox !== null : !helper.validBox(row.src_bbox)) throw new Error(`${row.id} needs ${row.from === 'target' ? 'a null' : 'a valid'} source box.`);
                if (!(helper.validBox(row.tgt_bbox) || row.tgt_bbox === null && row.from !== 'target')) throw new Error(`${row.id} needs a valid output box; only reference elements can have a null removal target.`);
            } else {
                if (Object.hasOwn(row, 'src_bbox') || Object.hasOwn(row, 'tgt_bbox')) throw new Error(`${row.id} needs a from value for its editing boxes.`);
                if (!helper.validBox(row.bbox)) throw new Error(`${row.id} needs [top, left, bottom, right] corners between 0 and 1000.`);
            }
        });
    };
    helper.parse = source => {
        const text = shared.stripCommentLines(source, /^\s*#/).trim();
        // Search for a complete trailing array, not brackets inside descriptions or captions.
        for (let i = 0; i < text.length; i++) {
            if (text[i] !== '[') continue;
            let rows;
            try { rows = JSON.parse(text.slice(i)); } catch (_) { continue; }
            if (!Array.isArray(rows)) continue;
            helper.validateRows(rows);
            return { caption: text.slice(0, i).trimEnd(), rows };
        }
        if (/\[\s*(?:\{|\]|$)/.test(text) || /^[{\[]/.test(text) || text.includes('```'))
            throw new Error('Expected a scene caption followed by one complete JSON element array. Keep the original text below and correct its syntax.');
        return { caption: text, rows: [] };
    };
    helper.serialize = (doc, comments) => shared.restoreComments(doc.caption + (doc.rows.length ? '\n\n' + JSON.stringify(doc.rows, null, 2) : ''), comments);

    // Read the ordered thumbnail strip, never the enlarged preview (which repeats a reference).
    // This is the same gallery class used by WanGP's reference-image preview controls.
    helper.referenceUrls = () => {
        const gallery = Array.from(document.querySelectorAll('.amg-remove-selected .amg-image-gallery')).find(node=>node.getClientRects().length);
        if (!gallery) return [];
        let images = gallery.querySelectorAll('.thumbnail-small img');
        if (!images.length) images = gallery.querySelectorAll('.thumbnail-lg img');
        return Array.from(images, image => {
            const url = new URL(image.currentSrc || image.src, location.href);
            url.searchParams.delete('__wangp_gallery_preview');
            return url.href;
        });
    };
    helper.referenceMode = () => {
        const component=(window.gradio_config?.components || []).find(item=>item.props?.label==='Inject Reference Images');
        if(!component)return '';
        const root=document.getElementById(component.props.elem_id || `component-${component.id}`);
        const text=root?.querySelector('input')?.value;
        const choice=component.props.choices.find(([label,value])=>label===text || value===text);
        return choice ? choice[1] : component.props.value;
    };
    function init(card) {
        if (card._flux3Layout) return card._flux3Layout;
        const config = JSON.parse(card.dataset.flux3Config);
        const popup = card.closest('[data-wangp-model-info-popup]');
        const canvas = card.querySelector('canvas'), ctx = canvas.getContext('2d');
        const fields = Object.fromEntries(Array.from(card.querySelectorAll('[data-field]'), node => [node.dataset.field, node]));
        const buttons = Object.fromEntries(Array.from(card.querySelectorAll('[data-action]'), node => [node.dataset.action, node]));
        const rawPanel = card.querySelector('[data-raw]'), rawInput = card.querySelector('[data-raw-input]');
        const grid = card.querySelector('.flux3-helper-grid'), list = card.querySelector('.flux3-helper-list');
        const status = card.querySelector('[data-status]'), refStatus = card.querySelector('[data-reference-status]');
        const state = { doc: {caption:'', rows:[]}, selected:-1, source:'', sourceDoc:'', comments:null,
            baseline:'', baselineDoc:'', refs:[], urls:[], loadId:0, snapshots:[], cursor:0, gesture:null,
            invalid:false, removedBoxes:{}, pendingConfirm:false, referenceErrors:[] };
        const promptField = () => document.getElementById(config.promptTarget)?.querySelector('textarea, input');
        const selected = () => state.doc.rows[state.selected];
        const message = (text='', error=false) => { status.textContent=text; status.classList.toggle('error', error); };
        const rawMode = () => !rawPanel.hidden;
        function fieldChanges() {
            const row = selected();
            return row && (fields.id.value !== row.id || fields.target.value !== (targetBox(row) === null ? '' : JSON.stringify(targetBox(row))) || fields.source.value !== (isEdit(row) && row.src_bbox !== null ? JSON.stringify(row.src_bbox) : ''));
        }
        function dirty() {
            if (rawMode()) return rawInput.value !== state.baseline;
            return state.source !== state.baseline || JSON.stringify(state.doc) !== state.baselineDoc || !!fieldChanges();
        }
        function snapshot() { return JSON.stringify({doc:state.doc, selected:state.selected, removedBoxes:state.removedBoxes}); }
        function history() {
            const value = snapshot();
            const previous = JSON.parse(state.snapshots[state.cursor]);
            if (JSON.stringify(previous.doc) !== JSON.stringify(state.doc) || JSON.stringify(previous.removedBoxes) !== JSON.stringify(state.removedBoxes)) {
                state.snapshots = state.snapshots.slice(0, state.cursor + 1);
                state.snapshots.push(value);
                if (state.snapshots.length > 100) state.snapshots.shift();
                state.cursor = state.snapshots.length - 1;
            }
            buttons.undo.disabled = state.cursor === 0;
            buttons.redo.disabled = state.cursor >= state.snapshots.length - 1;
            if(rawMode()) for(const key of ['add','delete','undo','redo'])buttons[key].disabled=true;
        }
        function undo(delta) {
            // Commit valid pending text as one operation before undo, without discarding an invalid draft.
            if (!commitFields()) return;
            history();
            const at = state.cursor + delta;
            if (at < 0 || at >= state.snapshots.length) return;
            state.cursor = at;
            Object.assign(state, JSON.parse(state.snapshots[at]));
            refresh(); message();
        }
        function options() {
            fields.from.replaceChildren();
            const choices = [['generation','Generated Element'], ['target','New Content in Edited Image']];
            // All supported IDs stay available for layouts prepared before references are uploaded.
            for (let i=0; i<10; i++) choices.push([`ref_image_${i}`, `Reference ${i + 1} — ref_image_${i}${state.refs[i] ? '' : ' (not loaded)'}`]);
            for (const [value, label] of choices) fields.from.add(new Option(label, value));
        }
        function refreshFields() {
            fields.caption.value = state.doc.caption;
            const row = selected();
            for (const key of ['id','desc','from','target','source','removed']) { fields[key].disabled=!row; fields[key].removeAttribute('aria-invalid'); }
            fields.id.value = row?.id || ''; fields.desc.value = row?.desc || '';
            fields.from.value = row ? (isEdit(row) ? row.from : 'generation') : 'generation';
            fields.target.value = row && targetBox(row) !== null ? JSON.stringify(targetBox(row)) : '';
            fields.source.value = row && isEdit(row) && row.src_bbox !== null ? JSON.stringify(row.src_bbox) : '';
            fields.source.disabled = !row || !isEdit(row) || row.from === 'target';
            fields.removed.disabled = fields.source.disabled;
            fields.removed.checked = !!row && targetBox(row) === null;
            fields.target.placeholder = fields.removed.checked ? 'Removed (no output box)' : '[100, 100, 900, 900]';
            fields.target.disabled = !row || fields.removed.checked;
            fields.view.options[1].disabled = fields.source.disabled;
            if (fields.source.disabled) fields.view.value='target';
        }
        function refreshList() {
            const scrollTop = list.scrollTop;
            list.replaceChildren();
            state.doc.rows.forEach((row,i) => {
                const button = document.createElement('button'); button.type='button';
                button.textContent = `${row.id}${targetBox(row) === null ? ' · Removed' : ''} — ${row.desc || 'Add a description'}`;
                button.title=button.textContent;
                button.setAttribute('aria-selected', String(i === state.selected));
                button.addEventListener('click', () => select(i)); list.append(button);
            });
            list.scrollTop = scrollTop;
            buttons.add.disabled = rawMode();
            buttons.delete.disabled = rawMode() || state.selected < 0;
            buttons.undo.disabled = rawMode() || state.cursor === 0;
            buttons.redo.disabled = rawMode() || state.cursor >= state.snapshots.length - 1;
        }
        function warnings() {
            const missing = state.doc.rows.filter(row => isEdit(row) && row.from !== 'target' && !state.refs[Number(row.from.slice(10))]);
            const citations = state.doc.rows.filter(row => !state.doc.caption.includes(`<${row.id}>`));
            refStatus.textContent = [`${state.refs.filter(Boolean).length} reference image(s) loaded.`,
                state.doc.rows.some(isEdit) && !helper.referenceMode().includes('I') ? 'Enable KI or I in Inject Reference Images before generating this edit.' : '',
                missing.length ? `Upload references for: ${missing.map(row=>row.id).join(', ')}.` : '',
                citations.length ? `Add caption tags for: ${citations.map(row=>`<${row.id}>`).join(', ')}.` : '',
                ...state.referenceErrors].filter(Boolean).join(' ');
        }
        function geometry() {
            const popupRect=popup.getBoundingClientRect();
            const left=clamp(popupRect.left,0,Math.max(0,innerWidth-popupRect.width));
            const top=clamp(popupRect.top,0,Math.max(0,innerHeight-popupRect.height));
            // WanGP can zoom the UI. Only adjust overflow, converting display pixels back to CSS pixels.
            if(left!==popupRect.left)popup.style.left=`${parseFloat(getComputedStyle(popup).left)+(left-popupRect.left)*popup.offsetWidth/popupRect.width}px`;
            if(top!==popupRect.top)popup.style.top=`${parseFloat(getComputedStyle(popup).top)+(top-popupRect.top)*popup.offsetHeight/popupRect.height}px`;
            const resolution = shared.readResolution(config.resolutionTarget);
            const row=selected(), source=fields.view.value==='source';
            const image=source && row ? state.refs[Number(row.from.slice(10))] : null;
            const shape=image || (fields.aspect.value==='reference' ? state.refs[0] : null);
            const width=shape ? shape.naturalWidth : resolution.width, height=shape ? shape.naturalHeight : resolution.height;
            const wrap=canvas.parentElement;
            const availableW=Math.max(1,wrap.clientWidth-4), availableH=Math.max(1,wrap.clientHeight-4);
            const scale=Math.min(availableW / width, availableH / height);
            canvas.style.width=`${Math.max(1,Math.floor(width*scale))}px`;
            canvas.style.height=`${Math.max(1,Math.floor(height*scale))}px`;
            canvas.width=Math.max(1,Math.round(width*scale*devicePixelRatio));
            canvas.height=Math.max(1,Math.round(height*scale*devicePixelRatio));
            card.querySelector('[data-resolution]').textContent = source ? `${row?.from || 'Source'} · ${image ? width+' × '+height : 'Reference not loaded'}` : `Output · ${shape ? 'First reference aspect' : width+' × '+height} · 0–1000 coordinates`;
            return {image,source};
        }
        function shownBoxes() {
            if (fields.view.value==='source') { const row=selected(); return row?.src_bbox ? [{row,index:state.selected,box:row.src_bbox}] : []; }
            return state.doc.rows.map((row,index)=>({row,index,box:targetBox(row)})).filter(item=>item.box);
        }
        function handles(box) {
            const [t,l,b,r]=box, x=(l+r)/2, y=(t+b)/2;
            return [['nw',l,t],['n',x,t],['ne',r,t],['e',r,y],['se',r,b],['s',x,b],['sw',l,b],['w',l,y]];
        }
        function draw() {
            if (popup.hidden || rawMode()) return;
            const {image,source}=geometry(), w=canvas.width, h=canvas.height;
            ctx.fillStyle='#fff'; ctx.fillRect(0,0,w,h);
            if (image) ctx.drawImage(image,0,0,w,h);
            else { ctx.strokeStyle='#e0e6ec'; ctx.lineWidth=devicePixelRatio;
                for(let i=1;i<4;i++){ctx.beginPath();ctx.moveTo(i*w/4,0);ctx.lineTo(i*w/4,h);ctx.moveTo(0,i*h/4);ctx.lineTo(w,i*h/4);ctx.stroke();} }
            ctx.save(); ctx.scale(w/1000,h/1000);
            for (const {row,index,box} of shownBoxes()) {
                const [t,l,b,r]=box, active=index===state.selected, lw=(active?2.5:1.5)*devicePixelRatio;
                ctx.fillStyle=active?'rgba(0,112,180,.18)':'rgba(38,93,146,.08)';ctx.fillRect(l,t,r-l,b-t);
                // Insets keep all four strokes visible for full-canvas boxes.
                const ix=lw*500/w, iy=lw*500/h;
                ctx.strokeStyle=active?'#0069aa':'#446986';ctx.lineWidth=lw*1000/w;
                ctx.strokeRect(Math.max(ix,l),Math.max(iy,t),Math.min(1000-ix,r)-Math.max(ix,l),Math.min(1000-iy,b)-Math.max(iy,t));
                ctx.save();ctx.setTransform(1,0,0,1,0,0);ctx.font=`600 ${12*devicePixelRatio}px sans-serif`;
                const label=row.id+(source?' · source':''), tw=Math.min(ctx.measureText(label).width+8*devicePixelRatio,w);
                const x=clamp(l*w/1000,0,w-tw), y=clamp(t*h/1000,0,h-20*devicePixelRatio);
                ctx.fillStyle=active?'#0069aa':'#446986';ctx.fillRect(x,y,tw,20*devicePixelRatio);
                ctx.fillStyle='#fff';ctx.fillText(label,x+4*devicePixelRatio,y+14*devicePixelRatio,tw-8*devicePixelRatio);
                if(active) for(const [,hx,hy] of handles(box)){const size=7*devicePixelRatio;ctx.fillStyle='#fff';ctx.strokeStyle='#0069aa';ctx.lineWidth=devicePixelRatio;const px=clamp(hx*w/1000-size/2,0,w-size),py=clamp(hy*h/1000-size/2,0,h-size);ctx.fillRect(px,py,size,size);ctx.strokeRect(px,py,size,size);}
                ctx.restore();
            }
            ctx.restore();
        }
        function refresh() { refreshFields(); refreshList(); warnings(); draw(); }
        function select(index) { if(!commitFields()) return; history(); state.selected=index; refresh(); }
        function commitFields() {
            if(rawMode()) return false;
            if(!selected()) return !state.invalid;
            const row=selected();
            try {
                const id=fields.id.value.trim();
                if(!validId(id) || state.doc.rows.some(other=>other!==row && other.id===id)) { fields.id.setAttribute('aria-invalid','true'); throw new Error('Use a unique element ID starting with a letter; reference IDs are reserved.'); }
                const boxes={};
                for(const key of ['target','source']) {
                    if(fields[key].disabled) continue;
                    try { boxes[key]=JSON.parse(fields[key].value); if(!helper.validBox(boxes[key])) throw new Error(); }
                    catch(_){fields[key].setAttribute('aria-invalid','true');throw new Error(`${key==='source'?'Source':'Output'} box needs [top, left, bottom, right], with positive size inside 0–1000.`);}
                }
                if(id!==row.id) { state.doc.caption=state.doc.caption.split(`<${row.id}>`).join(`<${id}>`); row.id=id; fields.caption.value=state.doc.caption; }
                if(boxes.target) setTarget(row,boxes.target);
                if(boxes.source) row.src_bbox=boxes.source;
                for(const key of ['id','target','source']) fields[key].removeAttribute('aria-invalid');
                // Canonicalize successful field input only; never silently fix invalid coordinates.
                fields.id.value=row.id;
                fields.target.value=targetBox(row)===null?'':JSON.stringify(targetBox(row));
                fields.source.value=row.src_bbox ? JSON.stringify(row.src_bbox):'';
                return true;
            } catch(error) { message(error.message,true); return false; }
        }
        function loadSource(source, opening=false) {
            try {
                const doc=helper.parse(source);
                state.doc=doc; state.source=source; state.sourceDoc=JSON.stringify(doc); state.comments=shared.collectComments(source, /^\s*#/);
                state.selected=doc.rows.length ? 0 : -1; state.invalid=false; state.removedBoxes={};
                rawPanel.hidden=true;grid.hidden=false;buttons.raw.disabled=false;
                state.snapshots=[snapshot()];state.cursor=0;
                if(opening) state.baselineDoc=JSON.stringify(doc);
                message();refresh();
                return true;
            } catch(error) {
                state.invalid=true;rawInput.value=source;rawPanel.hidden=false;grid.hidden=true;buttons.raw.disabled=true;
                for(const key of ['add','delete','undo','redo'])buttons[key].disabled=true;
                message(error.message,true);return false;
            }
        }
        function currentText() {
            return JSON.stringify(state.doc)===state.sourceDoc ? state.source : helper.serialize(state.doc,state.comments);
        }
        function releaseImages() { state.loadId++;state.refs=[];canvas.width=1;canvas.height=1; }
        function close() { popup.hidden=true;releaseImages(); }
        async function references() {
            const token=++state.loadId, urls=helper.referenceUrls();
            state.urls=urls;state.refs=[];state.referenceErrors=[];options();refresh();
            const results=await Promise.all(urls.map(async url=>{const img=new Image();img.src=url;try{await img.decode();return img;}catch(_){return null;}}));
            if(token!==state.loadId || !card.isConnected) return;
            state.refs=results; state.referenceErrors=results.flatMap((img,i)=>img?[]:[`Reference ${i+1} could not be loaded.`]);
            const origin=fields.from.value;
            options();fields.from.value=origin;warnings();draw();
        }
        function add(box=[200,200,800,800]) {
            if(!commitFields()) return;
            if(!state.gesture)history(); let n=1;while(state.doc.rows.some(row=>row.id===`obj_${n}`)) n++;
            const row={id:`obj_${n}`,desc:''};
            const editing=helper.referenceMode().includes('I') || state.doc.rows.some(isEdit);
            Object.assign(row,editing?{from:'target',src_bbox:null,tgt_bbox:box}:{bbox:box});
            state.doc.rows.push(row);state.selected=state.doc.rows.length-1;
            state.doc.caption += `${state.doc.caption ? ' ' : ''}<${row.id}>`;
            if(!state.gesture)history();refresh();message('Describe the new element and its place in the complete scene.');
        }
        function remove() {
            if(!selected() || !commitFields()) return;
            history();const row=selected();state.doc.caption=state.doc.caption.split(`<${row.id}>`).join('');
            state.doc.rows.splice(state.selected,1);state.selected=Math.min(state.selected,state.doc.rows.length-1);
            history();refresh();message();
        }
        function save() {
            if(rawMode()) { if(!loadSource(rawInput.value)) return false; }
            if(!commitFields()) return false;
            try { helper.validateRows(state.doc.rows); if(state.doc.rows.some(row=>!row.desc.trim())) throw new Error('Every element needs a description.'); }
            catch(error){message(error.message,true);return false;}
            const field=promptField();
            if(!field){message('The prompt field is no longer available. Keep this editor open and copy your prompt text.',true);return false;}
            if(field.value!==state.baseline){message('The main prompt changed while this editor was open. Copy your layout from Edit Prompt Text before closing; the newer prompt has not been overwritten.',true);return false;}
            if(JSON.stringify(helper.referenceUrls())!==JSON.stringify(state.urls)){message('Reference images or their order changed. Refresh References and check the source boxes before saving.',true);return false;}
            const value=dirty()?currentText():state.baseline;
            const setter=Object.getOwnPropertyDescriptor(field instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype,'value').set;
            setter.call(field,value);field.dispatchEvent(new Event('input',{bubbles:true}));field.dispatchEvent(new Event('change',{bubbles:true}));
            state.baseline=value;state.baselineDoc=JSON.stringify(state.doc);close();return true;
        }
        // Pointer editing uses the displayed canvas dimensions so handle hit areas stay usable at any aspect ratio.
        const point=event=>{const r=canvas.getBoundingClientRect();return {x:clamp((event.clientX-r.left)*1000/r.width,0,1000),y:clamp((event.clientY-r.top)*1000/r.height,0,1000)};};
        function hit(p) {
            const bounds=canvas.getBoundingClientRect(), boxes=shownBoxes(), active=boxes.find(item=>item.index===state.selected);
            if(active) for(const [mode,x,y] of handles(active.box)) if(Math.abs(p.x-x)<9*1000/bounds.width && Math.abs(p.y-y)<9*1000/bounds.height) return {...active,mode};
            for(const item of boxes.slice().reverse()) {const [t,l,b,r]=item.box;if(p.x>=l&&p.x<=r&&p.y>=t&&p.y<=b)return {...item,mode:'move'};}
            return null;
        }
        canvas.addEventListener('pointerdown',event=>{
            if(event.button!==0 || !commitFields() || fields.view.value==='source' && !selected()?.src_bbox) return;
            history();const start=point(event), found=event.altKey?null:hit(start);
            state.gesture={start,found,before:snapshot(),moved:false,pointer:event.pointerId};
            if(found){state.selected=found.index;state.gesture.box=found.box.slice();refresh();}
            canvas.setPointerCapture(event.pointerId);canvas.focus();event.preventDefault();
        });
        canvas.addEventListener('pointermove',event=>{
            const p=point(event), g=state.gesture;
            if(!g){const found=hit(p);canvas.style.cursor=found ? (found.mode==='move'?'move':found.mode+'-resize') : 'crosshair';return;}
            if(Math.abs(p.x-g.start.x)+Math.abs(p.y-g.start.y)<2 && !g.moved)return;
            g.moved=true;let box;
            if(!g.found) {
                box=[Math.round(Math.min(p.y,g.start.y)),Math.round(Math.min(p.x,g.start.x)),Math.round(Math.max(p.y,g.start.y)),Math.round(Math.max(p.x,g.start.x))];
                if(!helper.validBox(box))return;
                if(fields.view.value==='source')selected().src_bbox=box;
                else if(g.added===undefined){add(box);g.added=state.selected;}
                else setTarget(state.doc.rows[g.added],box);
            } else {
                let [t,l,b,r]=g.box;const mode=g.found.mode;
                if(mode==='move'){const dx=clamp(Math.round(p.x-g.start.x),-l,1000-r),dy=clamp(Math.round(p.y-g.start.y),-t,1000-b);box=[t+dy,l+dx,b+dy,r+dx];}
                else{if(mode.includes('n'))t=clamp(Math.round(p.y),0,b-1);if(mode.includes('s'))b=clamp(Math.round(p.y),t+1,1000);if(mode.includes('w'))l=clamp(Math.round(p.x),0,r-1);if(mode.includes('e'))r=clamp(Math.round(p.x),l+1,1000);box=[t,l,b,r];}
                if(fields.view.value==='source')selected().src_bbox=box;else setTarget(selected(),box);
            }
            refreshFields();draw();
        });
        function endGesture(event) {
            const g=state.gesture;if(!g)return;
            state.gesture=null;
            if(event.type==='pointercancel')Object.assign(state,JSON.parse(g.before));
            else if(g.moved)history();
            if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);
            refresh();
        }
        canvas.addEventListener('pointerup',endGesture);canvas.addEventListener('pointercancel',endGesture);
        for(const key of ['caption','desc']) {
            fields[key].addEventListener('input',()=>{if(key==='caption')state.doc.caption=fields.caption.value;else if(selected())selected().desc=fields.desc.value;refreshList();warnings();});
            fields[key].addEventListener('change',history);
        }
        for(const key of ['id','target','source'])fields[key].addEventListener('change',()=>{if(commitFields()){history();refreshList();warnings();draw();message();}});
        fields.from.addEventListener('change',()=>{
            const origin=fields.from.value;if(!commitFields()){fields.from.value=isEdit(selected())?selected().from:'generation';return;}
            history();const row=selected(),box=targetBox(row)||row.src_bbox.slice();
            if(origin==='generation'){delete row.from;delete row.src_bbox;delete row.tgt_bbox;row.bbox=box;}
            else{delete row.bbox;row.from=origin;row.src_bbox=origin==='target'?null:row.src_bbox||[0,0,1000,1000];row.tgt_bbox=box;}
            history();refresh();message(origin.startsWith('ref_image_')?'Switch to Source view to select the region in this reference.':'');
        });
        fields.removed.addEventListener('change',()=>{
            if(!commitFields()) {fields.removed.checked=targetBox(selected())===null;return;}
            history();const row=selected();
            if(fields.removed.checked){state.removedBoxes[row.id]=row.tgt_bbox;row.tgt_bbox=null;}
            else row.tgt_bbox=state.removedBoxes[row.id]||row.src_bbox.slice();
            history();refresh();message();
        });
        fields.view.addEventListener('change',()=>{if(commitFields()){history();draw();}});
        fields.aspect.addEventListener('change',draw);
        buttons.add.addEventListener('click',()=>add());buttons.delete.addEventListener('click',remove);
        buttons.undo.addEventListener('click',()=>undo(-1));buttons.redo.addEventListener('click',()=>undo(1));
        buttons.references.addEventListener('click',()=>{if(commitFields()){history();references();}});
        buttons.raw.addEventListener('click',()=>{if(!commitFields())return;history();rawInput.value=currentText();rawPanel.hidden=false;grid.hidden=true;buttons.raw.disabled=true;for(const key of ['add','delete','undo','redo'])buttons[key].disabled=true;message('Changes stay in the editor until Save Prompt.');});
        buttons.retry.addEventListener('click',()=>loadSource(rawInput.value));buttons.save.addEventListener('click',save);
        card.addEventListener('keydown',event=>{
            if(event.key==='Tab') {
                const controls=Array.from(card.querySelectorAll('button:not(:disabled), input:not(:disabled), textarea:not(:disabled), select:not(:disabled), canvas[tabindex]')).filter(node=>node.getClientRects().length);
                const at=controls.indexOf(document.activeElement), next=(at+(event.shiftKey?-1:1)+controls.length)%controls.length;
                event.preventDefault();controls[next].focus();return;
            }
            if(event.key==='Escape'){event.preventDefault();event.stopPropagation();window.wangpModelInfo.requestClose(popup);return;}
            if(event.target!==canvas)return;
            if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='z'){event.preventDefault();undo(event.shiftKey?1:-1);}
            else if(event.key==='Delete'||event.key==='Backspace'){event.preventDefault();remove();}
        });
        popup.addEventListener('wangp:model-info-before-close',event=>{
            if(!dirty()){releaseImages();return;}event.preventDefault();if(state.pendingConfirm)return;state.pendingConfirm=true;
            window.wangpConfirm({title:'Discard Layout Changes?',message:'Save the layout to the prompt, discard your edits, or keep editing.',buttons:[
                {text:'Keep Editing',value:'keep',cancel:true},{text:'Discard',value:'discard',danger:true},
                {text:'Save Changes',value:'save',primary:true,action:save}]}).then(choice=>{state.pendingConfirm=false;if(choice==='discard')close();});
        });
        const api={open(){
            state.baseline=promptField()?.value||'';state.baselineDoc='';fields.view.value='target';
            const resolution=(window.gradio_config?.components || []).find(item=>item.props?.elem_id===config.resolutionTarget);
            fields.aspect.value=helper.referenceMode().includes('K') && resolution?.props.label!=='Output Resolution'?'reference':'resolution';
            loadSource(state.baseline,true);references();
            for(const panel of card.querySelectorAll('.flux3-helper-content, .flux3-helper-grid, .flux3-helper-visual, .flux3-helper-fields'))panel.scrollTop=0;
            (rawMode()?rawInput:canvas).focus({preventScroll:true});
        },draw};
        card._flux3Layout=api;return api;
    }
    document.addEventListener('wangp:model-info-opened',event=>{const card=event.target.querySelector?.('[data-flux3-helper]');if(card)init(card).open();});
    const redraw=()=>document.querySelectorAll('[data-flux3-helper]').forEach(card=>card._flux3Layout?.draw());
    window.addEventListener('resize',redraw);
    window.addEventListener('pointerup',redraw);
})();
