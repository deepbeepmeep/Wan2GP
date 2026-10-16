// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
(() => {
    if (window.wangpMossLayout) return;
    const helper = window.wangpMossLayout = {}, shared = window.wangpPromptLayout;
    const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
    const comments = /^\s*#/;
    const empty = caption => ({summary:{meta:[],canvas:'single',caption},items:[],bbox_coord_space:'abs_centralized',bbox_format:'yxyx'});

    // Recover punctuation only. Never evaluate input, invent unfinished values or drop unknown fields.
    helper.parse = source => {
        let raw = shared.stripCommentLines(source, comments).trim(), notes = [];
        const note = text => {if (!notes.includes(text)) notes.push(text);};
        if (raw.startsWith('```')) {
            raw = raw.replace(/^```(?:json)?\s*\n?/i, '').replace(/\s*```\s*$/, '').trim();
            note('Removed Markdown fences.');
        }
        if (!raw || !/^[{\[]/.test(raw)) {
            if (/^['"]?(?:summary|items)['"]?\s*:/.test(raw)) throw new Error('The layout is missing its opening object brace.');
            return {doc:empty(raw), notes, plain:true};
        }
        let doc;
        {
            let at = 0;
            const fail = message => {const before=raw.slice(0,at),line=before.split('\n').length,col=at-before.lastIndexOf('\n');throw new Error(`${message} (line ${line}, column ${col}). Original text is preserved.`);};
            const ws = () => {while (/\s/.test(raw[at] || '') && at<raw.length) at++;};
            const string = () => {
                const quote=raw[at++]; let value='';
                if (quote==="'") note('Converted single-quoted strings to JSON strings.');
                while (at<raw.length) {
                    let ch=raw[at++];
                    if (ch===quote) return value;
                    if (ch==='\\') {
                        if (at===raw.length) fail('Unfinished escape');
                        ch=raw[at++];
                        const escapes={'"':'"',"'":"'",'\\':'\\','/':'/','b':'\b','f':'\f','n':'\n','r':'\r','t':'\t'};
                        if (ch==='u') {const hex=raw.slice(at,at+4);if(!/^[a-f\d]{4}$/i.test(hex))fail('Invalid Unicode escape');value+=String.fromCharCode(parseInt(hex,16));at+=4;}
                        else if (ch==='_') {value+='_';note('Removed Markdown underscore escapes.');}
                        else if (Object.hasOwn(escapes,ch)) value+=escapes[ch];
                        else fail('Invalid string escape');
                    } else {if(ch==='\n'||ch==='\r') note('Escaped literal line breaks inside strings.');value+=ch;}
                }
                fail('Unfinished quoted string');
            };
            const value = (depth=0) => {
                if(depth>100) fail('Layout nesting is too deep');
                ws();const ch=raw[at];
                if(ch==='"'||ch==="'") return string();
                if(ch==='{'||ch==='[') {
                    at++;const object=ch==='{', end=object?'}':']', out=object?Object.create(null):[];
                    ws(); if(raw[at]===end){at++;return out;}
                    while (at<raw.length) {
                        if(object) {
                            ws();let key;
                            if(raw[at]==='"'||raw[at]==="'")key=string();
                            else {const m=raw.slice(at).match(/^[A-Za-z_][\w-]*/);if(!m)fail('Expected a JSON key');key=m[0];at+=key.length;note('Quoted unquoted JSON keys.');}
                            if(Object.hasOwn(out,key)) fail(`Duplicate key ${key}`);
                            ws();if(raw[at++]!==':')fail('Expected a colon after the key');
                            out[key]=value(depth+1);
                        } else out.push(value(depth+1));
                        ws();if(raw[at]===end){at++;return out;}
                        if(at===raw.length){note('Completed missing closing brackets.');return out;}
                        if(raw[at]===',') {
                            at++;ws();if(raw[at]===end){at++;note('Removed trailing commas.');return out;}
                            if(at===raw.length){note('Removed trailing commas.');note('Completed missing closing brackets.');return out;}
                        } else if(raw[at]==='}'||raw[at]===']') fail('Mismatched closing bracket');
                        else note('Inserted missing commas.');
                    }
                    fail('Missing object field or array value');
                }
                const m=raw.slice(at).match(/^(?:-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)(?=\s|[,}\]]|$)/);
                if(!m)fail('Expected a complete JSON value');
                at+=m[0].length;const parsed=JSON.parse(m[0]);if(typeof parsed==='number'&&!Number.isFinite(parsed))fail('Number is outside the supported range');return parsed;
            };
            doc=value();ws();if(at!==raw.length)fail('Unexpected text after the layout');
        }
        if (!doc || typeof doc!=='object' || Array.isArray(doc) || !doc.summary || typeof doc.summary!=='object' || Array.isArray(doc.summary) || !Array.isArray(doc.items))
            throw new Error('Expected a summary object and an items array. Original text is preserved.');
        if (doc.summary.caption!==undefined && typeof doc.summary.caption!=='string') throw new Error('summary.caption must be text.');
        for(const [i,item] of doc.items.entries()) {
            if(!item || typeof item!=='object' || Array.isArray(item))throw new Error(`Item ${i+1} must be an object.`);
            for(const key of ['label','caption','text'])if(item[key]!==undefined && typeof item[key]!=='string')throw new Error(`Item ${i+1}: ${key} must be text.`);
        }
        for(const [key,expected] of [['bbox_format','yxyx'],['bbox_coord_space','abs_centralized']]) {
            if(doc[key]===undefined){doc[key]=expected;note(`Added ${key}: ${expected}.`);}
            else if(doc[key]!==expected)throw new Error(`This editor requires ${key}: ${expected}; coordinates have not been reinterpreted.`);
        }
        return {doc,notes,plain:false};
    };
    helper.validBox = (box, size) => Array.isArray(box) && box.length===4 && box.every(Number.isFinite) &&
        box[0]<box[2] && box[1]<box[3] && box[0]>=-size.height/2 && box[2]<=size.height/2 && box[1]>=-size.width/2 && box[3]<=size.width/2;
    helper.boxProblem = (box, size) => {
        if(!Array.isArray(box)||box.length!==4||!box.every(Number.isFinite))return 'The box needs four finite numbers: [top, left, bottom, right].';
        if(box[0]>=box[2]||box[1]>=box[3])return 'The top must be above the bottom and the left must be before the right.';
        const edges=[];
        if(box[0]<-size.height/2)edges.push(`top ${box[0]} is above ${-size.height/2}`);
        if(box[2]>size.height/2)edges.push(`bottom ${box[2]} is below ${size.height/2}`);
        if(box[1]<-size.width/2)edges.push(`left ${box[1]} is before ${-size.width/2}`);
        if(box[3]>size.width/2)edges.push(`right ${box[3]} is beyond ${size.width/2}`);
        return edges.length ? `${edges.join('; ')} on the ${size.width} × ${size.height} canvas.` : '';
    };
    helper.fitBox = (box, size) => {
        if(!Array.isArray(box)||box.length!==4||!box.every(Number.isFinite))return box;
        if(helper.validBox(box,size))return box;
        const axis=(a,b,length)=>{
            const start=Math.round(Math.min(a,b)),end=Math.round(Math.max(a,b));
            const extent=clamp(end-start,1,length),low=clamp(start,-length/2,length/2-extent);
            return [low,low+extent];
        };
        const [t,b]=axis(box[0],box[2],size.height),[l,r]=axis(box[1],box[3],size.width);
        return [t,l,b,r];
    };
    helper.serialize = (doc, source) => shared.restoreComments(JSON.stringify(doc,null,2),shared.collectComments(source,comments));
    const handles = ([t,l,b,r]) => [['nw',l,t],['n',(l+r)/2,t],['ne',r,t],['e',r,(t+b)/2],['se',r,b],['s',(l+r)/2,b],['sw',l,b],['w',l,(t+b)/2]];

    function init(card) {
        if(card._mossLayout)return card._mossLayout;
        const config=JSON.parse(card.dataset.mossConfig),popup=card.closest('[data-wangp-model-info-popup]');
        const canvas=card.querySelector('canvas'),ctx=canvas.getContext('2d'),wrap=canvas.parentElement;
        const raw=card.querySelector('[data-raw]'),rawInput=card.querySelector('[data-raw-input]'),grid=card.querySelector('.moss-helper-grid');
        const status=card.querySelector('[data-status]'),list=card.querySelector('.moss-helper-list');
        const fields=Object.fromEntries([...card.querySelectorAll('[data-field]')].map(node=>[node.dataset.field,node]));
        const buttons=Object.fromEntries([...card.querySelectorAll('[data-action]')].map(node=>[node.dataset.action,node]));
        const state={doc:empty(''),selected:-1,size:{width:1024,height:1024},baseline:'',baselineDoc:'',source:'',sourceDoc:'',notes:[],plain:false,history:[],cursor:0,gesture:null};
        const field=()=>document.getElementById(config.promptTarget)?.querySelector('textarea,input');
        const item=()=>state.doc.items[state.selected];
        const message=(text='',error=false)=>{status.textContent=text;status.classList.toggle('error',error);};
        const snapshot=()=>JSON.stringify(state.doc);
        const dirty=()=>raw.hidden ? snapshot()!==state.baselineDoc || state.source!==state.baseline || (item() && fields.bbox.value!==(item().bbox===undefined?'':JSON.stringify(item().bbox))) : rawInput.value!==state.baseline;
        const currentText=()=>snapshot()===state.sourceDoc ? state.source : helper.serialize(state.doc,state.source);
        function checkpoint() {
            const text=snapshot();if(text===state.history[state.cursor])return;
            state.history.splice(state.cursor+1);state.history.push(text);state.cursor=state.history.length-1;
            buttons.undo.disabled=state.cursor===0;buttons.redo.disabled=true;
        }
        function restore(delta) {
            if(!commit())return;checkpoint();const next=state.cursor+delta;if(next<0||next>=state.history.length)return;
            state.cursor=next;state.doc=JSON.parse(state.history[next]);state.selected=Math.min(state.selected,state.doc.items.length-1);refresh();
        }
        function warnings(notice='') {
            const bad=state.doc.items.flatMap((row,i)=>{const problem=helper.boxProblem(row.bbox,state.size);return problem?[`Item ${i+1}: ${problem}`]:[];});
            message([...state.notes,notice,...bad.slice(0,3),bad.length>3?`${bad.length-3} more items need valid boxes.`:'',bad.length?'Edit coordinates or use Fit Boxes to Canvas.':'',state.notes.length?'Save Prompt applies these repairs; the original prompt is unchanged.':''].filter(Boolean).join(' '),bad.length>0);
        }
        function refreshList() {
            list.replaceChildren();state.doc.items.forEach((row,i)=>{
                const button=document.createElement('button');button.type='button';button.textContent=`${i+1}. ${row.text || row.caption || row.label || 'New item'}`;
                button.setAttribute('aria-selected',String(i===state.selected));button.addEventListener('click',()=>{if(!commit())return;checkpoint();state.selected=i;refresh();});list.appendChild(button);
            });
        }
        function refreshFields() {
            fields.scene.value=state.doc.summary.caption || '';
            for(const key of ['label','caption','text','bbox']){fields[key].disabled=!item();fields[key].value=key==='bbox'?(item()?.bbox===undefined?'':JSON.stringify(item().bbox)):item()?.[key] || '';fields[key].removeAttribute('aria-invalid');}
        }
        function draw() {
            if(popup.hidden||!raw.hidden)return;
            const {width:w,height:h}=state.size,scale=Math.min(wrap.clientWidth/w,(wrap.clientHeight||400)/h);
            canvas.style.width=`${Math.max(1,w*scale)}px`;canvas.style.height=`${Math.max(1,h*scale)}px`;
            canvas.width=Math.round(Math.min(1400,w));canvas.height=Math.round(canvas.width*h/w);
            ctx.scale(canvas.width/w,canvas.height/h);ctx.fillStyle=getComputedStyle(card).getPropertyValue('--background-fill-primary').trim()||'#fff';ctx.fillRect(0,0,w,h);
            ctx.translate(w/2,h/2);const unit=1/Math.max(scale,.01);
            ctx.strokeStyle='#8a929b';ctx.lineWidth=unit;ctx.setLineDash([4*unit,4*unit]);ctx.beginPath();ctx.moveTo(-w/2,0);ctx.lineTo(w/2,0);ctx.moveTo(0,-h/2);ctx.lineTo(0,h/2);ctx.stroke();ctx.setLineDash([]);
            state.doc.items.forEach((row,i)=>{
                if(!Array.isArray(row.bbox)||row.bbox.length!==4||!row.bbox.every(Number.isFinite))return;
                const [t,l,b,r]=row.bbox,color=helper.validBox(row.bbox,state.size)?`hsl(${(i*61+200)%360} 65% 48%)`:'#d32f46';
                ctx.strokeStyle=color;ctx.fillStyle=color;ctx.globalAlpha=.09;ctx.fillRect(l,t,r-l,b-t);ctx.globalAlpha=1;ctx.lineWidth=(i===state.selected?2:1)*unit;
                ctx.strokeRect(l+unit/2,t+unit/2,r-l-unit,b-t-unit);ctx.font=`${12*unit}px sans-serif`;
                const label=`${i+1}. ${row.text || row.label || 'Item'}`;ctx.fillText(label,l+6*unit,t+17*unit,Math.max(1,r-l-12*unit));
                if(i===state.selected)for(const [,x,y] of handles(row.bbox))ctx.fillRect(clamp(x-3*unit,-w/2,w/2-6*unit),clamp(y-3*unit,-h/2,h/2-6*unit),6*unit,6*unit);
            });
        }
        function refresh() {
            refreshFields();refreshList();draw();warnings();
            buttons.delete.disabled=!item();buttons.back.disabled=state.selected<=0;buttons.front.disabled=state.selected<0||state.selected>=state.doc.items.length-1;
            buttons.undo.disabled=state.cursor===0;buttons.redo.disabled=state.cursor>=state.history.length-1;
        }
        function commit() {
            if(!raw.hidden)return false;
            if(item() && fields.bbox.value.trim()) {
                let box;try{box=JSON.parse(fields.bbox.value);}catch(_){message('Use four numbers: [top, left, bottom, right].',true);fields.bbox.setAttribute('aria-invalid','true');return false;}
                if(!Array.isArray(box)||box.length!==4||!box.every(Number.isFinite)){message('The box needs four finite numbers.',true);return false;}
                item().bbox=box;
            } else if(item()?.bbox!==undefined){message('The box cannot be empty. Enter four numbers or delete the item.',true);fields.bbox.setAttribute('aria-invalid','true');return false;}
            return true;
        }
        function load(source,opening=false) {
            try {
                const parsed=helper.parse(source);Object.assign(state,parsed);state.source=source;state.sourceDoc=snapshot();state.selected=state.doc.items.length?0:-1;
                state.history=[snapshot()];state.cursor=0;raw.hidden=true;grid.hidden=false;
                for(const key of ['add','fit','raw'])buttons[key].disabled=false;
                if(opening)state.baselineDoc=snapshot();refresh();return true;
            } catch(error) {
                rawInput.value=source;raw.hidden=false;grid.hidden=true;
                for(const key of ['add','delete','back','front','undo','redo','fit','raw'])buttons[key].disabled=true;
                message(error.message,true);return false;
            }
        }
        function add(box=[-state.size.height/8,-state.size.width/4,state.size.height/8,state.size.width/4]) {
            if(!commit())return;checkpoint();state.doc.items.push({label:'object',caption:'',bbox:box.map(Math.round)});state.selected=state.doc.items.length-1;checkpoint();refresh();
        }
        function remove() {if(!item()||!commit())return;checkpoint();state.doc.items.splice(state.selected,1);state.selected=Math.min(state.selected,state.doc.items.length-1);checkpoint();refresh();}
        function save() {
            if(!raw.hidden&&!load(rawInput.value))return false;
            if(!commit())return false;
            const bad=state.doc.items.findIndex(row=>!helper.validBox(row.bbox,state.size));
            if(bad>=0){warnings();return false;}
            const target=field();if(!target){message('The prompt field is unavailable. Copy your work from Edit Prompt Text.',true);return false;}
            if(target.value!==state.baseline){message('The main prompt changed while this editor was open. Copy your layout from Edit Prompt Text before closing; the newer prompt has not been overwritten.',true);return false;}
            const value=dirty()||state.notes.length ? helper.serialize(state.doc,state.source) : state.baseline;
            const setter=Object.getOwnPropertyDescriptor(target instanceof HTMLTextAreaElement?HTMLTextAreaElement.prototype:HTMLInputElement.prototype,'value').set;
            setter.call(target,value);target.dispatchEvent(new Event('input',{bubbles:true}));target.dispatchEvent(new Event('change',{bubbles:true}));
            state.baseline=value;state.baselineDoc=snapshot();popup.hidden=true;return true;
        }
        const point=event=>{const r=canvas.getBoundingClientRect();return {x:clamp(Math.round((event.clientX-r.left)/r.width*state.size.width-state.size.width/2),-state.size.width/2,state.size.width/2),y:clamp(Math.round((event.clientY-r.top)/r.height*state.size.height-state.size.height/2),-state.size.height/2,state.size.height/2)};};
        function hit(p) {
            const scale=state.size.width/canvas.getBoundingClientRect().width;
            if(item()&&helper.validBox(item().bbox,state.size))for(const [mode,x,y] of handles(item().bbox))if(Math.abs(p.x-x)<8*scale&&Math.abs(p.y-y)<8*scale)return {index:state.selected,mode};
            for(let i=state.doc.items.length-1;i>=0;i--){const box=state.doc.items[i].bbox;if(!helper.validBox(box,state.size))continue;const [t,l,b,r]=box;if(p.x>=l&&p.x<=r&&p.y>=t&&p.y<=b)return {index:i,mode:'move'};}return null;
        }
        canvas.addEventListener('pointerdown',event=>{
            if(event.button!==0||!commit())return;checkpoint();const start=point(event),found=event.altKey?null:hit(start);
            state.gesture={start,found,before:snapshot(),selected:state.selected};if(found){state.selected=found.index;state.gesture.box=item().bbox.slice();refresh();}
            canvas.setPointerCapture(event.pointerId);canvas.focus();event.preventDefault();
        });
        canvas.addEventListener('pointermove',event=>{
            const p=point(event),g=state.gesture;if(!g){const found=hit(p);canvas.style.cursor=found?(found.mode==='move'?'move':found.mode+'-resize'):'crosshair';return;}
            if(Math.abs(p.x-g.start.x)+Math.abs(p.y-g.start.y)<2)return;
            if(!g.found){const box=[Math.min(p.y,g.start.y),Math.min(p.x,g.start.x),Math.max(p.y,g.start.y),Math.max(p.x,g.start.x)];if(box[0]===box[2]||box[1]===box[3])return;
                if(g.added===undefined){state.doc.items.push({label:'object',caption:'',bbox:box});g.added=state.doc.items.length-1;state.selected=g.added;}else item().bbox=box;
            }else{let [t,l,b,r]=g.box;const mode=g.found.mode,{width:w,height:h}=state.size;
                if(mode==='move'){const dx=clamp(p.x-g.start.x,-w/2-l,w/2-r),dy=clamp(p.y-g.start.y,-h/2-t,h/2-b);item().bbox=[t+dy,l+dx,b+dy,r+dx];}
                else{if(mode.includes('n'))t=clamp(p.y,-h/2,b-1);if(mode.includes('s'))b=clamp(p.y,t+1,h/2);if(mode.includes('w'))l=clamp(p.x,-w/2,r-1);if(mode.includes('e'))r=clamp(p.x,l+1,w/2);item().bbox=[t,l,b,r];}
            }refreshFields();draw();
        });
        const end=event=>{const g=state.gesture;if(!g)return;state.gesture=null;if(event.type==='pointercancel'){state.doc=JSON.parse(g.before);state.selected=g.selected;}else checkpoint();if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);refresh();};
        canvas.addEventListener('pointerup',end);canvas.addEventListener('pointercancel',end);
        for(const key of ['scene','label','caption','text']) {
            fields[key].addEventListener('input',()=>{if(key==='scene')state.doc.summary.caption=fields.scene.value;else if(item())item()[key]=fields[key].value;refreshList();draw();});
            fields[key].addEventListener('change',checkpoint);
        }
        fields.bbox.addEventListener('input',()=>fields.bbox.setAttribute('aria-invalid','false'));
        fields.bbox.addEventListener('change',()=>{if(commit()){checkpoint();draw();warnings();}});
        buttons.add.addEventListener('click',()=>add());buttons.delete.addEventListener('click',remove);
        for(const [key,delta] of [['back',-1],['front',1]])buttons[key].addEventListener('click',()=>{if(!commit())return;checkpoint();const i=state.selected,j=i+delta;if(j<0||j>=state.doc.items.length)return;[state.doc.items[i],state.doc.items[j]]=[state.doc.items[j],state.doc.items[i]];state.selected=j;checkpoint();refresh();});
        buttons.undo.addEventListener('click',()=>restore(-1));buttons.redo.addEventListener('click',()=>restore(1));
        buttons.fit.addEventListener('click',()=>{
            if(!commit())return;checkpoint();let count=0;
            for(const row of state.doc.items){const fitted=helper.fitBox(row.bbox,state.size);if(fitted===row.bbox)continue;row.bbox=fitted;count++;}
            checkpoint();refresh();warnings(`Fitted ${count} box${count===1?'':'es'} to the canvas, preserving size where possible. Undo restores previous coordinates; Save Prompt applies the changes.`);
        });
        buttons.raw.addEventListener('click',()=>{if(!commit())return;checkpoint();rawInput.value=currentText();raw.hidden=false;grid.hidden=true;for(const key of ['add','delete','back','front','undo','redo','fit','raw'])buttons[key].disabled=true;message('The original prompt stays unchanged until Save Prompt.');});
        buttons.retry.addEventListener('click',()=>load(rawInput.value));buttons.save.addEventListener('click',save);
        card.addEventListener('keydown',event=>{
            if(event.key==='Escape'){event.preventDefault();event.stopPropagation();window.wangpModelInfo.requestClose(popup);return;}
            if(event.key==='Tab'){const nodes=[...card.querySelectorAll('button:not(:disabled),input:not(:disabled),textarea:not(:disabled),canvas[tabindex]')].filter(n=>n.getClientRects().length);const next=(nodes.indexOf(document.activeElement)+(event.shiftKey?-1:1)+nodes.length)%nodes.length;event.preventDefault();nodes[next].focus();return;}
            if(event.target!==canvas)return;
            if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='z'){event.preventDefault();restore(event.shiftKey?1:-1);}else if(event.key==='Delete'||event.key==='Backspace'){event.preventDefault();remove();}
        });
        popup.addEventListener('wangp:model-info-before-close',event=>{
            if(!dirty())return;event.preventDefault();if(state.confirming)return;state.confirming=true;
            window.wangpConfirm({title:'Discard Layout Changes?',message:'Save the layout to the prompt, discard your edits, or keep editing.',buttons:[{text:'Keep Editing',value:'keep',cancel:true},{text:'Discard',value:'discard',danger:true},{text:'Save Changes',value:'save',primary:true,action:save}]}).then(choice=>{state.confirming=false;if(choice==='discard')popup.hidden=true;});
        });
        const api={open(){state.size=shared.readResolution(config.resolutionTarget);state.baseline=field()?.value||'';state.baselineDoc='';state.source=state.baseline;load(state.baseline,true);card.querySelector('[data-resolution]').textContent=`${state.size.width} × ${state.size.height} pixels · Origin at Canvas Center`;card.querySelector('[data-bounds]').textContent=`Pixel corners relative to the canvas center, X right and Y down. Top/Bottom: ${-state.size.height/2} to ${state.size.height/2}. Left/Right: ${-state.size.width/2} to ${state.size.width/2}.`;requestAnimationFrame(draw);(raw.hidden?canvas:rawInput).focus({preventScroll:true});},draw};
        card._mossLayout=api;return api;
    }
    document.addEventListener('wangp:model-info-opened',event=>{const card=event.target.querySelector?.('[data-moss-helper]');if(card)init(card).open();});
    const redraw=()=>document.querySelectorAll('[data-moss-helper]').forEach(card=>card._mossLayout?.draw());
    window.addEventListener('resize',redraw);window.addEventListener('pointerup',redraw);
})();
