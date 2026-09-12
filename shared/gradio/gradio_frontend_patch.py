"""Avoid propagating Gradio layout updates into unaffected Svelte branches."""
from functools import lru_cache, wraps
from pathlib import Path

from fastapi.responses import Response
from gradio import routes

from shared.gradio.gradio_model_change_queue import _REPLACEMENTS as _QUEUE_REPLACEMENTS


_EDITOR_PATH = Path(__file__).parent / 'wangp_image_editor/templates/component/index.js'
# Pixi's scheduler only serves wall-clock GC tasks in this bundle. Retain its
# callbacks, offsets and repeat bookkeeping, but wake at their actual deadlines.
# The separate pointer ticker must still run whenever hit testing unpauses it.
_EDITOR_SCHEDULER = """
JC = class extends JC {
    init() {}
    repeat(...args) {
        const id = super.repeat(...args);
        this.wangpSchedule();
        return id;
    }
    cancel(id) {
        super.cancel(id);
        this.wangpSchedule();
    }
    wangpSchedule() {
        clearTimeout(this.wangpTimer);
        if (!this._tasks.length) return;
        const due = Math.min(...this._tasks.map(task => task.last + task.offset + task.duration));
        this.wangpTimer = setTimeout(() => {
            super._update();
            this.wangpSchedule();
        }, Math.max(0, Math.ceil(due - performance.now())));
    }
    destroy() {
        clearTimeout(this.wangpTimer);
        super.destroy();
    }
};
"""


# Gradio mutates layout nodes in place and publishes the entire tree each frame.
# Mark affected nodes and their ancestors before publishing. A Node can skip an
# incoming $set only when both its revision and ALL incoming props are unchanged.
# Same-value outputs still mark their nodes; normal event/prop handling is kept.
# Dynamic layout rebuilds invalidate every node, preserving upstream rendering.
_NODE_INPUTS = '"root"in w&&t(1,n=w.root),"node"in w&&t(0,s=w.node)'
_NODE_SKIP = """
const wangpVersion = ("node" in w ? w.node : s).__wangp_revision;
if (wangpVersion === wangpNodeVersion && Object.keys(w).every(key => w[key] === wangpNodeInputs[key])) return;
wangpNodeVersion = wangpVersion;
Object.assign(wangpNodeInputs, w);
"""
# Gradio's helper holds only its constructor inputs and bound dispatch/load
# methods. Reuse it across value/status updates; replace it on context changes.
_GRADIO_CONTEXT = """
let wangpGradioArgs, wangpGradioValue;
function wangpGradio(...args) {
    if (!wangpGradioArgs || args.some((arg, index) => arg !== wangpGradioArgs[index])) {
        wangpGradioArgs = args;
        wangpGradioValue = new er(...args);
    }
    return wangpGradioValue;
}
"""
_MARK_ANCESTORS = """
for (let node = f; node; node = node.parent) wangpDirty.add(node);
}
for (const node of wangpDirty) node.__wangp_revision = (node.__wangp_revision || 0) + 1;
"""
# Gradio's status store replaces an entry on each actual status/progress change.
# Mn also runs after data messages and visits previously completed outputs. Do
# not publish those same entries again: doing so dirties whole old forms.
# The input map also retains completed entries. Re-publishing pending=False
# after a media-selection callback invalidates its gallery on unrelated events.
_STATUS_ORIGINAL = 'function Mn(S){let J=[];Object.entries(S).forEach(([R,oe])=>{if(d.closed&&oe.status==="error")return;let de=u.find(he=>he.id==oe.fn_index);de!==void 0&&(oe.scroll_to_output=de.scroll_to_output,oe.show_progress=de.show_progress,J.push({id:parseInt(R),prop:"loading_status",value:oe}))});const K=ie.get_inputs_to_update(),pe=Array.from(K).map(([R,oe])=>({id:R,prop:"pending",value:oe==="pending"}));T([...J,...pe])}'
_STATUS_UPDATE = """const wangpStatusCache=new Map(), wangpPendingCache=new Map();
function Mn(S){
    const updates=[], dependencies=new Map(u.map(fn=>[fn.id,fn]));
    for(const [id,status] of Object.entries(S)){
        if(d.closed&&status.status==="error")continue;
        const dependency=dependencies.get(status.fn_index);
        if(dependency===undefined)continue;
        const previous=wangpStatusCache.get(id);
        if(previous&&previous[0]===status&&previous[1]===dependency.scroll_to_output&&previous[2]===dependency.show_progress)continue;
        status.scroll_to_output=dependency.scroll_to_output;
        status.show_progress=dependency.show_progress;
        wangpStatusCache.set(id,[status,status.scroll_to_output,status.show_progress]);
        updates.push({id:parseInt(id),prop:"loading_status",value:status});
    }
    const inputs=[];
    for(const [id,status] of ie.get_inputs_to_update()){
        const pending=status==="pending";
        if(wangpPendingCache.get(id)===pending)continue;
        wangpPendingCache.set(id,pending);
        inputs.push({id,prop:"pending",value:pending});
    }
    T([...updates,...inputs]);
}
"""

_PATCHES = {
    'Gallery-D7vc32lN.js': [
        # Selection is a user event. Server values/indices notify change once,
        # after normalization; index-only updates must still refresh consumers.
        ('let ne=m;function se(s){', 'let ne=m,wangpGalleryUser=false,wangpGalleryExplicit=false,wangpGalleryChanged=false;function se(s){wangpGalleryUser=true;'),
        ('function Re(s){switch(s.code){', 'function Re(s){if(["Escape","ArrowLeft","ArrowRight"].includes(s.code))wangpGalleryUser=true;switch(s.code){'),
        ('const qe=s=>l(1,m=s);', 'const qe=s=>{wangpGalleryUser=true;return l(1,m=s)};'),
        ('Oe=s=>{m===null', 'Oe=s=>{wangpGalleryUser=true;m===null'),
        ('Ge=()=>{l(1,m=null)', 'Ge=()=>{wangpGalleryUser=true;l(1,m=null)'),
        ('n.$$set=s=>{"show_label"', 'n.$$set=s=>{wangpGalleryExplicit||="selected_index"in s&&("value"in s||s.selected_index!==m);if(wangpGalleryUser&&(("selected_index"in s&&s.selected_index!==m)||("value"in s&&!et(r,s.value))))wangpGalleryUser=false;"show_label"'),
        ('K?(l(1,m=a&&r?.length?0:null),l(29,K=!1))', 'K?(!wangpGalleryExplicit&&l(1,m=a&&r?.length?0:null),l(29,K=r==null||r.length===0))'),
        ('J("change"),l(30,le=r)', 'wangpGalleryChanged=true,l(30,le=r)'),
        ('(l(31,ne=m),m!==null&&(P!=null&&l(1,m=Math.max(0,Math.min(m,P.length-1))),J("select",{index:m,value:P?.[m]})))', '(m!==null&&P!=null&&l(1,m=Math.max(0,Math.min(m,P.length-1))),l(31,ne=m),wangpGalleryUser?(m!==null&&J("select",{index:m,value:P?.[m]})):wangpGalleryChanged=true)'),
        ('l(22,i=m!=null&&P!=null?P[m]:null)},[r,m', 'l(22,i=m!=null&&P!=null?P[m]:null);if(wangpGalleryChanged)J("change");wangpGalleryChanged=wangpGalleryUser=wangpGalleryExplicit=false},[r,m'),
    ],
    'index.js': [
        ('QC.SchedulerSystem = JC;', _EDITOR_SCHEDULER + 'QC.SchedulerSystem = JC;'),
        ('this._pauseUpdate = e;', 'this._pauseUpdate = e; e ? this.removeTickerListener() : this.addTickerListener();'),
        ('this._tickerAdded || !this.domElement ||', 'this._pauseUpdate || this._tickerAdded || !this.domElement ||'),
    ],
    'Blocks-BMC4HgbM.js': [
        # Hide the API footer fragment (including its divider), not the API.
        ('y=l[5]&&Qi(l);', 'y=false;'),
        ('b[5]?y?y.p(b,q):(y=Qi(b),y.c(),y.m(e,t)):y&&(y.d(1),y=null),', ''),
        ('l[22]("common.built_with_gradio")+""', '"Powered by Gradio-GP"'),
        ('b[22]("common.built_with_gradio")+""', '"Powered by Gradio-GP"'),
        # Reuse the native settings action from the credit link. Omit the
        # separate settings button/divider and the panel's PWA section.
        ('le(n,"href","https://gradio.app")', 'le(n,"href","?view=settings")'),
        ('le(n,"target","_blank"),le(n,"rel","noreferrer"),', ''),
        ('ue(e,_),ue(e,u),ue(e,f),ue(e,p),ue(p,h),ue(p,$),ue(p,m),', ''),
        ('v=Nn(p,"click",l[44])', 'v=Nn(n,"click",event=>{event.preventDefault();l[44]()})'),
        ('Je(q,f,j),Je(q,p,j),ve(p,g),ve(g,$),ve(p,m),ve(p,w),b.m(w,null),', ''),
        (_STATUS_ORIGINAL, _STATUS_UPDATE),
        # Re-publish after a component clears its indicator or a layout rebuild.
        ('function Ao(S,J,K){', 'function Ao(S,J,K){wangpStatusCache.delete(String(S));'),
        ('function Bo(ae){', 'function Bo(ae){wangpStatusCache.clear();wangpPendingCache.clear();'),
        # Own the wrapper's props once instead of copying the accumulated object
        # on every $set. Rest props are still freshly derived by Svelte's ji().
        ('function $u(l,e,t){', 'function $u(l,e,t){e=el({},e);'),
        ('e=el(el({},e),au(b)),t(9,s=ji(e,n))', 'el(e,au(b)),t(9,s=ji(e,n))'),
        ('function Pu(l,e,t){', 'function Pu(l,e,t){' + _GRADIO_CONTEXT + 'let wangpNodeVersion=e.node.__wangp_revision,wangpNodeInputs={...e};'),
        ('new er(s.id,o,a,c,n,r,_,eu,u,Yo)', 'wangpGradio(s.id,o,a,c,n,r,_,eu,u,Yo)'),
        ('return l.$$set=w=>{' + _NODE_INPUTS, 'return l.$$set=w=>{' + _NODE_SKIP + _NODE_INPUTS),
    ],
    'index-Do3LSwBC.js': [
        ('function q(){l.update(k=>{for(let g=0;', 'function q(){l.update(k=>{const wangpDirty=new Set;for(let g=0;'),
        ('f.props[v.prop]=j}return k}),ge=[]', 'f.props[v.prop]=j;' + _MARK_ANCESTORS + 'return k}),ge=[]'),
        ('l.set(h)', 'Object.values(s).forEach(node=>node.__wangp_revision=(node.__wangp_revision||0)+1),l.set(h)'),
    ],
}


@lru_cache(maxsize=len(_PATCHES))
def _asset(path):
    source = Path(path).read_text(encoding='utf-8')
    name = Path(path).name
    replacements = list(_QUEUE_REPLACEMENTS.items()) if name.startswith('Blocks-') else []
    replacements += _PATCHES[name]
    for old, new in replacements:
        expected = 2 if old == 'l.set(h)' else 1
        if source.count(old) != expected:
            raise RuntimeError('Gradio frontend patches require the pinned Gradio 5.29 and WanGP editor assets')
        source = source.replace(old, new)
    return source


def install():
    original = routes.FileResponse
    if getattr(original, '_wangp_frontend', False):
        return
    # Validate the pinned assets at startup rather than fail a browser import.
    asset_paths = {_EDITOR_PATH if name == 'index.js' else Path(routes.BUILD_PATH_LIB) / name for name in _PATCHES}
    for path in asset_paths:
        _asset(str(path))

    @wraps(original)
    def file_response(path, *args, **kwargs):
        if Path(path) in asset_paths:
            return Response(_asset(str(path)), media_type='application/javascript', headers={'Cache-Control': 'no-store'})
        return original(path, *args, **kwargs)

    file_response._wangp_frontend = True
    routes.FileResponse = file_response
