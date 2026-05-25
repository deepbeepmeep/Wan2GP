const {
  SvelteComponent: Kt,
  append: E,
  attr: S,
  check_outros: ft,
  create_component: Qt,
  destroy_block: Wt,
  destroy_component: Jt,
  detach: se,
  element: fe,
  empty: ht,
  ensure_array_like: be,
  flush: Z,
  group_outros: dt,
  init: Ut,
  insert: ae,
  listen: Ie,
  mount_component: Xt,
  outro_and_destroy_block: Yt,
  run_all: _t,
  safe_not_equal: $t,
  set_data: mt,
  set_style: Fe,
  space: Y,
  svg_element: Q,
  text: gt,
  toggle_class: Se,
  transition_in: he,
  transition_out: Me,
  update_keyed_each: Qe
} = window.__gradio__svelte__internal;
function We(l, e, t) {
  const n = l.slice();
  n[13] = e[t];
  const i = (
    /*value*/
    n[4].includes(
      /*valueForItem*/
      n[7](
        /*item*/
        n[13]
      )
    )
  );
  return n[14] = i, n;
}
function Je(l, e, t) {
  const n = l.slice();
  n[17] = e[t];
  const i = pt(
    /*folder*/
    n[17]
  );
  return n[18] = i, n;
}
function Ue(l) {
  let e, t;
  return e = new vt({
    props: {
      folders: (
        /*folder*/
        l[17].folders || []
      ),
      items: (
        /*folder*/
        l[17].items || []
      ),
      depth: (
        /*depth*/
        l[2] + 1
      ),
      expanded: (
        /*expanded*/
        l[3]
      ),
      value: (
        /*value*/
        l[4]
      ),
      toggleItem: (
        /*toggleItem*/
        l[5]
      ),
      toggleFolder: (
        /*toggleFolder*/
        l[6]
      ),
      valueForItem: (
        /*valueForItem*/
        l[7]
      ),
      labelForItem: (
        /*labelForItem*/
        l[8]
      )
    }
  }), {
    c() {
      Qt(e.$$.fragment);
    },
    m(n, i) {
      Xt(e, n, i), t = !0;
    },
    p(n, i) {
      const s = {};
      i & /*folders*/
      1 && (s.folders = /*folder*/
      n[17].folders || []), i & /*folders*/
      1 && (s.items = /*folder*/
      n[17].items || []), i & /*depth*/
      4 && (s.depth = /*depth*/
      n[2] + 1), i & /*expanded*/
      8 && (s.expanded = /*expanded*/
      n[3]), i & /*value*/
      16 && (s.value = /*value*/
      n[4]), i & /*toggleItem*/
      32 && (s.toggleItem = /*toggleItem*/
      n[5]), i & /*toggleFolder*/
      64 && (s.toggleFolder = /*toggleFolder*/
      n[6]), i & /*valueForItem*/
      128 && (s.valueForItem = /*valueForItem*/
      n[7]), i & /*labelForItem*/
      256 && (s.labelForItem = /*labelForItem*/
      n[8]), e.$set(s);
    },
    i(n) {
      t || (he(e.$$.fragment, n), t = !0);
    },
    o(n) {
      Me(e.$$.fragment, n), t = !1;
    },
    d(n) {
      Jt(e, n);
    }
  };
}
function Xe(l, e) {
  let t, n, i, s, a, o, _, k, m, w = we(
    /*folder*/
    e[17]
  ) + "", u, c, p, z = (
    /*expanded*/
    e[3].has(
      /*path*/
      e[18]
    )
  ), h, I, B, v;
  function H() {
    return (
      /*click_handler*/
      e[9](
        /*path*/
        e[18]
      )
    );
  }
  function C(...b) {
    return (
      /*keydown_handler*/
      e[10](
        /*path*/
        e[18],
        ...b
      )
    );
  }
  let g = z && Ue(e);
  return {
    key: l,
    first: null,
    c() {
      t = fe("div"), n = Q("svg"), i = Q("path"), s = Y(), a = Q("svg"), o = Q("path"), _ = Q("path"), k = Y(), m = fe("span"), u = gt(w), p = Y(), g && g.c(), h = ht(), S(i, "d", "M6 4.5L10 8l-4 3.5"), S(n, "class", "hierarchy-twist svelte-1p05czg"), S(n, "viewBox", "0 0 16 16"), S(n, "aria-hidden", "true"), Se(
        n,
        "hierarchy-twist-open",
        /*expanded*/
        e[3].has(
          /*path*/
          e[18]
        )
      ), S(o, "d", "M2.75 6.25h5.4l1.55 1.7h7.55c.55 0 1 .45 1 1v6.3c0 .55-.45 1-1 1H2.75c-.55 0-1-.45-1-1v-8c0-.55.45-1 1-1Z"), S(_, "d", "M2.25 7.95V5.6c0-.55.45-1 1-1h4.5l1.35 1.65"), S(a, "class", "hierarchy-icon hierarchy-folder-icon svelte-1p05czg"), S(a, "viewBox", "0 0 20 20"), S(a, "aria-hidden", "true"), S(m, "class", "hierarchy-name svelte-1p05czg"), S(t, "class", "hierarchy-row hierarchy-folder svelte-1p05czg"), S(t, "role", "button"), S(t, "tabindex", "0"), S(t, "title", c = we(
        /*folder*/
        e[17]
      )), Fe(t, "padding-left", `${/*depth*/
      e[2] * 18 + 6}px`), this.first = t;
    },
    m(b, f) {
      ae(b, t, f), E(t, n), E(n, i), E(t, s), E(t, a), E(a, o), E(a, _), E(t, k), E(t, m), E(m, u), ae(b, p, f), g && g.m(b, f), ae(b, h, f), I = !0, B || (v = [
        Ie(t, "click", H),
        Ie(t, "keydown", C)
      ], B = !0);
    },
    p(b, f) {
      e = b, (!I || f & /*expanded, folderPath, folders*/
      9) && Se(
        n,
        "hierarchy-twist-open",
        /*expanded*/
        e[3].has(
          /*path*/
          e[18]
        )
      ), (!I || f & /*folders*/
      1) && w !== (w = we(
        /*folder*/
        e[17]
      ) + "") && mt(u, w), (!I || f & /*folders*/
      1 && c !== (c = we(
        /*folder*/
        e[17]
      ))) && S(t, "title", c), f & /*depth*/
      4 && Fe(t, "padding-left", `${/*depth*/
      e[2] * 18 + 6}px`), f & /*expanded, folders*/
      9 && (z = /*expanded*/
      e[3].has(
        /*path*/
        e[18]
      )), z ? g ? (g.p(e, f), f & /*expanded, folders*/
      9 && he(g, 1)) : (g = Ue(e), g.c(), he(g, 1), g.m(h.parentNode, h)) : g && (dt(), Me(g, 1, 1, () => {
        g = null;
      }), ft());
    },
    i(b) {
      I || (he(g), I = !0);
    },
    o(b) {
      Me(g), I = !1;
    },
    d(b) {
      b && (se(t), se(p), se(h)), g && g.d(b), B = !1, _t(v);
    }
  };
}
function Ye(l, e) {
  let t, n, i, s, a, o, _, k, m = (
    /*labelForItem*/
    e[8](
      /*item*/
      e[13]
    ) + ""
  ), w, u, c, p, z, h;
  function I() {
    return (
      /*click_handler_1*/
      e[11](
        /*item*/
        e[13]
      )
    );
  }
  function B(...v) {
    return (
      /*keydown_handler_1*/
      e[12](
        /*item*/
        e[13],
        ...v
      )
    );
  }
  return {
    key: l,
    first: null,
    c() {
      t = fe("div"), n = fe("span"), i = Y(), s = Q("svg"), a = Q("path"), o = Q("path"), _ = Y(), k = fe("span"), w = gt(m), u = Y(), S(n, "class", "hierarchy-twist-spacer svelte-1p05czg"), S(a, "d", "M5.25 2.75h6.05L15.75 7.2v10.05H5.25V2.75Z"), S(o, "d", "M11.25 2.95V7.3h4.3"), S(s, "class", "hierarchy-icon hierarchy-item-icon svelte-1p05czg"), S(s, "viewBox", "0 0 20 20"), S(s, "aria-hidden", "true"), S(k, "class", "hierarchy-name svelte-1p05czg"), S(t, "class", "hierarchy-row hierarchy-item svelte-1p05czg"), S(t, "title", c = /*labelForItem*/
      e[8](
        /*item*/
        e[13]
      )), S(t, "role", "button"), S(t, "tabindex", "0"), S(t, "aria-pressed", p = /*selected*/
      e[14]), Se(
        t,
        "hierarchy-item-selected",
        /*selected*/
        e[14]
      ), Fe(t, "padding-left", `${/*depth*/
      e[2] * 18 + 6}px`), this.first = t;
    },
    m(v, H) {
      ae(v, t, H), E(t, n), E(t, i), E(t, s), E(s, a), E(s, o), E(t, _), E(t, k), E(k, w), E(t, u), z || (h = [
        Ie(t, "click", I),
        Ie(t, "keydown", B)
      ], z = !0);
    },
    p(v, H) {
      e = v, H & /*labelForItem, items*/
      258 && m !== (m = /*labelForItem*/
      e[8](
        /*item*/
        e[13]
      ) + "") && mt(w, m), H & /*labelForItem, items*/
      258 && c !== (c = /*labelForItem*/
      e[8](
        /*item*/
        e[13]
      )) && S(t, "title", c), H & /*value, valueForItem, items*/
      146 && p !== (p = /*selected*/
      e[14]) && S(t, "aria-pressed", p), H & /*value, valueForItem, items*/
      146 && Se(
        t,
        "hierarchy-item-selected",
        /*selected*/
        e[14]
      ), H & /*depth*/
      4 && Fe(t, "padding-left", `${/*depth*/
      e[2] * 18 + 6}px`);
    },
    d(v) {
      v && se(t), z = !1, _t(h);
    }
  };
}
function xt(l) {
  let e = [], t = /* @__PURE__ */ new Map(), n, i = [], s = /* @__PURE__ */ new Map(), a, o, _ = be(
    /*folders*/
    l[0]
  );
  const k = (u) => pt(
    /*folder*/
    u[17]
  );
  for (let u = 0; u < _.length; u += 1) {
    let c = Je(l, _, u), p = k(c);
    t.set(p, e[u] = Xe(p, c));
  }
  let m = be(
    /*items*/
    l[1]
  );
  const w = (u) => (
    /*valueForItem*/
    u[7](
      /*item*/
      u[13]
    )
  );
  for (let u = 0; u < m.length; u += 1) {
    let c = We(l, m, u), p = w(c);
    s.set(p, i[u] = Ye(p, c));
  }
  return {
    c() {
      for (let u = 0; u < e.length; u += 1)
        e[u].c();
      n = Y();
      for (let u = 0; u < i.length; u += 1)
        i[u].c();
      a = ht();
    },
    m(u, c) {
      for (let p = 0; p < e.length; p += 1)
        e[p] && e[p].m(u, c);
      ae(u, n, c);
      for (let p = 0; p < i.length; p += 1)
        i[p] && i[p].m(u, c);
      ae(u, a, c), o = !0;
    },
    p(u, [c]) {
      c & /*folders, depth, expanded, value, toggleItem, toggleFolder, valueForItem, labelForItem, folderPath, folderLabel*/
      509 && (_ = be(
        /*folders*/
        u[0]
      ), dt(), e = Qe(e, c, k, 1, u, _, t, n.parentNode, Yt, Xe, n, Je), ft()), c & /*labelForItem, items, value, valueForItem, depth, toggleItem*/
      438 && (m = be(
        /*items*/
        u[1]
      ), i = Qe(i, c, w, 1, u, m, s, a.parentNode, Wt, Ye, a, We));
    },
    i(u) {
      if (!o) {
        for (let c = 0; c < _.length; c += 1)
          he(e[c]);
        o = !0;
      }
    },
    o(u) {
      for (let c = 0; c < e.length; c += 1)
        Me(e[c]);
      o = !1;
    },
    d(u) {
      u && (se(n), se(a));
      for (let c = 0; c < e.length; c += 1)
        e[c].d(u);
      for (let c = 0; c < i.length; c += 1)
        i[c].d(u);
    }
  };
}
function we(l) {
  return String(l.name || l.path || "");
}
function pt(l) {
  return String(l.path || l.name || "");
}
function el(l, e, t) {
  let { folders: n = [] } = e, { items: i = [] } = e, { depth: s = 0 } = e, { expanded: a } = e, { value: o } = e, { toggleItem: _ } = e, { toggleFolder: k } = e, { valueForItem: m } = e, { labelForItem: w } = e;
  const u = (h) => k(h), c = (h, I) => {
    (I.key === "Enter" || I.key === " ") && (I.preventDefault(), k(h));
  }, p = (h) => _(h), z = (h, I) => {
    (I.key === "Enter" || I.key === " ") && (I.preventDefault(), _(h));
  };
  return l.$$set = (h) => {
    "folders" in h && t(0, n = h.folders), "items" in h && t(1, i = h.items), "depth" in h && t(2, s = h.depth), "expanded" in h && t(3, a = h.expanded), "value" in h && t(4, o = h.value), "toggleItem" in h && t(5, _ = h.toggleItem), "toggleFolder" in h && t(6, k = h.toggleFolder), "valueForItem" in h && t(7, m = h.valueForItem), "labelForItem" in h && t(8, w = h.labelForItem);
  }, [
    n,
    i,
    s,
    a,
    o,
    _,
    k,
    m,
    w,
    u,
    c,
    p,
    z
  ];
}
class vt extends Kt {
  constructor(e) {
    super(), Ut(this, e, el, xt, $t, {
      folders: 0,
      items: 1,
      depth: 2,
      expanded: 3,
      value: 4,
      toggleItem: 5,
      toggleFolder: 6,
      valueForItem: 7,
      labelForItem: 8
    });
  }
  get folders() {
    return this.$$.ctx[0];
  }
  set folders(e) {
    this.$$set({ folders: e }), Z();
  }
  get items() {
    return this.$$.ctx[1];
  }
  set items(e) {
    this.$$set({ items: e }), Z();
  }
  get depth() {
    return this.$$.ctx[2];
  }
  set depth(e) {
    this.$$set({ depth: e }), Z();
  }
  get expanded() {
    return this.$$.ctx[3];
  }
  set expanded(e) {
    this.$$set({ expanded: e }), Z();
  }
  get value() {
    return this.$$.ctx[4];
  }
  set value(e) {
    this.$$set({ value: e }), Z();
  }
  get toggleItem() {
    return this.$$.ctx[5];
  }
  set toggleItem(e) {
    this.$$set({ toggleItem: e }), Z();
  }
  get toggleFolder() {
    return this.$$.ctx[6];
  }
  set toggleFolder(e) {
    this.$$set({ toggleFolder: e }), Z();
  }
  get valueForItem() {
    return this.$$.ctx[7];
  }
  set valueForItem(e) {
    this.$$set({ valueForItem: e }), Z();
  }
  get labelForItem() {
    return this.$$.ctx[8];
  }
  set labelForItem(e) {
    this.$$set({ labelForItem: e }), Z();
  }
}
const {
  SvelteComponent: tl,
  action_destroyer: ll,
  append: M,
  attr: d,
  binding_callbacks: ke,
  check_outros: Ve,
  create_component: nl,
  destroy_block: il,
  destroy_component: rl,
  destroy_each: sl,
  detach: R,
  element: V,
  empty: Be,
  ensure_array_like: De,
  flush: N,
  group_outros: Ne,
  init: al,
  insert: q,
  listen: A,
  mount_component: ol,
  noop: Pe,
  null_to_empty: $e,
  run_all: Ae,
  safe_not_equal: ul,
  set_data: de,
  set_input_value: xe,
  space: j,
  stop_propagation: bt,
  svg_element: ze,
  text: $,
  toggle_class: W,
  transition_in: G,
  transition_out: x,
  update_keyed_each: cl
} = window.__gradio__svelte__internal, { onDestroy: fl, onMount: hl, tick: Ce } = window.__gradio__svelte__internal;
function et(l, e, t) {
  const n = l.slice();
  n[67] = e[t];
  const i = (
    /*selectedValue*/
    n[8].includes(_e(
      /*item*/
      n[67]
    ))
  );
  return n[68] = i, n;
}
function tt(l, e, t) {
  const n = l.slice();
  return n[68] = e[t], n[72] = t, n;
}
function lt(l) {
  let e, t, n, i, s, a, o, _, k, m, w, u, c, p, z, h, I, B, v = (
    /*show_label*/
    l[4] && /*label*/
    l[2] && nt(l)
  ), H = De(
    /*selectedValue*/
    l[8]
  ), C = [];
  for (let f = 0; f < H.length; f += 1)
    C[f] = it(tt(l, H, f));
  let g = (
    /*open*/
    l[6] && rt(l)
  ), b = (
    /*info*/
    l[3] && ot(l)
  );
  return {
    c() {
      e = V("div"), t = V("div"), v && v.c(), n = j(), i = V("div"), s = V("div");
      for (let f = 0; f < C.length; f += 1)
        C[f].c();
      a = j(), o = V("input"), c = j(), g && g.c(), p = j(), b && b.c(), d(o, "class", "hierarchy-selector-search-input svelte-n293wu"), d(o, "type", "text"), d(o, "autocomplete", "off"), d(o, "spellcheck", "false"), o.disabled = _ = !/*interactive*/
      l[5], d(o, "tabindex", k = /*interactive*/
      l[5] ? 0 : -1), d(o, "placeholder", m = /*selectedValue*/
      l[8].length === 0 ? (
        /*label*/
        l[2]
      ) : ""), d(
        o,
        "aria-label",
        /*label*/
        l[2]
      ), d(s, "class", "hierarchy-selector-chips svelte-n293wu"), d(i, "class", "hierarchy-selector-input svelte-n293wu"), d(i, "role", "combobox"), d(i, "tabindex", w = /*interactive*/
      l[5] ? 0 : -1), d(i, "aria-haspopup", "tree"), d(i, "aria-expanded", u = /*open*/
      l[6] ? "true" : "false"), d(
        i,
        "aria-controls",
        /*panelId*/
        l[22]
      ), W(i, "hierarchy-selector-disabled", !/*interactive*/
      l[5]), d(t, "class", "hierarchy-selector-field svelte-n293wu"), d(
        e,
        "id",
        /*elem_id*/
        l[0]
      ), d(e, "class", z = $e(
        /*classes*/
        l[21]
      ) + " svelte-n293wu"), d(
        e,
        "style",
        /*style*/
        l[20]
      );
    },
    m(f, D) {
      q(f, e, D), M(e, t), v && v.m(t, null), M(t, n), M(t, i), M(i, s);
      for (let L = 0; L < C.length; L += 1)
        C[L] && C[L].m(s, null);
      M(s, a), M(s, o), l[52](o), xe(
        o,
        /*searchQuery*/
        l[7]
      ), l[54](i), M(t, c), g && g.m(t, null), M(e, p), b && b.m(e, null), l[58](e), h = !0, I || (B = [
        A(
          o,
          "input",
          /*input_input_handler*/
          l[53]
        ),
        A(
          o,
          "focus",
          /*onSearchFocus*/
          l[27]
        ),
        A(
          o,
          "input",
          /*onSearchInput*/
          l[28]
        ),
        A(o, "keydown", bt(
          /*onInputKeydown*/
          l[33]
        )),
        A(
          i,
          "mousedown",
          /*onInputPointerDown*/
          l[29]
        ),
        A(
          i,
          "keydown",
          /*onInputKeydown*/
          l[33]
        )
      ], I = !0);
    },
    p(f, D) {
      if (/*show_label*/
      f[4] && /*label*/
      f[2] ? v ? v.p(f, D) : (v = nt(f), v.c(), v.m(t, n)) : v && (v.d(1), v = null), D[0] & /*interactive, draggedIndex, dragOverIndex, onDragStart, removeValue, displayValue, selectedValue*/
      1115881760 | D[1] & /*onDragEnd, onDrop*/
      3) {
        H = De(
          /*selectedValue*/
          f[8]
        );
        let L;
        for (L = 0; L < H.length; L += 1) {
          const F = tt(f, H, L);
          C[L] ? C[L].p(F, D) : (C[L] = it(F), C[L].c(), C[L].m(s, a));
        }
        for (; L < C.length; L += 1)
          C[L].d(1);
        C.length = H.length;
      }
      (!h || D[0] & /*interactive*/
      32 && _ !== (_ = !/*interactive*/
      f[5])) && (o.disabled = _), (!h || D[0] & /*interactive*/
      32 && k !== (k = /*interactive*/
      f[5] ? 0 : -1)) && d(o, "tabindex", k), (!h || D[0] & /*selectedValue, label*/
      260 && m !== (m = /*selectedValue*/
      f[8].length === 0 ? (
        /*label*/
        f[2]
      ) : "")) && d(o, "placeholder", m), (!h || D[0] & /*label*/
      4) && d(
        o,
        "aria-label",
        /*label*/
        f[2]
      ), D[0] & /*searchQuery*/
      128 && o.value !== /*searchQuery*/
      f[7] && xe(
        o,
        /*searchQuery*/
        f[7]
      ), (!h || D[0] & /*interactive*/
      32 && w !== (w = /*interactive*/
      f[5] ? 0 : -1)) && d(i, "tabindex", w), (!h || D[0] & /*open*/
      64 && u !== (u = /*open*/
      f[6] ? "true" : "false")) && d(i, "aria-expanded", u), (!h || D[0] & /*panelId*/
      4194304) && d(
        i,
        "aria-controls",
        /*panelId*/
        f[22]
      ), (!h || D[0] & /*interactive*/
      32) && W(i, "hierarchy-selector-disabled", !/*interactive*/
      f[5]), /*open*/
      f[6] ? g ? (g.p(f, D), D[0] & /*open*/
      64 && G(g, 1)) : (g = rt(f), g.c(), G(g, 1), g.m(t, null)) : g && (Ne(), x(g, 1, 1, () => {
        g = null;
      }), Ve()), /*info*/
      f[3] ? b ? b.p(f, D) : (b = ot(f), b.c(), b.m(e, null)) : b && (b.d(1), b = null), (!h || D[0] & /*elem_id*/
      1) && d(
        e,
        "id",
        /*elem_id*/
        f[0]
      ), (!h || D[0] & /*classes*/
      2097152 && z !== (z = $e(
        /*classes*/
        f[21]
      ) + " svelte-n293wu")) && d(e, "class", z), (!h || D[0] & /*style*/
      1048576) && d(
        e,
        "style",
        /*style*/
        f[20]
      );
    },
    i(f) {
      h || (G(g), h = !0);
    },
    o(f) {
      x(g), h = !1;
    },
    d(f) {
      f && R(e), v && v.d(), sl(C, f), l[52](null), l[54](null), g && g.d(), b && b.d(), l[58](null), I = !1, Ae(B);
    }
  };
}
function nt(l) {
  let e, t;
  return {
    c() {
      e = V("span"), t = $(
        /*label*/
        l[2]
      ), d(e, "class", "hierarchy-selector-label svelte-n293wu");
    },
    m(n, i) {
      q(n, e, i), M(e, t);
    },
    p(n, i) {
      i[0] & /*label*/
      4 && de(
        t,
        /*label*/
        n[2]
      );
    },
    d(n) {
      n && R(e);
    }
  };
}
function it(l) {
  let e, t, n = (
    /*displayValue*/
    l[23](
      /*selected*/
      l[68]
    ) + ""
  ), i, s, a, o, _;
  function k() {
    return (
      /*click_handler*/
      l[47](
        /*index*/
        l[72]
      )
    );
  }
  function m(...c) {
    return (
      /*dragstart_handler*/
      l[48](
        /*index*/
        l[72],
        ...c
      )
    );
  }
  function w(...c) {
    return (
      /*dragover_handler*/
      l[49](
        /*index*/
        l[72],
        ...c
      )
    );
  }
  function u(...c) {
    return (
      /*drop_handler*/
      l[51](
        /*index*/
        l[72],
        ...c
      )
    );
  }
  return {
    c() {
      e = V("span"), t = V("span"), i = $(n), s = j(), a = V("button"), a.textContent = "x", d(t, "class", "hierarchy-selector-chip-text svelte-n293wu"), d(a, "type", "button"), d(a, "class", "hierarchy-selector-remove svelte-n293wu"), d(a, "aria-label", "Remove"), d(e, "class", "hierarchy-selector-chip svelte-n293wu"), d(e, "role", "listitem"), d(
        e,
        "draggable",
        /*interactive*/
        l[5]
      ), W(
        e,
        "hierarchy-selector-chip-dragging",
        /*draggedIndex*/
        l[16] === /*index*/
        l[72]
      ), W(
        e,
        "hierarchy-selector-chip-over",
        /*dragOverIndex*/
        l[17] === /*index*/
        l[72]
      );
    },
    m(c, p) {
      q(c, e, p), M(e, t), M(t, i), M(e, s), M(e, a), o || (_ = [
        A(a, "click", bt(k)),
        A(e, "dragstart", m),
        A(
          e,
          "dragend",
          /*onDragEnd*/
          l[31]
        ),
        A(e, "dragover", w),
        A(
          e,
          "dragleave",
          /*dragleave_handler*/
          l[50]
        ),
        A(e, "drop", u)
      ], o = !0);
    },
    p(c, p) {
      l = c, p[0] & /*selectedValue*/
      256 && n !== (n = /*displayValue*/
      l[23](
        /*selected*/
        l[68]
      ) + "") && de(i, n), p[0] & /*interactive*/
      32 && d(
        e,
        "draggable",
        /*interactive*/
        l[5]
      ), p[0] & /*draggedIndex*/
      65536 && W(
        e,
        "hierarchy-selector-chip-dragging",
        /*draggedIndex*/
        l[16] === /*index*/
        l[72]
      ), p[0] & /*dragOverIndex*/
      131072 && W(
        e,
        "hierarchy-selector-chip-over",
        /*dragOverIndex*/
        l[17] === /*index*/
        l[72]
      );
    },
    d(c) {
      c && R(e), o = !1, Ae(_);
    }
  };
}
function rt(l) {
  let e, t, n, i, s, a;
  const o = [_l, dl], _ = [];
  function k(m, w) {
    return (
      /*searchMode*/
      m[9] ? 0 : 1
    );
  }
  return t = k(l), n = _[t] = o[t](l), {
    c() {
      e = V("div"), n.c(), d(
        e,
        "id",
        /*panelId*/
        l[22]
      ), d(e, "class", "hierarchy-selector-panel svelte-n293wu"), d(
        e,
        "style",
        /*panelStyle*/
        l[18]
      );
    },
    m(m, w) {
      q(m, e, w), _[t].m(e, null), l[57](e), i = !0, s || (a = ll(Fl.call(null, e)), s = !0);
    },
    p(m, w) {
      let u = t;
      t = k(m), t === u ? _[t].p(m, w) : (Ne(), x(_[u], 1, 1, () => {
        _[u] = null;
      }), Ve(), n = _[t], n ? n.p(m, w) : (n = _[t] = o[t](m), n.c()), G(n, 1), n.m(e, null)), (!i || w[0] & /*panelId*/
      4194304) && d(
        e,
        "id",
        /*panelId*/
        m[22]
      ), (!i || w[0] & /*panelStyle*/
      262144) && d(
        e,
        "style",
        /*panelStyle*/
        m[18]
      );
    },
    i(m) {
      i || (G(n), i = !0);
    },
    o(m) {
      x(n), i = !1;
    },
    d(m) {
      m && R(e), _[t].d(), l[57](null), s = !1, a();
    }
  };
}
function dl(l) {
  let e, t;
  return e = new vt({
    props: {
      folders: (
        /*normalizedHierarchy*/
        l[10].folders || []
      ),
      items: (
        /*normalizedHierarchy*/
        l[10].items || []
      ),
      depth: 0,
      expanded: (
        /*expanded*/
        l[15]
      ),
      value: (
        /*selectedValue*/
        l[8]
      ),
      toggleItem: (
        /*toggleItem*/
        l[24]
      ),
      toggleFolder: (
        /*toggleFolder*/
        l[26]
      ),
      valueForItem: _e,
      labelForItem: ee
    }
  }), {
    c() {
      nl(e.$$.fragment);
    },
    m(n, i) {
      ol(e, n, i), t = !0;
    },
    p(n, i) {
      const s = {};
      i[0] & /*normalizedHierarchy*/
      1024 && (s.folders = /*normalizedHierarchy*/
      n[10].folders || []), i[0] & /*normalizedHierarchy*/
      1024 && (s.items = /*normalizedHierarchy*/
      n[10].items || []), i[0] & /*expanded*/
      32768 && (s.expanded = /*expanded*/
      n[15]), i[0] & /*selectedValue*/
      256 && (s.value = /*selectedValue*/
      n[8]), e.$set(s);
    },
    i(n) {
      t || (G(e.$$.fragment, n), t = !0);
    },
    o(n) {
      x(e.$$.fragment, n), t = !1;
    },
    d(n) {
      rl(e, n);
    }
  };
}
function _l(l) {
  let e;
  function t(s, a) {
    return (
      /*searchResults*/
      s[19].length ? gl : ml
    );
  }
  let n = t(l), i = n(l);
  return {
    c() {
      i.c(), e = Be();
    },
    m(s, a) {
      i.m(s, a), q(s, e, a);
    },
    p(s, a) {
      n === (n = t(s)) && i ? i.p(s, a) : (i.d(1), i = n(s), i && (i.c(), i.m(e.parentNode, e)));
    },
    i: Pe,
    o: Pe,
    d(s) {
      s && R(e), i.d(s);
    }
  };
}
function ml(l) {
  let e;
  return {
    c() {
      e = V("div"), e.textContent = "No matching LoRAs", d(e, "class", "hierarchy-search-empty svelte-n293wu");
    },
    m(t, n) {
      q(t, e, n);
    },
    p: Pe,
    d(t) {
      t && R(e);
    }
  };
}
function gl(l) {
  let e = [], t = /* @__PURE__ */ new Map(), n, i = De(
    /*searchResults*/
    l[19]
  );
  const s = (a) => _e(
    /*item*/
    a[67]
  );
  for (let a = 0; a < i.length; a += 1) {
    let o = et(l, i, a), _ = s(o);
    t.set(_, e[a] = at(_, o));
  }
  return {
    c() {
      for (let a = 0; a < e.length; a += 1)
        e[a].c();
      n = Be();
    },
    m(a, o) {
      for (let _ = 0; _ < e.length; _ += 1)
        e[_] && e[_].m(a, o);
      q(a, n, o);
    },
    p(a, o) {
      o[0] & /*searchResults, selectedValue, toggleItem*/
      17301760 && (i = De(
        /*searchResults*/
        a[19]
      ), e = cl(e, o, s, 1, a, i, t, n.parentNode, il, at, n, et));
    },
    d(a) {
      a && R(n);
      for (let o = 0; o < e.length; o += 1)
        e[o].d(a);
    }
  };
}
function st(l) {
  let e, t, n = (
    /*item*/
    l[67].search_path + ""
  ), i, s;
  return {
    c() {
      e = V("span"), t = $("["), i = $(n), s = $("]"), d(e, "class", "hierarchy-search-path svelte-n293wu");
    },
    m(a, o) {
      q(a, e, o), M(e, t), M(e, i), M(e, s);
    },
    p(a, o) {
      o[0] & /*searchResults*/
      524288 && n !== (n = /*item*/
      a[67].search_path + "") && de(i, n);
    },
    d(a) {
      a && R(e);
    }
  };
}
function at(l, e) {
  let t, n, i, s, a, o, _, k, m, w = ee(
    /*item*/
    e[67]
  ) + "", u, c, p, z, h, I, B, v = (
    /*item*/
    e[67].search_path && st(e)
  );
  function H() {
    return (
      /*click_handler_1*/
      e[55](
        /*item*/
        e[67]
      )
    );
  }
  function C(...g) {
    return (
      /*keydown_handler*/
      e[56](
        /*item*/
        e[67],
        ...g
      )
    );
  }
  return {
    key: l,
    first: null,
    c() {
      t = V("div"), n = V("span"), i = j(), s = ze("svg"), a = ze("path"), o = ze("path"), _ = j(), k = V("span"), m = V("span"), u = $(w), c = j(), v && v.c(), p = j(), d(n, "class", "hierarchy-search-spacer svelte-n293wu"), d(a, "d", "M5.25 2.75h6.05L15.75 7.2v10.05H5.25V2.75Z"), d(o, "d", "M11.25 2.95V7.3h4.3"), d(s, "class", "hierarchy-search-icon svelte-n293wu"), d(s, "viewBox", "0 0 20 20"), d(s, "aria-hidden", "true"), d(m, "class", "hierarchy-search-name svelte-n293wu"), d(k, "class", "hierarchy-search-label svelte-n293wu"), d(t, "class", "hierarchy-search-row svelte-n293wu"), d(t, "title", z = Ee(
        /*item*/
        e[67]
      )), d(t, "role", "button"), d(t, "tabindex", "0"), d(t, "aria-pressed", h = /*selected*/
      e[68]), W(
        t,
        "hierarchy-search-row-selected",
        /*selected*/
        e[68]
      ), this.first = t;
    },
    m(g, b) {
      q(g, t, b), M(t, n), M(t, i), M(t, s), M(s, a), M(s, o), M(t, _), M(t, k), M(k, m), M(m, u), M(k, c), v && v.m(k, null), M(t, p), I || (B = [
        A(t, "click", H),
        A(t, "keydown", C)
      ], I = !0);
    },
    p(g, b) {
      e = g, b[0] & /*searchResults*/
      524288 && w !== (w = ee(
        /*item*/
        e[67]
      ) + "") && de(u, w), /*item*/
      e[67].search_path ? v ? v.p(e, b) : (v = st(e), v.c(), v.m(k, null)) : v && (v.d(1), v = null), b[0] & /*searchResults*/
      524288 && z !== (z = Ee(
        /*item*/
        e[67]
      )) && d(t, "title", z), b[0] & /*selectedValue, searchResults*/
      524544 && h !== (h = /*selected*/
      e[68]) && d(t, "aria-pressed", h), b[0] & /*selectedValue, searchResults*/
      524544 && W(
        t,
        "hierarchy-search-row-selected",
        /*selected*/
        e[68]
      );
    },
    d(g) {
      g && R(t), v && v.d(), I = !1, Ae(B);
    }
  };
}
function ot(l) {
  let e, t;
  return {
    c() {
      e = V("div"), t = $(
        /*info*/
        l[3]
      ), d(e, "class", "hierarchy-selector-info svelte-n293wu");
    },
    m(n, i) {
      q(n, e, i), M(e, t);
    },
    p(n, i) {
      i[0] & /*info*/
      8 && de(
        t,
        /*info*/
        n[3]
      );
    },
    d(n) {
      n && R(e);
    }
  };
}
function pl(l) {
  let e, t, n = (
    /*visible*/
    l[1] && lt(l)
  );
  return {
    c() {
      n && n.c(), e = Be();
    },
    m(i, s) {
      n && n.m(i, s), q(i, e, s), t = !0;
    },
    p(i, s) {
      /*visible*/
      i[1] ? n ? (n.p(i, s), s[0] & /*visible*/
      2 && G(n, 1)) : (n = lt(i), n.c(), G(n, 1), n.m(e.parentNode, e)) : n && (Ne(), x(n, 1, 1, () => {
        n = null;
      }), Ve());
    },
    i(i) {
      t || (G(n), t = !0);
    },
    o(i) {
      x(n), t = !1;
    },
    d(i) {
      i && R(e), n && n.d(i);
    }
  };
}
const ut = 32, vl = 8, ye = 6, ce = 8;
function He(l) {
  return Array.isArray(l) ? l.map((e) => String(e)) : l == null || l === "" ? [] : [String(l)];
}
function bl(l) {
  return {
    folders: wt((l == null ? void 0 : l.folders) || []),
    items: kt((l == null ? void 0 : l.items) || [])
  };
}
function wt(l) {
  return l.map((e) => ({
    ...e,
    folders: wt(e.folders || []),
    items: kt(e.items || [])
  })).sort((e, t) => ct(e).localeCompare(ct(t), void 0, { sensitivity: "base" }));
}
function kt(l) {
  return l.map((e) => ({ ...e })).sort((e, t) => ee(e).localeCompare(ee(t), void 0, { sensitivity: "base" }));
}
function ct(l) {
  return String(l.name || l.path || "");
}
function ee(l) {
  return String(l.name || l.path || l.value || "");
}
function Ee(l) {
  return String(l.path || l.name || l.value || "");
}
function _e(l) {
  return String(l.value || l.path || l.name || "");
}
function wl(l) {
  const e = {};
  function t(n) {
    for (const i of n.items || []) e[_e(i)] = Ee(i);
    for (const i of n.folders || []) t(i);
  }
  return t(l), e;
}
function kl(l) {
  const e = [];
  function t(n, i = "") {
    for (const s of n.items || []) {
      const a = String(s.path || s.name || s.value || ""), o = a.lastIndexOf("/"), _ = o > -1 ? a.slice(0, o) : i;
      e.push({
        ...s,
        search_name: ee(s),
        search_path: _
      });
    }
    for (const s of n.folders || []) t(s, String(s.path || s.name || ""));
  }
  return t(l), e;
}
function yl(l, e) {
  return l.map((t) => {
    const n = String(t.search_name || ee(t)), i = String(t.search_path || "");
    return {
      item: t,
      index: n.toLowerCase().indexOf(e),
      name: n,
      path: i
    };
  }).filter((t) => t.index > -1).sort((t, n) => t.index - n.index || t.name.localeCompare(n.name, void 0, { sensitivity: "base" }) || t.path.localeCompare(n.path, void 0, { sensitivity: "base" })).map((t) => t.item);
}
function Il(l) {
  return l.replace(/\\/g, "/").replace(/\.[^/.]+$/, "");
}
function Fl(l) {
  return document.body.appendChild(l), {
    destroy() {
      l.remove();
    }
  };
}
function Sl(l, e, t) {
  let n, i, s, a, o, _, k, m, w, u, c, { elem_id: p = "" } = e, { elem_classes: z = [] } = e, { visible: h = !0 } = e, { value: I = [] } = e, { hierarchy: B = { folders: [], items: [] } } = e, { height: v = 10 } = e, { label: H = "Hierarchy Selector" } = e, { info: C = void 0 } = e, { show_label: g = !0 } = e, { container: b = !0 } = e, { scale: f = null } = e, { min_width: D = void 0 } = e, { interactive: L = !0 } = e, { gradio: F } = e, te, oe, J, le, ne = !1, ie = !1, re = /* @__PURE__ */ new Set(), O = "", U = null, ue = null, Te = {}, Re = "";
  const yt = `hierarchy-selector-${Math.random().toString(36).slice(2)}`;
  function It(r) {
    return Te[r] || Il(r);
  }
  function me(r, y = "change", P = void 0) {
    var T, K, ve;
    t(34, I = He(r)), (T = F == null ? void 0 : F.dispatch) == null || T.call(F, "input"), y === "select" && ((K = F == null ? void 0 : F.dispatch) == null || K.call(F, "select", P)), (ve = F == null ? void 0 : F.dispatch) == null || ve.call(F, "change"), Ce().then(X);
  }
  function ge(r) {
    if (!L) return;
    const y = _e(r);
    n.includes(y) ? me(n.filter((P) => P !== y), "select", { value: y, selected: !1 }) : me([...n, y], "select", { value: y, selected: !0 });
  }
  function Le(r) {
    if (!L) return;
    const y = n.filter((P, T) => T !== r);
    me(y, "select", {
      value: n[r],
      selected: !1
    });
  }
  function Ft(r) {
    re.has(r) ? re.delete(r) : re.add(r), t(15, re = new Set(re));
  }
  function X() {
    if (!oe || !ne) return;
    const r = oe.getBoundingClientRect(), y = w, P = r.top - ce - ye, T = window.innerHeight - r.bottom - ce - ye, K = P < y && T > P, ve = Math.max(ut * 4, K ? T : P), Ge = Math.min(y, ve), jt = K ? r.bottom + ye : Math.max(ce, r.top - Ge - ye), Ke = Math.max(ce, r.left), Gt = Math.max(240, Math.min(r.width, window.innerWidth - Ke - ce));
    t(18, Re = `top:${Math.round(jt)}px;left:${Math.round(Ke)}px;width:${Math.round(Gt)}px;height:${Math.round(Ge)}px;`);
  }
  function pe() {
    var y;
    if (!L) return;
    const r = ie;
    t(6, ne = !0), t(43, ie = !0), r || (y = F == null ? void 0 : F.dispatch) == null || y.call(F, "focus"), Ce().then(() => {
      J == null || J.focus(), X();
    });
  }
  function qe() {
    var r;
    !ne && !ie || (t(6, ne = !1), t(43, ie = !1), t(7, O = ""), (r = F == null ? void 0 : F.dispatch) == null || r.call(F, "blur"));
  }
  function St() {
    pe();
  }
  function Mt(r) {
    t(7, O = r.currentTarget.value), pe();
  }
  function Dt(r) {
    var y, P, T, K;
    !L || r.target === J || (P = (y = r.target).closest) != null && P.call(y, "button") || (K = (T = r.target).closest) != null && K.call(T, ".hierarchy-selector-chip") || (r.preventDefault(), pe());
  }
  function Oe(r) {
    te != null && te.contains(r.target) || le != null && le.contains(r.target) || qe();
  }
  function Ze(r, y) {
    var P;
    L && (t(16, U = r), (P = y.dataTransfer) == null || P.setData("text/plain", String(r)), y.dataTransfer && (y.dataTransfer.effectAllowed = "move"));
  }
  function Lt() {
    t(16, U = null), t(17, ue = null);
  }
  function je(r, y) {
    if (y.preventDefault(), U === null || U === r) return;
    const P = [...n], [T] = P.splice(U, 1);
    P.splice(r, 0, T), t(16, U = null), t(17, ue = null), me(P);
  }
  function zt(r) {
    r.key === "Escape" ? (r.preventDefault(), O ? t(7, O = "") : qe()) : r.key === "Enter" && o && _.length ? (r.preventDefault(), ge(_[0])) : r.key === "ArrowDown" ? (r.preventDefault(), pe()) : r.key === "Backspace" && !O && n.length && Le(n.length - 1);
  }
  function Ct() {
    return He(I);
  }
  hl(() => {
    document.addEventListener("pointerdown", Oe, !0), window.addEventListener("resize", X), window.addEventListener("scroll", X, !0);
  }), fl(() => {
    document.removeEventListener("pointerdown", Oe, !0), window.removeEventListener("resize", X), window.removeEventListener("scroll", X, !0);
  });
  const Ht = (r) => Le(r), Pt = (r, y) => Ze(r, y), Et = (r, y) => {
    y.preventDefault(), t(17, ue = r);
  }, Vt = () => t(17, ue = null), Bt = (r, y) => je(r, y);
  function Nt(r) {
    ke[r ? "unshift" : "push"](() => {
      J = r, t(13, J);
    });
  }
  function At() {
    O = this.value, t(7, O);
  }
  function Tt(r) {
    ke[r ? "unshift" : "push"](() => {
      oe = r, t(12, oe);
    });
  }
  const Rt = (r) => ge(r), qt = (r, y) => {
    (y.key === "Enter" || y.key === " ") && (y.preventDefault(), ge(r));
  };
  function Ot(r) {
    ke[r ? "unshift" : "push"](() => {
      le = r, t(14, le);
    });
  }
  function Zt(r) {
    ke[r ? "unshift" : "push"](() => {
      te = r, t(11, te);
    });
  }
  return l.$$set = (r) => {
    "elem_id" in r && t(0, p = r.elem_id), "elem_classes" in r && t(35, z = r.elem_classes), "visible" in r && t(1, h = r.visible), "value" in r && t(34, I = r.value), "hierarchy" in r && t(36, B = r.hierarchy), "height" in r && t(37, v = r.height), "label" in r && t(2, H = r.label), "info" in r && t(3, C = r.info), "show_label" in r && t(4, g = r.show_label), "container" in r && t(38, b = r.container), "scale" in r && t(39, f = r.scale), "min_width" in r && t(40, D = r.min_width), "interactive" in r && t(5, L = r.interactive), "gradio" in r && t(41, F = r.gradio);
  }, l.$$.update = () => {
    l.$$.dirty[1] & /*value*/
    8 && t(8, n = He(I)), l.$$.dirty[1] & /*hierarchy*/
    32 && t(10, i = bl(B)), l.$$.dirty[0] & /*normalizedHierarchy*/
    1024 && (Te = wl(i)), l.$$.dirty[0] & /*normalizedHierarchy*/
    1024 && t(46, s = kl(i)), l.$$.dirty[0] & /*searchQuery*/
    128 && t(45, a = O.trim().toLowerCase()), l.$$.dirty[1] & /*searchTerm*/
    16384 && t(9, o = a.length > 0), l.$$.dirty[0] & /*searchMode*/
    512 | l.$$.dirty[1] & /*flatItems, searchTerm*/
    49152 && t(19, _ = o ? yl(s, a) : []), l.$$.dirty[0] & /*elem_id*/
    1 && t(22, k = p ? `${p}-panel` : `${yt}-panel`), l.$$.dirty[1] & /*height*/
    64 && t(44, m = Math.max(10, Number(v) || 10)), l.$$.dirty[1] & /*panelRows*/
    8192 && (w = m * ut + vl), l.$$.dirty[1] & /*container, focused, elem_classes*/
    4240 && t(21, u = [
      "hierarchy-selector",
      b ? "hierarchy-selector-container" : "",
      ie ? "hierarchy-selector-focused" : "",
      ...Array.isArray(z) ? z : []
    ].filter(Boolean).join(" ")), l.$$.dirty[1] & /*scale, min_width*/
    768 && t(20, c = [
      f !== null ? `flex-grow:${f};` : "",
      D !== void 0 ? `min-width:${D}px;` : ""
    ].join("")), l.$$.dirty[0] & /*open, selectedValue, searchQuery*/
    448 && ne && (n || O) && Ce().then(X);
  }, [
    p,
    h,
    H,
    C,
    g,
    L,
    ne,
    O,
    n,
    o,
    i,
    te,
    oe,
    J,
    le,
    re,
    U,
    ue,
    Re,
    _,
    c,
    u,
    k,
    It,
    ge,
    Le,
    Ft,
    St,
    Mt,
    Dt,
    Ze,
    Lt,
    je,
    zt,
    I,
    z,
    B,
    v,
    b,
    f,
    D,
    F,
    Ct,
    ie,
    m,
    a,
    s,
    Ht,
    Pt,
    Et,
    Vt,
    Bt,
    Nt,
    At,
    Tt,
    Rt,
    qt,
    Ot,
    Zt
  ];
}
class Ml extends tl {
  constructor(e) {
    super(), al(
      this,
      e,
      Sl,
      pl,
      ul,
      {
        elem_id: 0,
        elem_classes: 35,
        visible: 1,
        value: 34,
        hierarchy: 36,
        height: 37,
        label: 2,
        info: 3,
        show_label: 4,
        container: 38,
        scale: 39,
        min_width: 40,
        interactive: 5,
        gradio: 41,
        get_value: 42
      },
      null,
      [-1, -1, -1]
    );
  }
  get elem_id() {
    return this.$$.ctx[0];
  }
  set elem_id(e) {
    this.$$set({ elem_id: e }), N();
  }
  get elem_classes() {
    return this.$$.ctx[35];
  }
  set elem_classes(e) {
    this.$$set({ elem_classes: e }), N();
  }
  get visible() {
    return this.$$.ctx[1];
  }
  set visible(e) {
    this.$$set({ visible: e }), N();
  }
  get value() {
    return this.$$.ctx[34];
  }
  set value(e) {
    this.$$set({ value: e }), N();
  }
  get hierarchy() {
    return this.$$.ctx[36];
  }
  set hierarchy(e) {
    this.$$set({ hierarchy: e }), N();
  }
  get height() {
    return this.$$.ctx[37];
  }
  set height(e) {
    this.$$set({ height: e }), N();
  }
  get label() {
    return this.$$.ctx[2];
  }
  set label(e) {
    this.$$set({ label: e }), N();
  }
  get info() {
    return this.$$.ctx[3];
  }
  set info(e) {
    this.$$set({ info: e }), N();
  }
  get show_label() {
    return this.$$.ctx[4];
  }
  set show_label(e) {
    this.$$set({ show_label: e }), N();
  }
  get container() {
    return this.$$.ctx[38];
  }
  set container(e) {
    this.$$set({ container: e }), N();
  }
  get scale() {
    return this.$$.ctx[39];
  }
  set scale(e) {
    this.$$set({ scale: e }), N();
  }
  get min_width() {
    return this.$$.ctx[40];
  }
  set min_width(e) {
    this.$$set({ min_width: e }), N();
  }
  get interactive() {
    return this.$$.ctx[5];
  }
  set interactive(e) {
    this.$$set({ interactive: e }), N();
  }
  get gradio() {
    return this.$$.ctx[41];
  }
  set gradio(e) {
    this.$$set({ gradio: e }), N();
  }
  get get_value() {
    return this.$$.ctx[42];
  }
}
export {
  Ml as default
};
