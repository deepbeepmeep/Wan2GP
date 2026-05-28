const {
  SvelteComponent: _l,
  append: P,
  attr: S,
  check_outros: jt,
  create_component: dl,
  destroy_block: ml,
  destroy_component: gl,
  detach: _e,
  element: Fe,
  empty: Dt,
  ensure_array_like: Be,
  flush: U,
  group_outros: Ht,
  init: pl,
  insert: de,
  listen: Ae,
  mount_component: bl,
  outro_and_destroy_block: yl,
  run_all: Lt,
  safe_not_equal: vl,
  set_data: Ct,
  set_style: Te,
  space: se,
  svg_element: te,
  text: Bt,
  toggle_class: Re,
  transition_in: ze,
  transition_out: qe,
  update_keyed_each: ot
} = window.__gradio__svelte__internal;
function ct(l, e, t) {
  const n = l.slice();
  n[13] = e[t];
  const r = (
    /*value*/
    n[4].includes(
      /*valueForItem*/
      n[7](
        /*item*/
        n[13]
      )
    )
  );
  return n[14] = r, n;
}
function ut(l, e, t) {
  const n = l.slice();
  n[17] = e[t];
  const r = Et(
    /*folder*/
    n[17]
  );
  return n[18] = r, n;
}
function ht(l) {
  let e, t;
  return e = new Vt({
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
      dl(e.$$.fragment);
    },
    m(n, r) {
      bl(e, n, r), t = !0;
    },
    p(n, r) {
      const s = {};
      r & /*folders*/
      1 && (s.folders = /*folder*/
      n[17].folders || []), r & /*folders*/
      1 && (s.items = /*folder*/
      n[17].items || []), r & /*depth*/
      4 && (s.depth = /*depth*/
      n[2] + 1), r & /*expanded*/
      8 && (s.expanded = /*expanded*/
      n[3]), r & /*value*/
      16 && (s.value = /*value*/
      n[4]), r & /*toggleItem*/
      32 && (s.toggleItem = /*toggleItem*/
      n[5]), r & /*toggleFolder*/
      64 && (s.toggleFolder = /*toggleFolder*/
      n[6]), r & /*valueForItem*/
      128 && (s.valueForItem = /*valueForItem*/
      n[7]), r & /*labelForItem*/
      256 && (s.labelForItem = /*labelForItem*/
      n[8]), e.$set(s);
    },
    i(n) {
      t || (ze(e.$$.fragment, n), t = !0);
    },
    o(n) {
      qe(e.$$.fragment, n), t = !1;
    },
    d(n) {
      gl(e, n);
    }
  };
}
function ft(l, e) {
  let t, n, r, s, a, o, _, b, w, m = Ee(
    /*folder*/
    e[17]
  ) + "", c, u, g, L = (
    /*expanded*/
    e[3].has(
      /*path*/
      e[18]
    )
  ), f, y, T, j;
  function v() {
    return (
      /*click_handler*/
      e[9](
        /*path*/
        e[18]
      )
    );
  }
  function G(...F) {
    return (
      /*keydown_handler*/
      e[10](
        /*path*/
        e[18],
        ...F
      )
    );
  }
  let k = L && ht(e);
  return {
    key: l,
    first: null,
    c() {
      t = Fe("div"), n = te("svg"), r = te("path"), s = se(), a = te("svg"), o = te("path"), _ = te("path"), b = se(), w = Fe("span"), c = Bt(m), g = se(), k && k.c(), f = Dt(), S(r, "d", "M6 4.5L10 8l-4 3.5"), S(n, "class", "hierarchy-twist svelte-t6vyzt"), S(n, "viewBox", "0 0 16 16"), S(n, "aria-hidden", "true"), Re(
        n,
        "hierarchy-twist-open",
        /*expanded*/
        e[3].has(
          /*path*/
          e[18]
        )
      ), S(o, "d", "M2.75 6.25h5.4l1.55 1.7h7.55c.55 0 1 .45 1 1v6.3c0 .55-.45 1-1 1H2.75c-.55 0-1-.45-1-1v-8c0-.55.45-1 1-1Z"), S(_, "d", "M2.25 7.95V5.6c0-.55.45-1 1-1h4.5l1.35 1.65"), S(a, "class", "hierarchy-icon hierarchy-folder-icon svelte-t6vyzt"), S(a, "viewBox", "0 0 20 20"), S(a, "aria-hidden", "true"), S(w, "class", "hierarchy-name svelte-t6vyzt"), S(t, "class", "hierarchy-row hierarchy-folder svelte-t6vyzt"), S(t, "role", "button"), S(t, "tabindex", "0"), S(t, "title", u = Ee(
        /*folder*/
        e[17]
      )), Te(t, "padding-left", `${/*depth*/
      e[2] * 18 + 6}px`), this.first = t;
    },
    m(F, I) {
      de(F, t, I), P(t, n), P(n, r), P(t, s), P(t, a), P(a, o), P(a, _), P(t, b), P(t, w), P(w, c), de(F, g, I), k && k.m(F, I), de(F, f, I), y = !0, T || (j = [
        Ae(t, "click", v),
        Ae(t, "keydown", G)
      ], T = !0);
    },
    p(F, I) {
      e = F, (!y || I & /*expanded, folderPath, folders*/
      9) && Re(
        n,
        "hierarchy-twist-open",
        /*expanded*/
        e[3].has(
          /*path*/
          e[18]
        )
      ), (!y || I & /*folders*/
      1) && m !== (m = Ee(
        /*folder*/
        e[17]
      ) + "") && Ct(c, m), (!y || I & /*folders*/
      1 && u !== (u = Ee(
        /*folder*/
        e[17]
      ))) && S(t, "title", u), I & /*depth*/
      4 && Te(t, "padding-left", `${/*depth*/
      e[2] * 18 + 6}px`), I & /*expanded, folders*/
      9 && (L = /*expanded*/
      e[3].has(
        /*path*/
        e[18]
      )), L ? k ? (k.p(e, I), I & /*expanded, folders*/
      9 && ze(k, 1)) : (k = ht(e), k.c(), ze(k, 1), k.m(f.parentNode, f)) : k && (Ht(), qe(k, 1, 1, () => {
        k = null;
      }), jt());
    },
    i(F) {
      y || (ze(k), y = !0);
    },
    o(F) {
      qe(k), y = !1;
    },
    d(F) {
      F && (_e(t), _e(g), _e(f)), k && k.d(F), T = !1, Lt(j);
    }
  };
}
function _t(l, e) {
  let t, n, r, s, a, o, _, b, w = (
    /*labelForItem*/
    e[8](
      /*item*/
      e[13]
    ) + ""
  ), m, c, u, g, L, f;
  function y() {
    return (
      /*click_handler_1*/
      e[11](
        /*item*/
        e[13]
      )
    );
  }
  function T(...j) {
    return (
      /*keydown_handler_1*/
      e[12](
        /*item*/
        e[13],
        ...j
      )
    );
  }
  return {
    key: l,
    first: null,
    c() {
      t = Fe("div"), n = Fe("span"), r = se(), s = te("svg"), a = te("path"), o = te("path"), _ = se(), b = Fe("span"), m = Bt(w), c = se(), S(n, "class", "hierarchy-twist-spacer svelte-t6vyzt"), S(a, "d", "M5.25 2.75h6.05L15.75 7.2v10.05H5.25V2.75Z"), S(o, "d", "M11.25 2.95V7.3h4.3"), S(s, "class", "hierarchy-icon hierarchy-item-icon svelte-t6vyzt"), S(s, "viewBox", "0 0 20 20"), S(s, "aria-hidden", "true"), S(b, "class", "hierarchy-name svelte-t6vyzt"), S(t, "class", "hierarchy-row hierarchy-item svelte-t6vyzt"), S(t, "title", u = /*labelForItem*/
      e[8](
        /*item*/
        e[13]
      )), S(t, "role", "button"), S(t, "tabindex", "0"), S(t, "aria-pressed", g = /*selected*/
      e[14]), Re(
        t,
        "hierarchy-item-selected",
        /*selected*/
        e[14]
      ), Te(t, "padding-left", `${/*depth*/
      e[2] * 18 + 6}px`), this.first = t;
    },
    m(j, v) {
      de(j, t, v), P(t, n), P(t, r), P(t, s), P(s, a), P(s, o), P(t, _), P(t, b), P(b, m), P(t, c), L || (f = [
        Ae(t, "click", y),
        Ae(t, "keydown", T)
      ], L = !0);
    },
    p(j, v) {
      e = j, v & /*labelForItem, items*/
      258 && w !== (w = /*labelForItem*/
      e[8](
        /*item*/
        e[13]
      ) + "") && Ct(m, w), v & /*labelForItem, items*/
      258 && u !== (u = /*labelForItem*/
      e[8](
        /*item*/
        e[13]
      )) && S(t, "title", u), v & /*value, valueForItem, items*/
      146 && g !== (g = /*selected*/
      e[14]) && S(t, "aria-pressed", g), v & /*value, valueForItem, items*/
      146 && Re(
        t,
        "hierarchy-item-selected",
        /*selected*/
        e[14]
      ), v & /*depth*/
      4 && Te(t, "padding-left", `${/*depth*/
      e[2] * 18 + 6}px`);
    },
    d(j) {
      j && _e(t), L = !1, Lt(f);
    }
  };
}
function kl(l) {
  let e = [], t = /* @__PURE__ */ new Map(), n, r = [], s = /* @__PURE__ */ new Map(), a, o, _ = Be(
    /*folders*/
    l[0]
  );
  const b = (c) => Et(
    /*folder*/
    c[17]
  );
  for (let c = 0; c < _.length; c += 1) {
    let u = ut(l, _, c), g = b(u);
    t.set(g, e[c] = ft(g, u));
  }
  let w = Be(
    /*items*/
    l[1]
  );
  const m = (c) => (
    /*valueForItem*/
    c[7](
      /*item*/
      c[13]
    )
  );
  for (let c = 0; c < w.length; c += 1) {
    let u = ct(l, w, c), g = m(u);
    s.set(g, r[c] = _t(g, u));
  }
  return {
    c() {
      for (let c = 0; c < e.length; c += 1)
        e[c].c();
      n = se();
      for (let c = 0; c < r.length; c += 1)
        r[c].c();
      a = Dt();
    },
    m(c, u) {
      for (let g = 0; g < e.length; g += 1)
        e[g] && e[g].m(c, u);
      de(c, n, u);
      for (let g = 0; g < r.length; g += 1)
        r[g] && r[g].m(c, u);
      de(c, a, u), o = !0;
    },
    p(c, [u]) {
      u & /*folders, depth, expanded, value, toggleItem, toggleFolder, valueForItem, labelForItem, folderPath, folderLabel*/
      509 && (_ = Be(
        /*folders*/
        c[0]
      ), Ht(), e = ot(e, u, b, 1, c, _, t, n.parentNode, yl, ft, n, ut), jt()), u & /*labelForItem, items, value, valueForItem, depth, toggleItem*/
      438 && (w = Be(
        /*items*/
        c[1]
      ), r = ot(r, u, m, 1, c, w, s, a.parentNode, ml, _t, a, ct));
    },
    i(c) {
      if (!o) {
        for (let u = 0; u < _.length; u += 1)
          ze(e[u]);
        o = !0;
      }
    },
    o(c) {
      for (let u = 0; u < e.length; u += 1)
        qe(e[u]);
      o = !1;
    },
    d(c) {
      c && (_e(n), _e(a));
      for (let u = 0; u < e.length; u += 1)
        e[u].d(c);
      for (let u = 0; u < r.length; u += 1)
        r[u].d(c);
    }
  };
}
function Ee(l) {
  return String(l.name || l.path || "");
}
function Et(l) {
  return String(l.path || l.name || "");
}
function wl(l, e, t) {
  let { folders: n = [] } = e, { items: r = [] } = e, { depth: s = 0 } = e, { expanded: a } = e, { value: o } = e, { toggleItem: _ } = e, { toggleFolder: b } = e, { valueForItem: w } = e, { labelForItem: m } = e;
  const c = (f) => b(f), u = (f, y) => {
    (y.key === "Enter" || y.key === " ") && (y.preventDefault(), b(f));
  }, g = (f) => _(f), L = (f, y) => {
    (y.key === "Enter" || y.key === " ") && (y.preventDefault(), _(f));
  };
  return l.$$set = (f) => {
    "folders" in f && t(0, n = f.folders), "items" in f && t(1, r = f.items), "depth" in f && t(2, s = f.depth), "expanded" in f && t(3, a = f.expanded), "value" in f && t(4, o = f.value), "toggleItem" in f && t(5, _ = f.toggleItem), "toggleFolder" in f && t(6, b = f.toggleFolder), "valueForItem" in f && t(7, w = f.valueForItem), "labelForItem" in f && t(8, m = f.labelForItem);
  }, [
    n,
    r,
    s,
    a,
    o,
    _,
    b,
    w,
    m,
    c,
    u,
    g,
    L
  ];
}
class Vt extends _l {
  constructor(e) {
    super(), pl(this, e, wl, kl, vl, {
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
    this.$$set({ folders: e }), U();
  }
  get items() {
    return this.$$.ctx[1];
  }
  set items(e) {
    this.$$set({ items: e }), U();
  }
  get depth() {
    return this.$$.ctx[2];
  }
  set depth(e) {
    this.$$set({ depth: e }), U();
  }
  get expanded() {
    return this.$$.ctx[3];
  }
  set expanded(e) {
    this.$$set({ expanded: e }), U();
  }
  get value() {
    return this.$$.ctx[4];
  }
  set value(e) {
    this.$$set({ value: e }), U();
  }
  get toggleItem() {
    return this.$$.ctx[5];
  }
  set toggleItem(e) {
    this.$$set({ toggleItem: e }), U();
  }
  get toggleFolder() {
    return this.$$.ctx[6];
  }
  set toggleFolder(e) {
    this.$$set({ toggleFolder: e }), U();
  }
  get valueForItem() {
    return this.$$.ctx[7];
  }
  set valueForItem(e) {
    this.$$set({ valueForItem: e }), U();
  }
  get labelForItem() {
    return this.$$.ctx[8];
  }
  set labelForItem(e) {
    this.$$set({ labelForItem: e }), U();
  }
}
const {
  SvelteComponent: Il,
  action_destroyer: Fl,
  append: z,
  attr: h,
  binding_callbacks: ke,
  check_outros: Xe,
  create_component: zl,
  destroy_block: Sl,
  destroy_component: Ml,
  destroy_each: jl,
  detach: O,
  element: B,
  empty: Ye,
  ensure_array_like: Oe,
  flush: V,
  group_outros: xe,
  init: Dl,
  insert: Z,
  listen: R,
  mount_component: Hl,
  noop: Je,
  null_to_empty: dt,
  run_all: $e,
  safe_not_equal: Ll,
  set_data: oe,
  set_input_value: mt,
  space: J,
  stop_propagation: et,
  svg_element: Qe,
  text: X,
  toggle_class: le,
  transition_in: Y,
  transition_out: ae,
  update_keyed_each: Cl
} = window.__gradio__svelte__internal, { onDestroy: Bl, onMount: El, tick: Ve } = window.__gradio__svelte__internal;
function gt(l, e, t) {
  const n = l.slice();
  n[84] = e[t];
  const r = (
    /*selectedValue*/
    n[13].includes(Se(
      /*item*/
      n[84]
    ))
  );
  return n[85] = r, n;
}
function pt(l, e, t) {
  const n = l.slice();
  return n[85] = e[t], n[89] = t, n;
}
function bt(l) {
  let e, t, n, r, s, a, o, _, b, w, m, c, u, g, L, f, y, T, j, v = (
    /*show_label*/
    l[6] && /*label*/
    l[4] && yt(l)
  ), G = Oe(
    /*selectedValue*/
    l[13]
  ), k = [];
  for (let d = 0; d < G.length; d += 1)
    k[d] = vt(pt(l, G, d));
  let F = (
    /*selectedValue*/
    l[13].length > 0 && kt(l)
  ), I = (
    /*open*/
    l[8] && wt(l)
  ), N = (
    /*info*/
    l[5] && zt(l)
  );
  return {
    c() {
      e = B("div"), t = B("div"), v && v.c(), n = J(), r = B("div"), s = B("div");
      for (let d = 0; d < k.length; d += 1)
        k[d].c();
      a = J(), o = B("input"), m = J(), F && F.c(), g = J(), I && I.c(), L = J(), N && N.c(), h(o, "class", "hierarchy-selector-search-input svelte-ohjrzb"), h(o, "type", "text"), h(o, "autocomplete", "off"), h(o, "spellcheck", "false"), o.disabled = _ = !/*interactive*/
      l[7], h(o, "tabindex", b = /*interactive*/
      l[7] ? 0 : -1), h(o, "placeholder", w = /*show_placeholder*/
      l[3] && /*selectedValue*/
      l[13].length === 0 ? (
        /*label*/
        l[4]
      ) : ""), h(
        o,
        "aria-label",
        /*label*/
        l[4]
      ), h(s, "class", "hierarchy-selector-chips svelte-ohjrzb"), h(r, "class", "hierarchy-selector-input svelte-ohjrzb"), h(r, "role", "combobox"), h(r, "tabindex", c = /*interactive*/
      l[7] ? 0 : -1), h(r, "aria-haspopup", "tree"), h(r, "aria-expanded", u = /*open*/
      l[8] ? "true" : "false"), h(
        r,
        "aria-controls",
        /*panelId*/
        l[26]
      ), le(r, "hierarchy-selector-disabled", !/*interactive*/
      l[7]), h(t, "class", "hierarchy-selector-field svelte-ohjrzb"), h(
        e,
        "id",
        /*elem_id*/
        l[0]
      ), h(e, "class", f = dt(
        /*classes*/
        l[25]
      ) + " svelte-ohjrzb"), h(
        e,
        "style",
        /*style*/
        l[24]
      );
    },
    m(d, D) {
      Z(d, e, D), z(e, t), v && v.m(t, null), z(t, n), z(t, r), z(r, s);
      for (let C = 0; C < k.length; C += 1)
        k[C] && k[C].m(s, null);
      z(s, a), z(s, o), l[64](o), mt(
        o,
        /*searchQuery*/
        l[10]
      ), z(r, m), F && F.m(r, null), l[66](r), z(t, g), I && I.m(t, null), z(e, L), N && N.m(e, null), l[71](e), y = !0, T || (j = [
        R(
          o,
          "input",
          /*input_input_handler*/
          l[65]
        ),
        R(
          o,
          "focus",
          /*onSearchFocus*/
          l[33]
        ),
        R(
          o,
          "input",
          /*onSearchInput*/
          l[34]
        ),
        R(o, "keydown", et(
          /*onInputKeydown*/
          l[39]
        )),
        R(
          r,
          "mousedown",
          /*onInputPointerDown*/
          l[35]
        ),
        R(
          r,
          "keydown",
          /*onInputKeydown*/
          l[39]
        )
      ], T = !0);
    },
    p(d, D) {
      if (/*show_label*/
      d[6] && /*label*/
      d[4] ? v ? v.p(d, D) : (v = yt(d), v.c(), v.m(t, n)) : v && (v.d(1), v = null), D[0] & /*interactive, draggedIndex, dragOverIndex, removeValue, displayValue, selectedValue*/
      1348477056 | D[1] & /*onDragStart, onDragEnd, onDrop*/
      224) {
        G = Oe(
          /*selectedValue*/
          d[13]
        );
        let C;
        for (C = 0; C < G.length; C += 1) {
          const me = pt(d, G, C);
          k[C] ? k[C].p(me, D) : (k[C] = vt(me), k[C].c(), k[C].m(s, a));
        }
        for (; C < k.length; C += 1)
          k[C].d(1);
        k.length = G.length;
      }
      (!y || D[0] & /*interactive*/
      128 && _ !== (_ = !/*interactive*/
      d[7])) && (o.disabled = _), (!y || D[0] & /*interactive*/
      128 && b !== (b = /*interactive*/
      d[7] ? 0 : -1)) && h(o, "tabindex", b), (!y || D[0] & /*show_placeholder, selectedValue, label*/
      8216 && w !== (w = /*show_placeholder*/
      d[3] && /*selectedValue*/
      d[13].length === 0 ? (
        /*label*/
        d[4]
      ) : "")) && h(o, "placeholder", w), (!y || D[0] & /*label*/
      16) && h(
        o,
        "aria-label",
        /*label*/
        d[4]
      ), D[0] & /*searchQuery*/
      1024 && o.value !== /*searchQuery*/
      d[10] && mt(
        o,
        /*searchQuery*/
        d[10]
      ), /*selectedValue*/
      d[13].length > 0 ? F ? F.p(d, D) : (F = kt(d), F.c(), F.m(r, null)) : F && (F.d(1), F = null), (!y || D[0] & /*interactive*/
      128 && c !== (c = /*interactive*/
      d[7] ? 0 : -1)) && h(r, "tabindex", c), (!y || D[0] & /*open*/
      256 && u !== (u = /*open*/
      d[8] ? "true" : "false")) && h(r, "aria-expanded", u), (!y || D[0] & /*panelId*/
      67108864) && h(
        r,
        "aria-controls",
        /*panelId*/
        d[26]
      ), (!y || D[0] & /*interactive*/
      128) && le(r, "hierarchy-selector-disabled", !/*interactive*/
      d[7]), /*open*/
      d[8] ? I ? (I.p(d, D), D[0] & /*open*/
      256 && Y(I, 1)) : (I = wt(d), I.c(), Y(I, 1), I.m(t, null)) : I && (xe(), ae(I, 1, 1, () => {
        I = null;
      }), Xe()), /*info*/
      d[5] ? N ? N.p(d, D) : (N = zt(d), N.c(), N.m(e, null)) : N && (N.d(1), N = null), (!y || D[0] & /*elem_id*/
      1) && h(
        e,
        "id",
        /*elem_id*/
        d[0]
      ), (!y || D[0] & /*classes*/
      33554432 && f !== (f = dt(
        /*classes*/
        d[25]
      ) + " svelte-ohjrzb")) && h(e, "class", f), (!y || D[0] & /*style*/
      16777216) && h(
        e,
        "style",
        /*style*/
        d[24]
      );
    },
    i(d) {
      y || (Y(I), y = !0);
    },
    o(d) {
      ae(I), y = !1;
    },
    d(d) {
      d && O(e), v && v.d(), jl(k, d), l[64](null), F && F.d(), l[66](null), I && I.d(), N && N.d(), l[71](null), T = !1, $e(j);
    }
  };
}
function yt(l) {
  let e, t;
  return {
    c() {
      e = B("span"), t = X(
        /*label*/
        l[4]
      ), h(e, "class", "hierarchy-selector-label svelte-ohjrzb");
    },
    m(n, r) {
      Z(n, e, r), z(e, t);
    },
    p(n, r) {
      r[0] & /*label*/
      16 && oe(
        t,
        /*label*/
        n[4]
      );
    },
    d(n) {
      n && O(e);
    }
  };
}
function vt(l) {
  let e, t, n = (
    /*displayValue*/
    l[28](
      /*selected*/
      l[85]
    ) + ""
  ), r, s, a, o, _;
  function b() {
    return (
      /*click_handler*/
      l[59](
        /*index*/
        l[89]
      )
    );
  }
  function w(...u) {
    return (
      /*dragstart_handler*/
      l[60](
        /*index*/
        l[89],
        ...u
      )
    );
  }
  function m(...u) {
    return (
      /*dragover_handler*/
      l[61](
        /*index*/
        l[89],
        ...u
      )
    );
  }
  function c(...u) {
    return (
      /*drop_handler*/
      l[63](
        /*index*/
        l[89],
        ...u
      )
    );
  }
  return {
    c() {
      e = B("span"), t = B("span"), r = X(n), s = J(), a = B("button"), a.textContent = "x", h(t, "class", "hierarchy-selector-chip-text svelte-ohjrzb"), h(a, "type", "button"), h(a, "class", "hierarchy-selector-remove svelte-ohjrzb"), h(a, "aria-label", "Remove"), h(e, "class", "hierarchy-selector-chip svelte-ohjrzb"), h(e, "role", "listitem"), h(
        e,
        "draggable",
        /*interactive*/
        l[7]
      ), le(
        e,
        "hierarchy-selector-chip-dragging",
        /*draggedIndex*/
        l[21] === /*index*/
        l[89]
      ), le(
        e,
        "hierarchy-selector-chip-over",
        /*dragOverIndex*/
        l[22] === /*index*/
        l[89]
      );
    },
    m(u, g) {
      Z(u, e, g), z(e, t), z(t, r), z(e, s), z(e, a), o || (_ = [
        R(a, "click", et(b)),
        R(e, "dragstart", w),
        R(
          e,
          "dragend",
          /*onDragEnd*/
          l[37]
        ),
        R(e, "dragover", m),
        R(
          e,
          "dragleave",
          /*dragleave_handler*/
          l[62]
        ),
        R(e, "drop", c)
      ], o = !0);
    },
    p(u, g) {
      l = u, g[0] & /*selectedValue*/
      8192 && n !== (n = /*displayValue*/
      l[28](
        /*selected*/
        l[85]
      ) + "") && oe(r, n), g[0] & /*interactive*/
      128 && h(
        e,
        "draggable",
        /*interactive*/
        l[7]
      ), g[0] & /*draggedIndex*/
      2097152 && le(
        e,
        "hierarchy-selector-chip-dragging",
        /*draggedIndex*/
        l[21] === /*index*/
        l[89]
      ), g[0] & /*dragOverIndex*/
      4194304 && le(
        e,
        "hierarchy-selector-chip-over",
        /*dragOverIndex*/
        l[22] === /*index*/
        l[89]
      );
    },
    d(u) {
      u && O(e), o = !1, $e(_);
    }
  };
}
function kt(l) {
  let e, t, n;
  return {
    c() {
      e = B("button"), e.textContent = "x", h(e, "type", "button"), h(e, "class", "hierarchy-selector-clear svelte-ohjrzb"), h(e, "aria-label", "Clear selection");
    },
    m(r, s) {
      Z(r, e, s), t || (n = R(e, "click", et(
        /*clearValues*/
        l[31]
      )), t = !0);
    },
    p: Je,
    d(r) {
      r && O(e), t = !1, n();
    }
  };
}
function wt(l) {
  let e, t, n, r, s, a, o;
  const _ = [Nl, Vl], b = [];
  function w(m, c) {
    return (
      /*searchMode*/
      m[14] ? 0 : 1
    );
  }
  return n = w(l), r = b[n] = _[n](l), {
    c() {
      e = B("div"), t = B("div"), r.c(), h(t, "class", "hierarchy-selector-panel-content svelte-ohjrzb"), h(
        e,
        "id",
        /*panelId*/
        l[26]
      ), h(e, "class", "hierarchy-selector-panel svelte-ohjrzb"), h(
        e,
        "style",
        /*panelStyle*/
        l[23]
      );
    },
    m(m, c) {
      Z(m, e, c), z(e, t), b[n].m(t, null), l[69](t), l[70](e), s = !0, a || (o = Fl(Ql.call(null, e)), a = !0);
    },
    p(m, c) {
      let u = n;
      n = w(m), n === u ? b[n].p(m, c) : (xe(), ae(b[u], 1, 1, () => {
        b[u] = null;
      }), Xe(), r = b[n], r ? r.p(m, c) : (r = b[n] = _[n](m), r.c()), Y(r, 1), r.m(t, null)), (!s || c[0] & /*panelId*/
      67108864) && h(
        e,
        "id",
        /*panelId*/
        m[26]
      ), (!s || c[0] & /*panelStyle*/
      8388608) && h(
        e,
        "style",
        /*panelStyle*/
        m[23]
      );
    },
    i(m) {
      s || (Y(r), s = !0);
    },
    o(m) {
      ae(r), s = !1;
    },
    d(m) {
      m && O(e), b[n].d(), l[69](null), l[70](null), a = !1, o();
    }
  };
}
function Vl(l) {
  let e, t;
  return e = new Vt({
    props: {
      folders: (
        /*normalizedHierarchy*/
        l[12].folders || []
      ),
      items: (
        /*normalizedHierarchy*/
        l[12].items || []
      ),
      depth: 0,
      expanded: (
        /*expanded*/
        l[9]
      ),
      value: (
        /*selectedValue*/
        l[13]
      ),
      toggleItem: (
        /*toggleItem*/
        l[29]
      ),
      toggleFolder: (
        /*toggleFolder*/
        l[32]
      ),
      valueForItem: Se,
      labelForItem: x
    }
  }), {
    c() {
      zl(e.$$.fragment);
    },
    m(n, r) {
      Hl(e, n, r), t = !0;
    },
    p(n, r) {
      const s = {};
      r[0] & /*normalizedHierarchy*/
      4096 && (s.folders = /*normalizedHierarchy*/
      n[12].folders || []), r[0] & /*normalizedHierarchy*/
      4096 && (s.items = /*normalizedHierarchy*/
      n[12].items || []), r[0] & /*expanded*/
      512 && (s.expanded = /*expanded*/
      n[9]), r[0] & /*selectedValue*/
      8192 && (s.value = /*selectedValue*/
      n[13]), e.$set(s);
    },
    i(n) {
      t || (Y(e.$$.fragment, n), t = !0);
    },
    o(n) {
      ae(e.$$.fragment, n), t = !1;
    },
    d(n) {
      Ml(e, n);
    }
  };
}
function Nl(l) {
  let e;
  function t(s, a) {
    return (
      /*searchResults*/
      s[11].length ? Al : Pl
    );
  }
  let n = t(l), r = n(l);
  return {
    c() {
      r.c(), e = Ye();
    },
    m(s, a) {
      r.m(s, a), Z(s, e, a);
    },
    p(s, a) {
      n === (n = t(s)) && r ? r.p(s, a) : (r.d(1), r = n(s), r && (r.c(), r.m(e.parentNode, e)));
    },
    i: Je,
    o: Je,
    d(s) {
      s && O(e), r.d(s);
    }
  };
}
function Pl(l) {
  let e, t;
  return {
    c() {
      e = B("div"), t = X(
        /*search_empty_label*/
        l[2]
      ), h(e, "class", "hierarchy-search-empty svelte-ohjrzb");
    },
    m(n, r) {
      Z(n, e, r), z(e, t);
    },
    p(n, r) {
      r[0] & /*search_empty_label*/
      4 && oe(
        t,
        /*search_empty_label*/
        n[2]
      );
    },
    d(n) {
      n && O(e);
    }
  };
}
function Al(l) {
  let e = [], t = /* @__PURE__ */ new Map(), n, r = Oe(
    /*searchResults*/
    l[11]
  );
  const s = (a) => Se(
    /*item*/
    a[84]
  );
  for (let a = 0; a < r.length; a += 1) {
    let o = gt(l, r, a), _ = s(o);
    t.set(_, e[a] = Ft(_, o));
  }
  return {
    c() {
      for (let a = 0; a < e.length; a += 1)
        e[a].c();
      n = Ye();
    },
    m(a, o) {
      for (let _ = 0; _ < e.length; _ += 1)
        e[_] && e[_].m(a, o);
      Z(a, n, o);
    },
    p(a, o) {
      o[0] & /*searchResults, selectedLabelForItem, selectedValue, toggleItem, breadcrumbMode*/
      671131648 && (r = Oe(
        /*searchResults*/
        a[11]
      ), e = Cl(e, o, s, 1, a, r, t, n.parentNode, Sl, Ft, n, gt));
    },
    d(a) {
      a && O(n);
      for (let o = 0; o < e.length; o += 1)
        e[o].d(a);
    }
  };
}
function Tl(l) {
  let e, t, n = x(
    /*item*/
    l[84]
  ) + "", r, s, a = (
    /*item*/
    l[84].search_path && It(l)
  );
  return {
    c() {
      e = B("span"), t = B("span"), r = X(n), s = J(), a && a.c(), h(t, "class", "hierarchy-search-name svelte-ohjrzb"), h(e, "class", "hierarchy-search-label svelte-ohjrzb");
    },
    m(o, _) {
      Z(o, e, _), z(e, t), z(t, r), z(e, s), a && a.m(e, null);
    },
    p(o, _) {
      _[0] & /*searchResults*/
      2048 && n !== (n = x(
        /*item*/
        o[84]
      ) + "") && oe(r, n), /*item*/
      o[84].search_path ? a ? a.p(o, _) : (a = It(o), a.c(), a.m(e, null)) : a && (a.d(1), a = null);
    },
    d(o) {
      o && O(e), a && a.d();
    }
  };
}
function Rl(l) {
  let e, t = (
    /*item*/
    l[84].search_display + ""
  ), n;
  return {
    c() {
      e = B("span"), n = X(t), h(e, "class", "hierarchy-search-label hierarchy-search-name svelte-ohjrzb");
    },
    m(r, s) {
      Z(r, e, s), z(e, n);
    },
    p(r, s) {
      s[0] & /*searchResults*/
      2048 && t !== (t = /*item*/
      r[84].search_display + "") && oe(n, t);
    },
    d(r) {
      r && O(e);
    }
  };
}
function It(l) {
  let e, t, n = (
    /*item*/
    l[84].search_path + ""
  ), r, s;
  return {
    c() {
      e = B("span"), t = X("["), r = X(n), s = X("]"), h(e, "class", "hierarchy-search-path svelte-ohjrzb");
    },
    m(a, o) {
      Z(a, e, o), z(e, t), z(e, r), z(e, s);
    },
    p(a, o) {
      o[0] & /*searchResults*/
      2048 && n !== (n = /*item*/
      a[84].search_path + "") && oe(r, n);
    },
    d(a) {
      a && O(e);
    }
  };
}
function Ft(l, e) {
  let t, n, r, s, a, o, _, b, w, m, c, u;
  function g(j, v) {
    return (
      /*breadcrumbMode*/
      j[15] ? Rl : Tl
    );
  }
  let L = g(e), f = L(e);
  function y() {
    return (
      /*click_handler_1*/
      e[67](
        /*item*/
        e[84]
      )
    );
  }
  function T(...j) {
    return (
      /*keydown_handler*/
      e[68](
        /*item*/
        e[84],
        ...j
      )
    );
  }
  return {
    key: l,
    first: null,
    c() {
      t = B("div"), n = B("span"), r = J(), s = Qe("svg"), a = Qe("path"), o = Qe("path"), _ = J(), f.c(), b = J(), h(n, "class", "hierarchy-search-spacer svelte-ohjrzb"), h(a, "d", "M5.25 2.75h6.05L15.75 7.2v10.05H5.25V2.75Z"), h(o, "d", "M11.25 2.95V7.3h4.3"), h(s, "class", "hierarchy-search-icon svelte-ohjrzb"), h(s, "viewBox", "0 0 20 20"), h(s, "aria-hidden", "true"), h(t, "class", "hierarchy-search-row svelte-ohjrzb"), h(t, "title", w = /*item*/
      e[84].search_display || /*selectedLabelForItem*/
      e[27](
        /*item*/
        e[84]
      )), h(t, "role", "button"), h(t, "tabindex", "0"), h(t, "aria-pressed", m = /*selected*/
      e[85]), le(
        t,
        "hierarchy-search-row-selected",
        /*selected*/
        e[85]
      ), this.first = t;
    },
    m(j, v) {
      Z(j, t, v), z(t, n), z(t, r), z(t, s), z(s, a), z(s, o), z(t, _), f.m(t, null), z(t, b), c || (u = [
        R(t, "click", y),
        R(t, "keydown", T)
      ], c = !0);
    },
    p(j, v) {
      e = j, L === (L = g(e)) && f ? f.p(e, v) : (f.d(1), f = L(e), f && (f.c(), f.m(t, b))), v[0] & /*searchResults*/
      2048 && w !== (w = /*item*/
      e[84].search_display || /*selectedLabelForItem*/
      e[27](
        /*item*/
        e[84]
      )) && h(t, "title", w), v[0] & /*selectedValue, searchResults*/
      10240 && m !== (m = /*selected*/
      e[85]) && h(t, "aria-pressed", m), v[0] & /*selectedValue, searchResults*/
      10240 && le(
        t,
        "hierarchy-search-row-selected",
        /*selected*/
        e[85]
      );
    },
    d(j) {
      j && O(t), f.d(), c = !1, $e(u);
    }
  };
}
function zt(l) {
  let e, t;
  return {
    c() {
      e = B("div"), t = X(
        /*info*/
        l[5]
      ), h(e, "class", "hierarchy-selector-info svelte-ohjrzb");
    },
    m(n, r) {
      Z(n, e, r), z(e, t);
    },
    p(n, r) {
      r[0] & /*info*/
      32 && oe(
        t,
        /*info*/
        n[5]
      );
    },
    d(n) {
      n && O(e);
    }
  };
}
function ql(l) {
  let e, t, n = (
    /*visible*/
    l[1] && bt(l)
  );
  return {
    c() {
      n && n.c(), e = Ye();
    },
    m(r, s) {
      n && n.m(r, s), Z(r, e, s), t = !0;
    },
    p(r, s) {
      /*visible*/
      r[1] ? n ? (n.p(r, s), s[0] & /*visible*/
      2 && Y(n, 1)) : (n = bt(r), n.c(), Y(n, 1), n.m(e.parentNode, e)) : n && (xe(), ae(n, 1, 1, () => {
        n = null;
      }), Xe());
    },
    i(r) {
      t || (Y(n), t = !0);
    },
    o(r) {
      ae(n), t = !1;
    },
    d(r) {
      r && O(e), n && n.d(r);
    }
  };
}
const Ne = 28, we = 8, Pe = 6, Ie = 8;
function We(l) {
  return Array.isArray(l) ? l.map((e) => String(e)) : l == null || l === "" ? [] : [String(l)];
}
function Ol(l, e) {
  const t = {
    folders: (l == null ? void 0 : l.folders) || [],
    items: (l == null ? void 0 : l.items) || []
  };
  return e ? {
    folders: Pt(t.folders),
    items: At(t.items)
  } : Nt(t);
}
function Nt(l) {
  return {
    ...l,
    folders: (l.folders || []).map((e) => Nt(e)),
    items: (l.items || []).map((e) => ({ ...e }))
  };
}
function Pt(l) {
  return l.map((e) => ({
    ...e,
    folders: Pt(e.folders || []),
    items: At(e.items || [])
  })).sort((e, t) => Ue(e).localeCompare(Ue(t), void 0, { sensitivity: "base" }));
}
function At(l) {
  return l.map((e) => ({ ...e })).sort((e, t) => x(e).localeCompare(x(t), void 0, { sensitivity: "base" }));
}
function Ue(l) {
  return String(l.name || l.path || "");
}
function x(l) {
  return String(l.name || l.path || l.value || "");
}
function Se(l) {
  return String(l.value || l.path || l.name || "");
}
function St(l, e = "") {
  const t = String(l.path || "");
  if (t) return t;
  const n = Ue(l);
  return e && n ? `${e}/${n}` : n;
}
function Mt(l, e = "") {
  const t = String(l.path || "");
  if (t) return t;
  const n = x(l);
  return e && n ? `${e}/${n}` : n;
}
function Zl(l) {
  return String(l || "").split("/").map((e) => e.trim()).filter(Boolean);
}
function Gl(l, e) {
  return l.map((t) => {
    const n = String(t.search_name || x(t)), r = String(t.search_path || ""), s = String(t.search_text || n), a = String(t.search_display || n);
    return {
      item: t,
      index: s.toLowerCase().indexOf(e),
      name: a,
      path: r
    };
  }).filter((t) => t.index > -1).sort((t, n) => t.index - n.index || t.name.localeCompare(n.name, void 0, { sensitivity: "base" }) || t.path.localeCompare(n.path, void 0, { sensitivity: "base" })).map((t) => t.item);
}
function Kl(l) {
  return l.replace(/\\/g, "/").replace(/\.[^/.]+$/, "");
}
function Ql(l) {
  return document.body.appendChild(l), {
    destroy() {
      l.remove();
    }
  };
}
function Wl(l, e, t) {
  let n, r, s, a, o, _, b, w, m, c, u, g, L, f, { elem_id: y = "" } = e, { elem_classes: T = [] } = e, { visible: j = !0 } = e, { value: v = [] } = e, { hierarchy: G = { folders: [], items: [] } } = e, { height: k = 10 } = e, { display_mode: F = "file" } = e, { breadcrumb_separator: I = " " } = e, { sort_hierarchy: N = !0 } = e, { search_empty_label: d = "No matches" } = e, { show_placeholder: D = !1 } = e, { label: C = "Hierarchy Selector" } = e, { info: me = void 0 } = e, { show_label: tt = !0 } = e, { container: Ze = !0 } = e, { scale: Me = null } = e, { min_width: je = void 0 } = e, { interactive: $ = !0 } = e, { gradio: M } = e, ce, ge, ne, ue, pe, he = !1, fe = !1, re = /* @__PURE__ */ new Set(), K = "", ie = null, be = null, lt = {}, nt = "";
  const Tt = `hierarchy-selector-${Math.random().toString(36).slice(2)}`;
  function Rt(i) {
    return De(i);
  }
  function qt(i, p, H) {
    const E = {};
    function q(Q, W = "") {
      for (const A of Q.items || []) E[Se(A)] = De(A, W);
      for (const A of Q.folders || []) q(A, St(A, W));
    }
    return q(i), E;
  }
  function Ot(i) {
    const p = Zl(i);
    return p.length ? p.join(I) : String(i || "");
  }
  function De(i, p = "") {
    const H = Mt(i, p) || String(i.value || "");
    return s ? Ot(H) : H;
  }
  function Zt(i, p = "") {
    return s ? De(i, p) : x(i);
  }
  function Gt(i, p, H) {
    const E = [];
    function q(Q, W = "") {
      for (const A of Q.items || []) {
        const Ce = Mt(A, W), ve = Ce.lastIndexOf("/"), Ke = ve > -1 ? Ce.slice(0, ve) : W;
        E.push({
          ...A,
          search_name: x(A),
          search_path: Ke,
          search_text: Zt(A, W),
          search_display: De(A, W)
        });
      }
      for (const A of Q.folders || []) q(A, St(A, W));
    }
    return q(i), E;
  }
  function Kt(i) {
    return lt[i] || Kl(i);
  }
  function ye(i, p = "change", H = void 0) {
    var E, q, Q;
    t(40, v = We(i)), (E = M == null ? void 0 : M.dispatch) == null || E.call(M, "input"), p === "select" && ((q = M == null ? void 0 : M.dispatch) == null || q.call(M, "select", H)), (Q = M == null ? void 0 : M.dispatch) == null || Q.call(M, "change"), Ve().then(ee);
  }
  function He(i) {
    if (!$) return;
    const p = Se(i);
    n.includes(p) ? ye(n.filter((H) => H !== p), "select", { value: p, selected: !1 }) : ye([...n, p], "select", { value: p, selected: !0 });
  }
  function Ge(i) {
    if (!$) return;
    const p = n.filter((H, E) => E !== i);
    ye(p, "select", {
      value: n[i],
      selected: !1
    });
  }
  function Qt() {
    if (!$ || n.length === 0) return;
    const i = n;
    t(10, K = ""), ye([], "select", { value: i, selected: !1 });
  }
  function Wt(i) {
    re.has(i) ? re.delete(i) : re.add(i), t(9, re = new Set(re)), Ve().then(ee);
  }
  function ee() {
    if (!ge || !he) return;
    const i = ge.getBoundingClientRect(), p = Math.max(Ne + we, pe ? Math.ceil(pe.scrollHeight + we) : Ne + we), H = c ? p : g, E = i.top - Ie - Pe, q = window.innerHeight - i.bottom - Ie - Pe, Q = E < H && q > E, W = Math.max(Ne + we, Q ? q : E), A = Math.min(H, W), Ce = Q ? i.bottom + Pe : Math.max(Ie, i.top - A - Pe), ve = Math.max(Ie, i.left), Ke = Math.max(240, Math.min(i.width, window.innerWidth - ve - Ie));
    t(23, nt = `top:${Math.round(Ce)}px;left:${Math.round(ve)}px;width:${Math.round(Ke)}px;height:${Math.round(A)}px;`);
  }
  function Le() {
    var p;
    if (!$) return;
    const i = fe;
    t(8, he = !0), t(52, fe = !0), i || (p = M == null ? void 0 : M.dispatch) == null || p.call(M, "focus"), Ve().then(() => {
      ne == null || ne.focus(), ee();
    });
  }
  function rt() {
    var i;
    !he && !fe || (t(8, he = !1), t(52, fe = !1), t(10, K = ""), (i = M == null ? void 0 : M.dispatch) == null || i.call(M, "blur"));
  }
  function Jt() {
    Le();
  }
  function Ut(i) {
    t(10, K = i.currentTarget.value), Le();
  }
  function Xt(i) {
    var p, H, E, q;
    !$ || i.target === ne || (H = (p = i.target).closest) != null && H.call(p, "button") || (q = (E = i.target).closest) != null && q.call(E, ".hierarchy-selector-chip") || (i.preventDefault(), Le());
  }
  function it(i) {
    ce != null && ce.contains(i.target) || ue != null && ue.contains(i.target) || rt();
  }
  function st(i, p) {
    var H;
    $ && (t(21, ie = i), (H = p.dataTransfer) == null || H.setData("text/plain", String(i)), p.dataTransfer && (p.dataTransfer.effectAllowed = "move"));
  }
  function Yt() {
    t(21, ie = null), t(22, be = null);
  }
  function at(i, p) {
    if (p.preventDefault(), ie === null || ie === i) return;
    const H = [...n], [E] = H.splice(ie, 1);
    H.splice(i, 0, E), t(21, ie = null), t(22, be = null), ye(H);
  }
  function xt(i) {
    i.key === "Escape" ? (i.preventDefault(), K ? t(10, K = "") : rt()) : i.key === "Enter" && _ && b.length ? (i.preventDefault(), He(b[0])) : i.key === "ArrowDown" ? (i.preventDefault(), Le()) : i.key === "Backspace" && !K && n.length && Ge(n.length - 1);
  }
  function $t() {
    return We(v);
  }
  El(() => {
    document.addEventListener("pointerdown", it, !0), window.addEventListener("resize", ee), window.addEventListener("scroll", ee, !0);
  }), Bl(() => {
    document.removeEventListener("pointerdown", it, !0), window.removeEventListener("resize", ee), window.removeEventListener("scroll", ee, !0);
  });
  const el = (i) => Ge(i), tl = (i, p) => st(i, p), ll = (i, p) => {
    p.preventDefault(), t(22, be = i);
  }, nl = () => t(22, be = null), rl = (i, p) => at(i, p);
  function il(i) {
    ke[i ? "unshift" : "push"](() => {
      ne = i, t(18, ne);
    });
  }
  function sl() {
    K = this.value, t(10, K);
  }
  function al(i) {
    ke[i ? "unshift" : "push"](() => {
      ge = i, t(17, ge);
    });
  }
  const ol = (i) => He(i), cl = (i, p) => {
    (p.key === "Enter" || p.key === " ") && (p.preventDefault(), He(i));
  };
  function ul(i) {
    ke[i ? "unshift" : "push"](() => {
      pe = i, t(20, pe);
    });
  }
  function hl(i) {
    ke[i ? "unshift" : "push"](() => {
      ue = i, t(19, ue);
    });
  }
  function fl(i) {
    ke[i ? "unshift" : "push"](() => {
      ce = i, t(16, ce);
    });
  }
  return l.$$set = (i) => {
    "elem_id" in i && t(0, y = i.elem_id), "elem_classes" in i && t(41, T = i.elem_classes), "visible" in i && t(1, j = i.visible), "value" in i && t(40, v = i.value), "hierarchy" in i && t(42, G = i.hierarchy), "height" in i && t(43, k = i.height), "display_mode" in i && t(44, F = i.display_mode), "breadcrumb_separator" in i && t(45, I = i.breadcrumb_separator), "sort_hierarchy" in i && t(46, N = i.sort_hierarchy), "search_empty_label" in i && t(2, d = i.search_empty_label), "show_placeholder" in i && t(3, D = i.show_placeholder), "label" in i && t(4, C = i.label), "info" in i && t(5, me = i.info), "show_label" in i && t(6, tt = i.show_label), "container" in i && t(47, Ze = i.container), "scale" in i && t(48, Me = i.scale), "min_width" in i && t(49, je = i.min_width), "interactive" in i && t(7, $ = i.interactive), "gradio" in i && t(50, M = i.gradio);
  }, l.$$.update = () => {
    l.$$.dirty[1] & /*value*/
    512 && t(13, n = We(v)), l.$$.dirty[1] & /*hierarchy, sort_hierarchy*/
    34816 && t(12, r = Ol(G, N)), l.$$.dirty[1] & /*display_mode*/
    8192 && t(15, s = F === "breadcrumb"), l.$$.dirty[0] & /*normalizedHierarchy, breadcrumbMode*/
    36864 | l.$$.dirty[1] & /*breadcrumb_separator*/
    16384 && (lt = qt(r)), l.$$.dirty[0] & /*normalizedHierarchy, breadcrumbMode*/
    36864 | l.$$.dirty[1] & /*breadcrumb_separator*/
    16384 && t(58, a = Gt(r)), l.$$.dirty[0] & /*searchQuery*/
    1024 && t(57, o = K.trim().toLowerCase()), l.$$.dirty[1] & /*searchTerm*/
    67108864 && t(14, _ = o.length > 0), l.$$.dirty[0] & /*searchMode*/
    16384 | l.$$.dirty[1] & /*flatItems, searchTerm*/
    201326592 && t(11, b = _ ? Gl(a, o) : []), l.$$.dirty[0] & /*elem_id*/
    1 && t(26, w = y ? `${y}-panel` : `${Tt}-panel`), l.$$.dirty[1] & /*height*/
    4096 && t(56, m = Number(k)), l.$$.dirty[1] & /*heightRows*/
    33554432 && t(54, c = m === 0), l.$$.dirty[1] & /*autoPanelHeight, heightRows*/
    41943040 && t(55, u = c ? 0 : Math.max(10, m || 10)), l.$$.dirty[1] & /*panelRows*/
    16777216 && t(53, g = u * Ne + we), l.$$.dirty[1] & /*container, focused, elem_classes*/
    2163712 && t(25, L = [
      "hierarchy-selector",
      Ze ? "hierarchy-selector-container" : "",
      fe ? "hierarchy-selector-focused" : "",
      ...Array.isArray(T) ? T : []
    ].filter(Boolean).join(" ")), l.$$.dirty[1] & /*scale, min_width*/
    393216 && t(24, f = [
      Me !== null ? `flex-grow:${Me};` : "",
      je !== void 0 ? `min-width:${je}px;` : ""
    ].join("")), l.$$.dirty[0] & /*open, selectedValue, searchQuery, expanded, normalizedHierarchy, searchResults*/
    16128 | l.$$.dirty[1] & /*autoPanelHeight, panelHeight*/
    12582912 && he && (n || K || re || r || b || c || g) && Ve().then(ee);
  }, [
    y,
    j,
    d,
    D,
    C,
    me,
    tt,
    $,
    he,
    re,
    K,
    b,
    r,
    n,
    _,
    s,
    ce,
    ge,
    ne,
    ue,
    pe,
    ie,
    be,
    nt,
    f,
    L,
    w,
    Rt,
    Kt,
    He,
    Ge,
    Qt,
    Wt,
    Jt,
    Ut,
    Xt,
    st,
    Yt,
    at,
    xt,
    v,
    T,
    G,
    k,
    F,
    I,
    N,
    Ze,
    Me,
    je,
    M,
    $t,
    fe,
    g,
    c,
    u,
    m,
    o,
    a,
    el,
    tl,
    ll,
    nl,
    rl,
    il,
    sl,
    al,
    ol,
    cl,
    ul,
    hl,
    fl
  ];
}
class Jl extends Il {
  constructor(e) {
    super(), Dl(
      this,
      e,
      Wl,
      ql,
      Ll,
      {
        elem_id: 0,
        elem_classes: 41,
        visible: 1,
        value: 40,
        hierarchy: 42,
        height: 43,
        display_mode: 44,
        breadcrumb_separator: 45,
        sort_hierarchy: 46,
        search_empty_label: 2,
        show_placeholder: 3,
        label: 4,
        info: 5,
        show_label: 6,
        container: 47,
        scale: 48,
        min_width: 49,
        interactive: 7,
        gradio: 50,
        get_value: 51
      },
      null,
      [-1, -1, -1]
    );
  }
  get elem_id() {
    return this.$$.ctx[0];
  }
  set elem_id(e) {
    this.$$set({ elem_id: e }), V();
  }
  get elem_classes() {
    return this.$$.ctx[41];
  }
  set elem_classes(e) {
    this.$$set({ elem_classes: e }), V();
  }
  get visible() {
    return this.$$.ctx[1];
  }
  set visible(e) {
    this.$$set({ visible: e }), V();
  }
  get value() {
    return this.$$.ctx[40];
  }
  set value(e) {
    this.$$set({ value: e }), V();
  }
  get hierarchy() {
    return this.$$.ctx[42];
  }
  set hierarchy(e) {
    this.$$set({ hierarchy: e }), V();
  }
  get height() {
    return this.$$.ctx[43];
  }
  set height(e) {
    this.$$set({ height: e }), V();
  }
  get display_mode() {
    return this.$$.ctx[44];
  }
  set display_mode(e) {
    this.$$set({ display_mode: e }), V();
  }
  get breadcrumb_separator() {
    return this.$$.ctx[45];
  }
  set breadcrumb_separator(e) {
    this.$$set({ breadcrumb_separator: e }), V();
  }
  get sort_hierarchy() {
    return this.$$.ctx[46];
  }
  set sort_hierarchy(e) {
    this.$$set({ sort_hierarchy: e }), V();
  }
  get search_empty_label() {
    return this.$$.ctx[2];
  }
  set search_empty_label(e) {
    this.$$set({ search_empty_label: e }), V();
  }
  get show_placeholder() {
    return this.$$.ctx[3];
  }
  set show_placeholder(e) {
    this.$$set({ show_placeholder: e }), V();
  }
  get label() {
    return this.$$.ctx[4];
  }
  set label(e) {
    this.$$set({ label: e }), V();
  }
  get info() {
    return this.$$.ctx[5];
  }
  set info(e) {
    this.$$set({ info: e }), V();
  }
  get show_label() {
    return this.$$.ctx[6];
  }
  set show_label(e) {
    this.$$set({ show_label: e }), V();
  }
  get container() {
    return this.$$.ctx[47];
  }
  set container(e) {
    this.$$set({ container: e }), V();
  }
  get scale() {
    return this.$$.ctx[48];
  }
  set scale(e) {
    this.$$set({ scale: e }), V();
  }
  get min_width() {
    return this.$$.ctx[49];
  }
  set min_width(e) {
    this.$$set({ min_width: e }), V();
  }
  get interactive() {
    return this.$$.ctx[7];
  }
  set interactive(e) {
    this.$$set({ interactive: e }), V();
  }
  get gradio() {
    return this.$$.ctx[50];
  }
  set gradio(e) {
    this.$$set({ gradio: e }), V();
  }
  get get_value() {
    return this.$$.ctx[51];
  }
}
export {
  Jl as default
};
