const {
  SvelteComponent: ol,
  append: A,
  attr: M,
  check_outros: It,
  create_component: cl,
  destroy_block: ul,
  destroy_component: hl,
  detach: de,
  element: ke,
  empty: Ft,
  ensure_array_like: ze,
  flush: U,
  group_outros: St,
  init: fl,
  insert: me,
  listen: Ee,
  mount_component: _l,
  outro_and_destroy_block: dl,
  run_all: Mt,
  safe_not_equal: ml,
  set_data: Dt,
  set_style: Ne,
  space: ie,
  svg_element: ee,
  text: Lt,
  toggle_class: Ae,
  transition_in: we,
  transition_out: Pe,
  update_keyed_each: lt
} = window.__gradio__svelte__internal;
function nt(l, e, t) {
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
function rt(l, e, t) {
  const n = l.slice();
  n[17] = e[t];
  const r = Ct(
    /*folder*/
    n[17]
  );
  return n[18] = r, n;
}
function it(l) {
  let e, t;
  return e = new zt({
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
      cl(e.$$.fragment);
    },
    m(n, r) {
      _l(e, n, r), t = !0;
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
      t || (we(e.$$.fragment, n), t = !0);
    },
    o(n) {
      Pe(e.$$.fragment, n), t = !1;
    },
    d(n) {
      hl(e, n);
    }
  };
}
function st(l, e) {
  let t, n, r, s, a, o, _, w, m, y = He(
    /*folder*/
    e[17]
  ) + "", u, c, g, H = (
    /*expanded*/
    e[3].has(
      /*path*/
      e[18]
    )
  ), f, b, P, D;
  function k() {
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
  let v = H && it(e);
  return {
    key: l,
    first: null,
    c() {
      t = ke("div"), n = ee("svg"), r = ee("path"), s = ie(), a = ee("svg"), o = ee("path"), _ = ee("path"), w = ie(), m = ke("span"), u = Lt(y), g = ie(), v && v.c(), f = Ft(), M(r, "d", "M6 4.5L10 8l-4 3.5"), M(n, "class", "hierarchy-twist svelte-1p05czg"), M(n, "viewBox", "0 0 16 16"), M(n, "aria-hidden", "true"), Ae(
        n,
        "hierarchy-twist-open",
        /*expanded*/
        e[3].has(
          /*path*/
          e[18]
        )
      ), M(o, "d", "M2.75 6.25h5.4l1.55 1.7h7.55c.55 0 1 .45 1 1v6.3c0 .55-.45 1-1 1H2.75c-.55 0-1-.45-1-1v-8c0-.55.45-1 1-1Z"), M(_, "d", "M2.25 7.95V5.6c0-.55.45-1 1-1h4.5l1.35 1.65"), M(a, "class", "hierarchy-icon hierarchy-folder-icon svelte-1p05czg"), M(a, "viewBox", "0 0 20 20"), M(a, "aria-hidden", "true"), M(m, "class", "hierarchy-name svelte-1p05czg"), M(t, "class", "hierarchy-row hierarchy-folder svelte-1p05czg"), M(t, "role", "button"), M(t, "tabindex", "0"), M(t, "title", c = He(
        /*folder*/
        e[17]
      )), Ne(t, "padding-left", `${/*depth*/
      e[2] * 18 + 6}px`), this.first = t;
    },
    m(F, I) {
      me(F, t, I), A(t, n), A(n, r), A(t, s), A(t, a), A(a, o), A(a, _), A(t, w), A(t, m), A(m, u), me(F, g, I), v && v.m(F, I), me(F, f, I), b = !0, P || (D = [
        Ee(t, "click", k),
        Ee(t, "keydown", G)
      ], P = !0);
    },
    p(F, I) {
      e = F, (!b || I & /*expanded, folderPath, folders*/
      9) && Ae(
        n,
        "hierarchy-twist-open",
        /*expanded*/
        e[3].has(
          /*path*/
          e[18]
        )
      ), (!b || I & /*folders*/
      1) && y !== (y = He(
        /*folder*/
        e[17]
      ) + "") && Dt(u, y), (!b || I & /*folders*/
      1 && c !== (c = He(
        /*folder*/
        e[17]
      ))) && M(t, "title", c), I & /*depth*/
      4 && Ne(t, "padding-left", `${/*depth*/
      e[2] * 18 + 6}px`), I & /*expanded, folders*/
      9 && (H = /*expanded*/
      e[3].has(
        /*path*/
        e[18]
      )), H ? v ? (v.p(e, I), I & /*expanded, folders*/
      9 && we(v, 1)) : (v = it(e), v.c(), we(v, 1), v.m(f.parentNode, f)) : v && (St(), Pe(v, 1, 1, () => {
        v = null;
      }), It());
    },
    i(F) {
      b || (we(v), b = !0);
    },
    o(F) {
      Pe(v), b = !1;
    },
    d(F) {
      F && (de(t), de(g), de(f)), v && v.d(F), P = !1, Mt(D);
    }
  };
}
function at(l, e) {
  let t, n, r, s, a, o, _, w, m = (
    /*labelForItem*/
    e[8](
      /*item*/
      e[13]
    ) + ""
  ), y, u, c, g, H, f;
  function b() {
    return (
      /*click_handler_1*/
      e[11](
        /*item*/
        e[13]
      )
    );
  }
  function P(...D) {
    return (
      /*keydown_handler_1*/
      e[12](
        /*item*/
        e[13],
        ...D
      )
    );
  }
  return {
    key: l,
    first: null,
    c() {
      t = ke("div"), n = ke("span"), r = ie(), s = ee("svg"), a = ee("path"), o = ee("path"), _ = ie(), w = ke("span"), y = Lt(m), u = ie(), M(n, "class", "hierarchy-twist-spacer svelte-1p05czg"), M(a, "d", "M5.25 2.75h6.05L15.75 7.2v10.05H5.25V2.75Z"), M(o, "d", "M11.25 2.95V7.3h4.3"), M(s, "class", "hierarchy-icon hierarchy-item-icon svelte-1p05czg"), M(s, "viewBox", "0 0 20 20"), M(s, "aria-hidden", "true"), M(w, "class", "hierarchy-name svelte-1p05czg"), M(t, "class", "hierarchy-row hierarchy-item svelte-1p05czg"), M(t, "title", c = /*labelForItem*/
      e[8](
        /*item*/
        e[13]
      )), M(t, "role", "button"), M(t, "tabindex", "0"), M(t, "aria-pressed", g = /*selected*/
      e[14]), Ae(
        t,
        "hierarchy-item-selected",
        /*selected*/
        e[14]
      ), Ne(t, "padding-left", `${/*depth*/
      e[2] * 18 + 6}px`), this.first = t;
    },
    m(D, k) {
      me(D, t, k), A(t, n), A(t, r), A(t, s), A(s, a), A(s, o), A(t, _), A(t, w), A(w, y), A(t, u), H || (f = [
        Ee(t, "click", b),
        Ee(t, "keydown", P)
      ], H = !0);
    },
    p(D, k) {
      e = D, k & /*labelForItem, items*/
      258 && m !== (m = /*labelForItem*/
      e[8](
        /*item*/
        e[13]
      ) + "") && Dt(y, m), k & /*labelForItem, items*/
      258 && c !== (c = /*labelForItem*/
      e[8](
        /*item*/
        e[13]
      )) && M(t, "title", c), k & /*value, valueForItem, items*/
      146 && g !== (g = /*selected*/
      e[14]) && M(t, "aria-pressed", g), k & /*value, valueForItem, items*/
      146 && Ae(
        t,
        "hierarchy-item-selected",
        /*selected*/
        e[14]
      ), k & /*depth*/
      4 && Ne(t, "padding-left", `${/*depth*/
      e[2] * 18 + 6}px`);
    },
    d(D) {
      D && de(t), H = !1, Mt(f);
    }
  };
}
function pl(l) {
  let e = [], t = /* @__PURE__ */ new Map(), n, r = [], s = /* @__PURE__ */ new Map(), a, o, _ = ze(
    /*folders*/
    l[0]
  );
  const w = (u) => Ct(
    /*folder*/
    u[17]
  );
  for (let u = 0; u < _.length; u += 1) {
    let c = rt(l, _, u), g = w(c);
    t.set(g, e[u] = st(g, c));
  }
  let m = ze(
    /*items*/
    l[1]
  );
  const y = (u) => (
    /*valueForItem*/
    u[7](
      /*item*/
      u[13]
    )
  );
  for (let u = 0; u < m.length; u += 1) {
    let c = nt(l, m, u), g = y(c);
    s.set(g, r[u] = at(g, c));
  }
  return {
    c() {
      for (let u = 0; u < e.length; u += 1)
        e[u].c();
      n = ie();
      for (let u = 0; u < r.length; u += 1)
        r[u].c();
      a = Ft();
    },
    m(u, c) {
      for (let g = 0; g < e.length; g += 1)
        e[g] && e[g].m(u, c);
      me(u, n, c);
      for (let g = 0; g < r.length; g += 1)
        r[g] && r[g].m(u, c);
      me(u, a, c), o = !0;
    },
    p(u, [c]) {
      c & /*folders, depth, expanded, value, toggleItem, toggleFolder, valueForItem, labelForItem, folderPath, folderLabel*/
      509 && (_ = ze(
        /*folders*/
        u[0]
      ), St(), e = lt(e, c, w, 1, u, _, t, n.parentNode, dl, st, n, rt), It()), c & /*labelForItem, items, value, valueForItem, depth, toggleItem*/
      438 && (m = ze(
        /*items*/
        u[1]
      ), r = lt(r, c, y, 1, u, m, s, a.parentNode, ul, at, a, nt));
    },
    i(u) {
      if (!o) {
        for (let c = 0; c < _.length; c += 1)
          we(e[c]);
        o = !0;
      }
    },
    o(u) {
      for (let c = 0; c < e.length; c += 1)
        Pe(e[c]);
      o = !1;
    },
    d(u) {
      u && (de(n), de(a));
      for (let c = 0; c < e.length; c += 1)
        e[c].d(u);
      for (let c = 0; c < r.length; c += 1)
        r[c].d(u);
    }
  };
}
function He(l) {
  return String(l.name || l.path || "");
}
function Ct(l) {
  return String(l.path || l.name || "");
}
function gl(l, e, t) {
  let { folders: n = [] } = e, { items: r = [] } = e, { depth: s = 0 } = e, { expanded: a } = e, { value: o } = e, { toggleItem: _ } = e, { toggleFolder: w } = e, { valueForItem: m } = e, { labelForItem: y } = e;
  const u = (f) => w(f), c = (f, b) => {
    (b.key === "Enter" || b.key === " ") && (b.preventDefault(), w(f));
  }, g = (f) => _(f), H = (f, b) => {
    (b.key === "Enter" || b.key === " ") && (b.preventDefault(), _(f));
  };
  return l.$$set = (f) => {
    "folders" in f && t(0, n = f.folders), "items" in f && t(1, r = f.items), "depth" in f && t(2, s = f.depth), "expanded" in f && t(3, a = f.expanded), "value" in f && t(4, o = f.value), "toggleItem" in f && t(5, _ = f.toggleItem), "toggleFolder" in f && t(6, w = f.toggleFolder), "valueForItem" in f && t(7, m = f.valueForItem), "labelForItem" in f && t(8, y = f.labelForItem);
  }, [
    n,
    r,
    s,
    a,
    o,
    _,
    w,
    m,
    y,
    u,
    c,
    g,
    H
  ];
}
class zt extends ol {
  constructor(e) {
    super(), fl(this, e, gl, pl, ml, {
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
  SvelteComponent: bl,
  action_destroyer: vl,
  append: S,
  attr: h,
  binding_callbacks: Be,
  check_outros: Ke,
  create_component: yl,
  destroy_block: kl,
  destroy_component: wl,
  destroy_each: Il,
  detach: O,
  element: E,
  empty: Qe,
  ensure_array_like: Te,
  flush: V,
  group_outros: We,
  init: Fl,
  insert: Z,
  listen: R,
  mount_component: Sl,
  noop: Ze,
  null_to_empty: ot,
  run_all: Je,
  safe_not_equal: Ml,
  set_data: ae,
  set_input_value: ct,
  space: J,
  stop_propagation: Ue,
  svg_element: Re,
  text: X,
  toggle_class: te,
  transition_in: Y,
  transition_out: se,
  update_keyed_each: Dl
} = window.__gradio__svelte__internal, { onDestroy: Ll, onMount: Cl, tick: je } = window.__gradio__svelte__internal;
function ut(l, e, t) {
  const n = l.slice();
  n[80] = e[t];
  const r = (
    /*selectedValue*/
    n[10].includes(Ie(
      /*item*/
      n[80]
    ))
  );
  return n[81] = r, n;
}
function ht(l, e, t) {
  const n = l.slice();
  return n[81] = e[t], n[85] = t, n;
}
function ft(l) {
  let e, t, n, r, s, a, o, _, w, m, y, u, c, g, H, f, b, P, D, k = (
    /*show_label*/
    l[6] && /*label*/
    l[4] && _t(l)
  ), G = Te(
    /*selectedValue*/
    l[10]
  ), v = [];
  for (let d = 0; d < G.length; d += 1)
    v[d] = dt(ht(l, G, d));
  let F = (
    /*selectedValue*/
    l[10].length > 0 && mt(l)
  ), I = (
    /*open*/
    l[8] && pt(l)
  ), T = (
    /*info*/
    l[5] && vt(l)
  );
  return {
    c() {
      e = E("div"), t = E("div"), k && k.c(), n = J(), r = E("div"), s = E("div");
      for (let d = 0; d < v.length; d += 1)
        v[d].c();
      a = J(), o = E("input"), y = J(), F && F.c(), g = J(), I && I.c(), H = J(), T && T.c(), h(o, "class", "hierarchy-selector-search-input svelte-14blvp4"), h(o, "type", "text"), h(o, "autocomplete", "off"), h(o, "spellcheck", "false"), o.disabled = _ = !/*interactive*/
      l[7], h(o, "tabindex", w = /*interactive*/
      l[7] ? 0 : -1), h(o, "placeholder", m = /*show_placeholder*/
      l[3] && /*selectedValue*/
      l[10].length === 0 ? (
        /*label*/
        l[4]
      ) : ""), h(
        o,
        "aria-label",
        /*label*/
        l[4]
      ), h(s, "class", "hierarchy-selector-chips svelte-14blvp4"), h(r, "class", "hierarchy-selector-input svelte-14blvp4"), h(r, "role", "combobox"), h(r, "tabindex", u = /*interactive*/
      l[7] ? 0 : -1), h(r, "aria-haspopup", "tree"), h(r, "aria-expanded", c = /*open*/
      l[8] ? "true" : "false"), h(
        r,
        "aria-controls",
        /*panelId*/
        l[25]
      ), te(r, "hierarchy-selector-disabled", !/*interactive*/
      l[7]), h(t, "class", "hierarchy-selector-field svelte-14blvp4"), h(
        e,
        "id",
        /*elem_id*/
        l[0]
      ), h(e, "class", f = ot(
        /*classes*/
        l[24]
      ) + " svelte-14blvp4"), h(
        e,
        "style",
        /*style*/
        l[23]
      );
    },
    m(d, C) {
      Z(d, e, C), S(e, t), k && k.m(t, null), S(t, n), S(t, r), S(r, s);
      for (let B = 0; B < v.length; B += 1)
        v[B] && v[B].m(s, null);
      S(s, a), S(s, o), l[60](o), ct(
        o,
        /*searchQuery*/
        l[9]
      ), S(r, y), F && F.m(r, null), l[62](r), S(t, g), I && I.m(t, null), S(e, H), T && T.m(e, null), l[66](e), b = !0, P || (D = [
        R(
          o,
          "input",
          /*input_input_handler*/
          l[61]
        ),
        R(
          o,
          "focus",
          /*onSearchFocus*/
          l[32]
        ),
        R(
          o,
          "input",
          /*onSearchInput*/
          l[33]
        ),
        R(o, "keydown", Ue(
          /*onInputKeydown*/
          l[38]
        )),
        R(
          r,
          "mousedown",
          /*onInputPointerDown*/
          l[34]
        ),
        R(
          r,
          "keydown",
          /*onInputKeydown*/
          l[38]
        )
      ], P = !0);
    },
    p(d, C) {
      if (/*show_label*/
      d[6] && /*label*/
      d[4] ? k ? k.p(d, C) : (k = _t(d), k.c(), k.m(t, n)) : k && (k.d(1), k = null), C[0] & /*interactive, draggedIndex, dragOverIndex, removeValue, displayValue, selectedValue*/
      672662656 | C[1] & /*onDragStart, onDragEnd, onDrop*/
      112) {
        G = Te(
          /*selectedValue*/
          d[10]
        );
        let B;
        for (B = 0; B < G.length; B += 1) {
          const oe = ht(d, G, B);
          v[B] ? v[B].p(oe, C) : (v[B] = dt(oe), v[B].c(), v[B].m(s, a));
        }
        for (; B < v.length; B += 1)
          v[B].d(1);
        v.length = G.length;
      }
      (!b || C[0] & /*interactive*/
      128 && _ !== (_ = !/*interactive*/
      d[7])) && (o.disabled = _), (!b || C[0] & /*interactive*/
      128 && w !== (w = /*interactive*/
      d[7] ? 0 : -1)) && h(o, "tabindex", w), (!b || C[0] & /*show_placeholder, selectedValue, label*/
      1048 && m !== (m = /*show_placeholder*/
      d[3] && /*selectedValue*/
      d[10].length === 0 ? (
        /*label*/
        d[4]
      ) : "")) && h(o, "placeholder", m), (!b || C[0] & /*label*/
      16) && h(
        o,
        "aria-label",
        /*label*/
        d[4]
      ), C[0] & /*searchQuery*/
      512 && o.value !== /*searchQuery*/
      d[9] && ct(
        o,
        /*searchQuery*/
        d[9]
      ), /*selectedValue*/
      d[10].length > 0 ? F ? F.p(d, C) : (F = mt(d), F.c(), F.m(r, null)) : F && (F.d(1), F = null), (!b || C[0] & /*interactive*/
      128 && u !== (u = /*interactive*/
      d[7] ? 0 : -1)) && h(r, "tabindex", u), (!b || C[0] & /*open*/
      256 && c !== (c = /*open*/
      d[8] ? "true" : "false")) && h(r, "aria-expanded", c), (!b || C[0] & /*panelId*/
      33554432) && h(
        r,
        "aria-controls",
        /*panelId*/
        d[25]
      ), (!b || C[0] & /*interactive*/
      128) && te(r, "hierarchy-selector-disabled", !/*interactive*/
      d[7]), /*open*/
      d[8] ? I ? (I.p(d, C), C[0] & /*open*/
      256 && Y(I, 1)) : (I = pt(d), I.c(), Y(I, 1), I.m(t, null)) : I && (We(), se(I, 1, 1, () => {
        I = null;
      }), Ke()), /*info*/
      d[5] ? T ? T.p(d, C) : (T = vt(d), T.c(), T.m(e, null)) : T && (T.d(1), T = null), (!b || C[0] & /*elem_id*/
      1) && h(
        e,
        "id",
        /*elem_id*/
        d[0]
      ), (!b || C[0] & /*classes*/
      16777216 && f !== (f = ot(
        /*classes*/
        d[24]
      ) + " svelte-14blvp4")) && h(e, "class", f), (!b || C[0] & /*style*/
      8388608) && h(
        e,
        "style",
        /*style*/
        d[23]
      );
    },
    i(d) {
      b || (Y(I), b = !0);
    },
    o(d) {
      se(I), b = !1;
    },
    d(d) {
      d && O(e), k && k.d(), Il(v, d), l[60](null), F && F.d(), l[62](null), I && I.d(), T && T.d(), l[66](null), P = !1, Je(D);
    }
  };
}
function _t(l) {
  let e, t;
  return {
    c() {
      e = E("span"), t = X(
        /*label*/
        l[4]
      ), h(e, "class", "hierarchy-selector-label svelte-14blvp4");
    },
    m(n, r) {
      Z(n, e, r), S(e, t);
    },
    p(n, r) {
      r[0] & /*label*/
      16 && ae(
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
function dt(l) {
  let e, t, n = (
    /*displayValue*/
    l[27](
      /*selected*/
      l[81]
    ) + ""
  ), r, s, a, o, _;
  function w() {
    return (
      /*click_handler*/
      l[55](
        /*index*/
        l[85]
      )
    );
  }
  function m(...c) {
    return (
      /*dragstart_handler*/
      l[56](
        /*index*/
        l[85],
        ...c
      )
    );
  }
  function y(...c) {
    return (
      /*dragover_handler*/
      l[57](
        /*index*/
        l[85],
        ...c
      )
    );
  }
  function u(...c) {
    return (
      /*drop_handler*/
      l[59](
        /*index*/
        l[85],
        ...c
      )
    );
  }
  return {
    c() {
      e = E("span"), t = E("span"), r = X(n), s = J(), a = E("button"), a.textContent = "x", h(t, "class", "hierarchy-selector-chip-text svelte-14blvp4"), h(a, "type", "button"), h(a, "class", "hierarchy-selector-remove svelte-14blvp4"), h(a, "aria-label", "Remove"), h(e, "class", "hierarchy-selector-chip svelte-14blvp4"), h(e, "role", "listitem"), h(
        e,
        "draggable",
        /*interactive*/
        l[7]
      ), te(
        e,
        "hierarchy-selector-chip-dragging",
        /*draggedIndex*/
        l[19] === /*index*/
        l[85]
      ), te(
        e,
        "hierarchy-selector-chip-over",
        /*dragOverIndex*/
        l[20] === /*index*/
        l[85]
      );
    },
    m(c, g) {
      Z(c, e, g), S(e, t), S(t, r), S(e, s), S(e, a), o || (_ = [
        R(a, "click", Ue(w)),
        R(e, "dragstart", m),
        R(
          e,
          "dragend",
          /*onDragEnd*/
          l[36]
        ),
        R(e, "dragover", y),
        R(
          e,
          "dragleave",
          /*dragleave_handler*/
          l[58]
        ),
        R(e, "drop", u)
      ], o = !0);
    },
    p(c, g) {
      l = c, g[0] & /*selectedValue*/
      1024 && n !== (n = /*displayValue*/
      l[27](
        /*selected*/
        l[81]
      ) + "") && ae(r, n), g[0] & /*interactive*/
      128 && h(
        e,
        "draggable",
        /*interactive*/
        l[7]
      ), g[0] & /*draggedIndex*/
      524288 && te(
        e,
        "hierarchy-selector-chip-dragging",
        /*draggedIndex*/
        l[19] === /*index*/
        l[85]
      ), g[0] & /*dragOverIndex*/
      1048576 && te(
        e,
        "hierarchy-selector-chip-over",
        /*dragOverIndex*/
        l[20] === /*index*/
        l[85]
      );
    },
    d(c) {
      c && O(e), o = !1, Je(_);
    }
  };
}
function mt(l) {
  let e, t, n;
  return {
    c() {
      e = E("button"), e.textContent = "x", h(e, "type", "button"), h(e, "class", "hierarchy-selector-clear svelte-14blvp4"), h(e, "aria-label", "Clear selection");
    },
    m(r, s) {
      Z(r, e, s), t || (n = R(e, "click", Ue(
        /*clearValues*/
        l[30]
      )), t = !0);
    },
    p: Ze,
    d(r) {
      r && O(e), t = !1, n();
    }
  };
}
function pt(l) {
  let e, t, n, r, s, a;
  const o = [Hl, zl], _ = [];
  function w(m, y) {
    return (
      /*searchMode*/
      m[11] ? 0 : 1
    );
  }
  return t = w(l), n = _[t] = o[t](l), {
    c() {
      e = E("div"), n.c(), h(
        e,
        "id",
        /*panelId*/
        l[25]
      ), h(e, "class", "hierarchy-selector-panel svelte-14blvp4"), h(
        e,
        "style",
        /*panelStyle*/
        l[21]
      );
    },
    m(m, y) {
      Z(m, e, y), _[t].m(e, null), l[65](e), r = !0, s || (a = vl(Ol.call(null, e)), s = !0);
    },
    p(m, y) {
      let u = t;
      t = w(m), t === u ? _[t].p(m, y) : (We(), se(_[u], 1, 1, () => {
        _[u] = null;
      }), Ke(), n = _[t], n ? n.p(m, y) : (n = _[t] = o[t](m), n.c()), Y(n, 1), n.m(e, null)), (!r || y[0] & /*panelId*/
      33554432) && h(
        e,
        "id",
        /*panelId*/
        m[25]
      ), (!r || y[0] & /*panelStyle*/
      2097152) && h(
        e,
        "style",
        /*panelStyle*/
        m[21]
      );
    },
    i(m) {
      r || (Y(n), r = !0);
    },
    o(m) {
      se(n), r = !1;
    },
    d(m) {
      m && O(e), _[t].d(), l[65](null), s = !1, a();
    }
  };
}
function zl(l) {
  let e, t;
  return e = new zt({
    props: {
      folders: (
        /*normalizedHierarchy*/
        l[13].folders || []
      ),
      items: (
        /*normalizedHierarchy*/
        l[13].items || []
      ),
      depth: 0,
      expanded: (
        /*expanded*/
        l[18]
      ),
      value: (
        /*selectedValue*/
        l[10]
      ),
      toggleItem: (
        /*toggleItem*/
        l[28]
      ),
      toggleFolder: (
        /*toggleFolder*/
        l[31]
      ),
      valueForItem: Ie,
      labelForItem: x
    }
  }), {
    c() {
      yl(e.$$.fragment);
    },
    m(n, r) {
      Sl(e, n, r), t = !0;
    },
    p(n, r) {
      const s = {};
      r[0] & /*normalizedHierarchy*/
      8192 && (s.folders = /*normalizedHierarchy*/
      n[13].folders || []), r[0] & /*normalizedHierarchy*/
      8192 && (s.items = /*normalizedHierarchy*/
      n[13].items || []), r[0] & /*expanded*/
      262144 && (s.expanded = /*expanded*/
      n[18]), r[0] & /*selectedValue*/
      1024 && (s.value = /*selectedValue*/
      n[10]), e.$set(s);
    },
    i(n) {
      t || (Y(e.$$.fragment, n), t = !0);
    },
    o(n) {
      se(e.$$.fragment, n), t = !1;
    },
    d(n) {
      wl(e, n);
    }
  };
}
function Hl(l) {
  let e;
  function t(s, a) {
    return (
      /*searchResults*/
      s[22].length ? Vl : Bl
    );
  }
  let n = t(l), r = n(l);
  return {
    c() {
      r.c(), e = Qe();
    },
    m(s, a) {
      r.m(s, a), Z(s, e, a);
    },
    p(s, a) {
      n === (n = t(s)) && r ? r.p(s, a) : (r.d(1), r = n(s), r && (r.c(), r.m(e.parentNode, e)));
    },
    i: Ze,
    o: Ze,
    d(s) {
      s && O(e), r.d(s);
    }
  };
}
function Bl(l) {
  let e, t;
  return {
    c() {
      e = E("div"), t = X(
        /*search_empty_label*/
        l[2]
      ), h(e, "class", "hierarchy-search-empty svelte-14blvp4");
    },
    m(n, r) {
      Z(n, e, r), S(e, t);
    },
    p(n, r) {
      r[0] & /*search_empty_label*/
      4 && ae(
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
function Vl(l) {
  let e = [], t = /* @__PURE__ */ new Map(), n, r = Te(
    /*searchResults*/
    l[22]
  );
  const s = (a) => Ie(
    /*item*/
    a[80]
  );
  for (let a = 0; a < r.length; a += 1) {
    let o = ut(l, r, a), _ = s(o);
    t.set(_, e[a] = bt(_, o));
  }
  return {
    c() {
      for (let a = 0; a < e.length; a += 1)
        e[a].c();
      n = Qe();
    },
    m(a, o) {
      for (let _ = 0; _ < e.length; _ += 1)
        e[_] && e[_].m(a, o);
      Z(a, n, o);
    },
    p(a, o) {
      o[0] & /*searchResults, selectedLabelForItem, selectedValue, toggleItem, breadcrumbMode*/
      339743744 && (r = Te(
        /*searchResults*/
        a[22]
      ), e = Dl(e, o, s, 1, a, r, t, n.parentNode, kl, bt, n, ut));
    },
    d(a) {
      a && O(n);
      for (let o = 0; o < e.length; o += 1)
        e[o].d(a);
    }
  };
}
function El(l) {
  let e, t, n = x(
    /*item*/
    l[80]
  ) + "", r, s, a = (
    /*item*/
    l[80].search_path && gt(l)
  );
  return {
    c() {
      e = E("span"), t = E("span"), r = X(n), s = J(), a && a.c(), h(t, "class", "hierarchy-search-name svelte-14blvp4"), h(e, "class", "hierarchy-search-label svelte-14blvp4");
    },
    m(o, _) {
      Z(o, e, _), S(e, t), S(t, r), S(e, s), a && a.m(e, null);
    },
    p(o, _) {
      _[0] & /*searchResults*/
      4194304 && n !== (n = x(
        /*item*/
        o[80]
      ) + "") && ae(r, n), /*item*/
      o[80].search_path ? a ? a.p(o, _) : (a = gt(o), a.c(), a.m(e, null)) : a && (a.d(1), a = null);
    },
    d(o) {
      o && O(e), a && a.d();
    }
  };
}
function Nl(l) {
  let e, t = (
    /*item*/
    l[80].search_display + ""
  ), n;
  return {
    c() {
      e = E("span"), n = X(t), h(e, "class", "hierarchy-search-label hierarchy-search-name svelte-14blvp4");
    },
    m(r, s) {
      Z(r, e, s), S(e, n);
    },
    p(r, s) {
      s[0] & /*searchResults*/
      4194304 && t !== (t = /*item*/
      r[80].search_display + "") && ae(n, t);
    },
    d(r) {
      r && O(e);
    }
  };
}
function gt(l) {
  let e, t, n = (
    /*item*/
    l[80].search_path + ""
  ), r, s;
  return {
    c() {
      e = E("span"), t = X("["), r = X(n), s = X("]"), h(e, "class", "hierarchy-search-path svelte-14blvp4");
    },
    m(a, o) {
      Z(a, e, o), S(e, t), S(e, r), S(e, s);
    },
    p(a, o) {
      o[0] & /*searchResults*/
      4194304 && n !== (n = /*item*/
      a[80].search_path + "") && ae(r, n);
    },
    d(a) {
      a && O(e);
    }
  };
}
function bt(l, e) {
  let t, n, r, s, a, o, _, w, m, y, u, c;
  function g(D, k) {
    return (
      /*breadcrumbMode*/
      D[12] ? Nl : El
    );
  }
  let H = g(e), f = H(e);
  function b() {
    return (
      /*click_handler_1*/
      e[63](
        /*item*/
        e[80]
      )
    );
  }
  function P(...D) {
    return (
      /*keydown_handler*/
      e[64](
        /*item*/
        e[80],
        ...D
      )
    );
  }
  return {
    key: l,
    first: null,
    c() {
      t = E("div"), n = E("span"), r = J(), s = Re("svg"), a = Re("path"), o = Re("path"), _ = J(), f.c(), w = J(), h(n, "class", "hierarchy-search-spacer svelte-14blvp4"), h(a, "d", "M5.25 2.75h6.05L15.75 7.2v10.05H5.25V2.75Z"), h(o, "d", "M11.25 2.95V7.3h4.3"), h(s, "class", "hierarchy-search-icon svelte-14blvp4"), h(s, "viewBox", "0 0 20 20"), h(s, "aria-hidden", "true"), h(t, "class", "hierarchy-search-row svelte-14blvp4"), h(t, "title", m = /*item*/
      e[80].search_display || /*selectedLabelForItem*/
      e[26](
        /*item*/
        e[80]
      )), h(t, "role", "button"), h(t, "tabindex", "0"), h(t, "aria-pressed", y = /*selected*/
      e[81]), te(
        t,
        "hierarchy-search-row-selected",
        /*selected*/
        e[81]
      ), this.first = t;
    },
    m(D, k) {
      Z(D, t, k), S(t, n), S(t, r), S(t, s), S(s, a), S(s, o), S(t, _), f.m(t, null), S(t, w), u || (c = [
        R(t, "click", b),
        R(t, "keydown", P)
      ], u = !0);
    },
    p(D, k) {
      e = D, H === (H = g(e)) && f ? f.p(e, k) : (f.d(1), f = H(e), f && (f.c(), f.m(t, w))), k[0] & /*searchResults*/
      4194304 && m !== (m = /*item*/
      e[80].search_display || /*selectedLabelForItem*/
      e[26](
        /*item*/
        e[80]
      )) && h(t, "title", m), k[0] & /*selectedValue, searchResults*/
      4195328 && y !== (y = /*selected*/
      e[81]) && h(t, "aria-pressed", y), k[0] & /*selectedValue, searchResults*/
      4195328 && te(
        t,
        "hierarchy-search-row-selected",
        /*selected*/
        e[81]
      );
    },
    d(D) {
      D && O(t), f.d(), u = !1, Je(c);
    }
  };
}
function vt(l) {
  let e, t;
  return {
    c() {
      e = E("div"), t = X(
        /*info*/
        l[5]
      ), h(e, "class", "hierarchy-selector-info svelte-14blvp4");
    },
    m(n, r) {
      Z(n, e, r), S(e, t);
    },
    p(n, r) {
      r[0] & /*info*/
      32 && ae(
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
function Al(l) {
  let e, t, n = (
    /*visible*/
    l[1] && ft(l)
  );
  return {
    c() {
      n && n.c(), e = Qe();
    },
    m(r, s) {
      n && n.m(r, s), Z(r, e, s), t = !0;
    },
    p(r, s) {
      /*visible*/
      r[1] ? n ? (n.p(r, s), s[0] & /*visible*/
      2 && Y(n, 1)) : (n = ft(r), n.c(), Y(n, 1), n.m(e.parentNode, e)) : n && (We(), se(n, 1, 1, () => {
        n = null;
      }), Ke());
    },
    i(r) {
      t || (Y(n), t = !0);
    },
    o(r) {
      se(n), t = !1;
    },
    d(r) {
      r && O(e), n && n.d(r);
    }
  };
}
const yt = 32, Pl = 8, Ve = 6, ye = 8;
function Oe(l) {
  return Array.isArray(l) ? l.map((e) => String(e)) : l == null || l === "" ? [] : [String(l)];
}
function Tl(l, e) {
  const t = {
    folders: (l == null ? void 0 : l.folders) || [],
    items: (l == null ? void 0 : l.items) || []
  };
  return e ? {
    folders: Bt(t.folders),
    items: Vt(t.items)
  } : Ht(t);
}
function Ht(l) {
  return {
    ...l,
    folders: (l.folders || []).map((e) => Ht(e)),
    items: (l.items || []).map((e) => ({ ...e }))
  };
}
function Bt(l) {
  return l.map((e) => ({
    ...e,
    folders: Bt(e.folders || []),
    items: Vt(e.items || [])
  })).sort((e, t) => Ge(e).localeCompare(Ge(t), void 0, { sensitivity: "base" }));
}
function Vt(l) {
  return l.map((e) => ({ ...e })).sort((e, t) => x(e).localeCompare(x(t), void 0, { sensitivity: "base" }));
}
function Ge(l) {
  return String(l.name || l.path || "");
}
function x(l) {
  return String(l.name || l.path || l.value || "");
}
function Ie(l) {
  return String(l.value || l.path || l.name || "");
}
function kt(l, e = "") {
  const t = String(l.path || "");
  if (t) return t;
  const n = Ge(l);
  return e && n ? `${e}/${n}` : n;
}
function wt(l, e = "") {
  const t = String(l.path || "");
  if (t) return t;
  const n = x(l);
  return e && n ? `${e}/${n}` : n;
}
function ql(l) {
  return String(l || "").split("/").map((e) => e.trim()).filter(Boolean);
}
function Rl(l, e) {
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
function jl(l) {
  return l.replace(/\\/g, "/").replace(/\.[^/.]+$/, "");
}
function Ol(l) {
  return document.body.appendChild(l), {
    destroy() {
      l.remove();
    }
  };
}
function Zl(l, e, t) {
  let n, r, s, a, o, _, w, m, y, u, c, g, { elem_id: H = "" } = e, { elem_classes: f = [] } = e, { visible: b = !0 } = e, { value: P = [] } = e, { hierarchy: D = { folders: [], items: [] } } = e, { height: k = 10 } = e, { display_mode: G = "file" } = e, { breadcrumb_separator: v = " " } = e, { sort_hierarchy: F = !0 } = e, { search_empty_label: I = "No matches" } = e, { show_placeholder: T = !1 } = e, { label: d = "Hierarchy Selector" } = e, { info: C = void 0 } = e, { show_label: B = !0 } = e, { container: oe = !0 } = e, { scale: Fe = null } = e, { min_width: Se = void 0 } = e, { interactive: $ = !0 } = e, { gradio: L } = e, ce, pe, le, ue, he = !1, fe = !1, _e = /* @__PURE__ */ new Set(), K = "", ne = null, ge = null, Xe = {}, Ye = "";
  const Et = `hierarchy-selector-${Math.random().toString(36).slice(2)}`;
  function Nt(i) {
    return Me(i);
  }
  function At(i, p, z) {
    const N = {};
    function j(W, Q = "") {
      for (const q of W.items || []) N[Ie(q)] = Me(q, Q);
      for (const q of W.folders || []) j(q, kt(q, Q));
    }
    return j(i), N;
  }
  function Pt(i) {
    const p = ql(i);
    return p.length ? p.join(v) : String(i || "");
  }
  function Me(i, p = "") {
    const z = wt(i, p) || String(i.value || "");
    return s ? Pt(z) : z;
  }
  function Tt(i, p = "") {
    return s ? Me(i, p) : x(i);
  }
  function qt(i, p, z) {
    const N = [];
    function j(W, Q = "") {
      for (const q of W.items || []) {
        const ve = wt(q, Q), Ce = ve.lastIndexOf("/"), al = Ce > -1 ? ve.slice(0, Ce) : Q;
        N.push({
          ...q,
          search_name: x(q),
          search_path: al,
          search_text: Tt(q, Q),
          search_display: Me(q, Q)
        });
      }
      for (const q of W.folders || []) j(q, kt(q, Q));
    }
    return j(i), N;
  }
  function Rt(i) {
    return Xe[i] || jl(i);
  }
  function be(i, p = "change", z = void 0) {
    var N, j, W;
    t(39, P = Oe(i)), (N = L == null ? void 0 : L.dispatch) == null || N.call(L, "input"), p === "select" && ((j = L == null ? void 0 : L.dispatch) == null || j.call(L, "select", z)), (W = L == null ? void 0 : L.dispatch) == null || W.call(L, "change"), je().then(re);
  }
  function De(i) {
    if (!$) return;
    const p = Ie(i);
    n.includes(p) ? be(n.filter((z) => z !== p), "select", { value: p, selected: !1 }) : be([...n, p], "select", { value: p, selected: !0 });
  }
  function qe(i) {
    if (!$) return;
    const p = n.filter((z, N) => N !== i);
    be(p, "select", {
      value: n[i],
      selected: !1
    });
  }
  function jt() {
    if (!$ || n.length === 0) return;
    const i = n;
    t(9, K = ""), be([], "select", { value: i, selected: !1 });
  }
  function Ot(i) {
    _e.has(i) ? _e.delete(i) : _e.add(i), t(18, _e = new Set(_e));
  }
  function re() {
    if (!pe || !he) return;
    const i = pe.getBoundingClientRect(), p = u, z = i.top - ye - Ve, N = window.innerHeight - i.bottom - ye - Ve, j = z < p && N > z, W = Math.max(yt * 4, j ? N : z), Q = Math.min(p, W), q = j ? i.bottom + Ve : Math.max(ye, i.top - Q - Ve), ve = Math.max(ye, i.left), Ce = Math.max(240, Math.min(i.width, window.innerWidth - ve - ye));
    t(21, Ye = `top:${Math.round(q)}px;left:${Math.round(ve)}px;width:${Math.round(Ce)}px;height:${Math.round(Q)}px;`);
  }
  function Le() {
    var p;
    if (!$) return;
    const i = fe;
    t(8, he = !0), t(51, fe = !0), i || (p = L == null ? void 0 : L.dispatch) == null || p.call(L, "focus"), je().then(() => {
      le == null || le.focus(), re();
    });
  }
  function xe() {
    var i;
    !he && !fe || (t(8, he = !1), t(51, fe = !1), t(9, K = ""), (i = L == null ? void 0 : L.dispatch) == null || i.call(L, "blur"));
  }
  function Zt() {
    Le();
  }
  function Gt(i) {
    t(9, K = i.currentTarget.value), Le();
  }
  function Kt(i) {
    var p, z, N, j;
    !$ || i.target === le || (z = (p = i.target).closest) != null && z.call(p, "button") || (j = (N = i.target).closest) != null && j.call(N, ".hierarchy-selector-chip") || (i.preventDefault(), Le());
  }
  function $e(i) {
    ce != null && ce.contains(i.target) || ue != null && ue.contains(i.target) || xe();
  }
  function et(i, p) {
    var z;
    $ && (t(19, ne = i), (z = p.dataTransfer) == null || z.setData("text/plain", String(i)), p.dataTransfer && (p.dataTransfer.effectAllowed = "move"));
  }
  function Qt() {
    t(19, ne = null), t(20, ge = null);
  }
  function tt(i, p) {
    if (p.preventDefault(), ne === null || ne === i) return;
    const z = [...n], [N] = z.splice(ne, 1);
    z.splice(i, 0, N), t(19, ne = null), t(20, ge = null), be(z);
  }
  function Wt(i) {
    i.key === "Escape" ? (i.preventDefault(), K ? t(9, K = "") : xe()) : i.key === "Enter" && _ && w.length ? (i.preventDefault(), De(w[0])) : i.key === "ArrowDown" ? (i.preventDefault(), Le()) : i.key === "Backspace" && !K && n.length && qe(n.length - 1);
  }
  function Jt() {
    return Oe(P);
  }
  Cl(() => {
    document.addEventListener("pointerdown", $e, !0), window.addEventListener("resize", re), window.addEventListener("scroll", re, !0);
  }), Ll(() => {
    document.removeEventListener("pointerdown", $e, !0), window.removeEventListener("resize", re), window.removeEventListener("scroll", re, !0);
  });
  const Ut = (i) => qe(i), Xt = (i, p) => et(i, p), Yt = (i, p) => {
    p.preventDefault(), t(20, ge = i);
  }, xt = () => t(20, ge = null), $t = (i, p) => tt(i, p);
  function el(i) {
    Be[i ? "unshift" : "push"](() => {
      le = i, t(16, le);
    });
  }
  function tl() {
    K = this.value, t(9, K);
  }
  function ll(i) {
    Be[i ? "unshift" : "push"](() => {
      pe = i, t(15, pe);
    });
  }
  const nl = (i) => De(i), rl = (i, p) => {
    (p.key === "Enter" || p.key === " ") && (p.preventDefault(), De(i));
  };
  function il(i) {
    Be[i ? "unshift" : "push"](() => {
      ue = i, t(17, ue);
    });
  }
  function sl(i) {
    Be[i ? "unshift" : "push"](() => {
      ce = i, t(14, ce);
    });
  }
  return l.$$set = (i) => {
    "elem_id" in i && t(0, H = i.elem_id), "elem_classes" in i && t(40, f = i.elem_classes), "visible" in i && t(1, b = i.visible), "value" in i && t(39, P = i.value), "hierarchy" in i && t(41, D = i.hierarchy), "height" in i && t(42, k = i.height), "display_mode" in i && t(43, G = i.display_mode), "breadcrumb_separator" in i && t(44, v = i.breadcrumb_separator), "sort_hierarchy" in i && t(45, F = i.sort_hierarchy), "search_empty_label" in i && t(2, I = i.search_empty_label), "show_placeholder" in i && t(3, T = i.show_placeholder), "label" in i && t(4, d = i.label), "info" in i && t(5, C = i.info), "show_label" in i && t(6, B = i.show_label), "container" in i && t(46, oe = i.container), "scale" in i && t(47, Fe = i.scale), "min_width" in i && t(48, Se = i.min_width), "interactive" in i && t(7, $ = i.interactive), "gradio" in i && t(49, L = i.gradio);
  }, l.$$.update = () => {
    l.$$.dirty[1] & /*value*/
    256 && t(10, n = Oe(P)), l.$$.dirty[1] & /*hierarchy, sort_hierarchy*/
    17408 && t(13, r = Tl(D, F)), l.$$.dirty[1] & /*display_mode*/
    4096 && t(12, s = G === "breadcrumb"), l.$$.dirty[0] & /*normalizedHierarchy, breadcrumbMode*/
    12288 | l.$$.dirty[1] & /*breadcrumb_separator*/
    8192 && (Xe = At(r)), l.$$.dirty[0] & /*normalizedHierarchy, breadcrumbMode*/
    12288 | l.$$.dirty[1] & /*breadcrumb_separator*/
    8192 && t(54, a = qt(r)), l.$$.dirty[0] & /*searchQuery*/
    512 && t(53, o = K.trim().toLowerCase()), l.$$.dirty[1] & /*searchTerm*/
    4194304 && t(11, _ = o.length > 0), l.$$.dirty[0] & /*searchMode*/
    2048 | l.$$.dirty[1] & /*flatItems, searchTerm*/
    12582912 && t(22, w = _ ? Rl(a, o) : []), l.$$.dirty[0] & /*elem_id*/
    1 && t(25, m = H ? `${H}-panel` : `${Et}-panel`), l.$$.dirty[1] & /*height*/
    2048 && t(52, y = Math.max(10, Number(k) || 10)), l.$$.dirty[1] & /*panelRows*/
    2097152 && (u = y * yt + Pl), l.$$.dirty[1] & /*container, focused, elem_classes*/
    1081856 && t(24, c = [
      "hierarchy-selector",
      oe ? "hierarchy-selector-container" : "",
      fe ? "hierarchy-selector-focused" : "",
      ...Array.isArray(f) ? f : []
    ].filter(Boolean).join(" ")), l.$$.dirty[1] & /*scale, min_width*/
    196608 && t(23, g = [
      Fe !== null ? `flex-grow:${Fe};` : "",
      Se !== void 0 ? `min-width:${Se}px;` : ""
    ].join("")), l.$$.dirty[0] & /*open, selectedValue, searchQuery*/
    1792 && he && (n || K) && je().then(re);
  }, [
    H,
    b,
    I,
    T,
    d,
    C,
    B,
    $,
    he,
    K,
    n,
    _,
    s,
    r,
    ce,
    pe,
    le,
    ue,
    _e,
    ne,
    ge,
    Ye,
    w,
    g,
    c,
    m,
    Nt,
    Rt,
    De,
    qe,
    jt,
    Ot,
    Zt,
    Gt,
    Kt,
    et,
    Qt,
    tt,
    Wt,
    P,
    f,
    D,
    k,
    G,
    v,
    F,
    oe,
    Fe,
    Se,
    L,
    Jt,
    fe,
    y,
    o,
    a,
    Ut,
    Xt,
    Yt,
    xt,
    $t,
    el,
    tl,
    ll,
    nl,
    rl,
    il,
    sl
  ];
}
class Gl extends bl {
  constructor(e) {
    super(), Fl(
      this,
      e,
      Zl,
      Al,
      Ml,
      {
        elem_id: 0,
        elem_classes: 40,
        visible: 1,
        value: 39,
        hierarchy: 41,
        height: 42,
        display_mode: 43,
        breadcrumb_separator: 44,
        sort_hierarchy: 45,
        search_empty_label: 2,
        show_placeholder: 3,
        label: 4,
        info: 5,
        show_label: 6,
        container: 46,
        scale: 47,
        min_width: 48,
        interactive: 7,
        gradio: 49,
        get_value: 50
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
    return this.$$.ctx[40];
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
    return this.$$.ctx[39];
  }
  set value(e) {
    this.$$set({ value: e }), V();
  }
  get hierarchy() {
    return this.$$.ctx[41];
  }
  set hierarchy(e) {
    this.$$set({ hierarchy: e }), V();
  }
  get height() {
    return this.$$.ctx[42];
  }
  set height(e) {
    this.$$set({ height: e }), V();
  }
  get display_mode() {
    return this.$$.ctx[43];
  }
  set display_mode(e) {
    this.$$set({ display_mode: e }), V();
  }
  get breadcrumb_separator() {
    return this.$$.ctx[44];
  }
  set breadcrumb_separator(e) {
    this.$$set({ breadcrumb_separator: e }), V();
  }
  get sort_hierarchy() {
    return this.$$.ctx[45];
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
    return this.$$.ctx[46];
  }
  set container(e) {
    this.$$set({ container: e }), V();
  }
  get scale() {
    return this.$$.ctx[47];
  }
  set scale(e) {
    this.$$set({ scale: e }), V();
  }
  get min_width() {
    return this.$$.ctx[48];
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
    return this.$$.ctx[49];
  }
  set gradio(e) {
    this.$$set({ gradio: e }), V();
  }
  get get_value() {
    return this.$$.ctx[50];
  }
}
export {
  Gl as default
};
