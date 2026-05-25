const {
  SvelteComponent: kt,
  append: D,
  attr: y,
  check_outros: Je,
  create_component: It,
  destroy_block: Ft,
  destroy_component: Mt,
  detach: U,
  element: te,
  empty: Qe,
  ensure_array_like: se,
  flush: V,
  group_outros: Ue,
  init: St,
  insert: X,
  listen: ue,
  mount_component: jt,
  outro_and_destroy_block: zt,
  run_all: Xe,
  safe_not_equal: Dt,
  set_data: Ye,
  set_style: fe,
  space: q,
  svg_element: B,
  text: $e,
  toggle_class: ce,
  transition_in: le,
  transition_out: he,
  update_keyed_each: Le
} = window.__gradio__svelte__internal;
function He(t, e, l) {
  const n = t.slice();
  n[13] = e[l];
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
function Ce(t, e, l) {
  const n = t.slice();
  n[17] = e[l];
  const i = xe(
    /*folder*/
    n[17]
  );
  return n[18] = i, n;
}
function Ee(t) {
  let e, l;
  return e = new et({
    props: {
      folders: (
        /*folder*/
        t[17].folders || []
      ),
      items: (
        /*folder*/
        t[17].items || []
      ),
      depth: (
        /*depth*/
        t[2] + 1
      ),
      expanded: (
        /*expanded*/
        t[3]
      ),
      value: (
        /*value*/
        t[4]
      ),
      toggleItem: (
        /*toggleItem*/
        t[5]
      ),
      toggleFolder: (
        /*toggleFolder*/
        t[6]
      ),
      valueForItem: (
        /*valueForItem*/
        t[7]
      ),
      labelForItem: (
        /*labelForItem*/
        t[8]
      )
    }
  }), {
    c() {
      It(e.$$.fragment);
    },
    m(n, i) {
      jt(e, n, i), l = !0;
    },
    p(n, i) {
      const u = {};
      i & /*folders*/
      1 && (u.folders = /*folder*/
      n[17].folders || []), i & /*folders*/
      1 && (u.items = /*folder*/
      n[17].items || []), i & /*depth*/
      4 && (u.depth = /*depth*/
      n[2] + 1), i & /*expanded*/
      8 && (u.expanded = /*expanded*/
      n[3]), i & /*value*/
      16 && (u.value = /*value*/
      n[4]), i & /*toggleItem*/
      32 && (u.toggleItem = /*toggleItem*/
      n[5]), i & /*toggleFolder*/
      64 && (u.toggleFolder = /*toggleFolder*/
      n[6]), i & /*valueForItem*/
      128 && (u.valueForItem = /*valueForItem*/
      n[7]), i & /*labelForItem*/
      256 && (u.labelForItem = /*labelForItem*/
      n[8]), e.$set(u);
    },
    i(n) {
      l || (le(e.$$.fragment, n), l = !0);
    },
    o(n) {
      he(e.$$.fragment, n), l = !1;
    },
    d(n) {
      Mt(e, n);
    }
  };
}
function Pe(t, e) {
  let l, n, i, u, h, m, p, F, w, v = oe(
    /*folder*/
    e[17]
  ) + "", o, f, a, g = (
    /*expanded*/
    e[3].has(
      /*path*/
      e[18]
    )
  ), d, _, I, b;
  function c() {
    return (
      /*click_handler*/
      e[9](
        /*path*/
        e[18]
      )
    );
  }
  function M(...j) {
    return (
      /*keydown_handler*/
      e[10](
        /*path*/
        e[18],
        ...j
      )
    );
  }
  let s = g && Ee(e);
  return {
    key: t,
    first: null,
    c() {
      l = te("div"), n = B("svg"), i = B("path"), u = q(), h = B("svg"), m = B("path"), p = B("path"), F = q(), w = te("span"), o = $e(v), a = q(), s && s.c(), d = Qe(), y(i, "d", "M6 4.5L10 8l-4 3.5"), y(n, "class", "hierarchy-twist svelte-1p05czg"), y(n, "viewBox", "0 0 16 16"), y(n, "aria-hidden", "true"), ce(
        n,
        "hierarchy-twist-open",
        /*expanded*/
        e[3].has(
          /*path*/
          e[18]
        )
      ), y(m, "d", "M2.75 6.25h5.4l1.55 1.7h7.55c.55 0 1 .45 1 1v6.3c0 .55-.45 1-1 1H2.75c-.55 0-1-.45-1-1v-8c0-.55.45-1 1-1Z"), y(p, "d", "M2.25 7.95V5.6c0-.55.45-1 1-1h4.5l1.35 1.65"), y(h, "class", "hierarchy-icon hierarchy-folder-icon svelte-1p05czg"), y(h, "viewBox", "0 0 20 20"), y(h, "aria-hidden", "true"), y(w, "class", "hierarchy-name svelte-1p05czg"), y(l, "class", "hierarchy-row hierarchy-folder svelte-1p05czg"), y(l, "role", "button"), y(l, "tabindex", "0"), y(l, "title", f = oe(
        /*folder*/
        e[17]
      )), fe(l, "padding-left", `${/*depth*/
      e[2] * 18 + 6}px`), this.first = l;
    },
    m(j, z) {
      X(j, l, z), D(l, n), D(n, i), D(l, u), D(l, h), D(h, m), D(h, p), D(l, F), D(l, w), D(w, o), X(j, a, z), s && s.m(j, z), X(j, d, z), _ = !0, I || (b = [
        ue(l, "click", c),
        ue(l, "keydown", M)
      ], I = !0);
    },
    p(j, z) {
      e = j, (!_ || z & /*expanded, folderPath, folders*/
      9) && ce(
        n,
        "hierarchy-twist-open",
        /*expanded*/
        e[3].has(
          /*path*/
          e[18]
        )
      ), (!_ || z & /*folders*/
      1) && v !== (v = oe(
        /*folder*/
        e[17]
      ) + "") && Ye(o, v), (!_ || z & /*folders*/
      1 && f !== (f = oe(
        /*folder*/
        e[17]
      ))) && y(l, "title", f), z & /*depth*/
      4 && fe(l, "padding-left", `${/*depth*/
      e[2] * 18 + 6}px`), z & /*expanded, folders*/
      9 && (g = /*expanded*/
      e[3].has(
        /*path*/
        e[18]
      )), g ? s ? (s.p(e, z), z & /*expanded, folders*/
      9 && le(s, 1)) : (s = Ee(e), s.c(), le(s, 1), s.m(d.parentNode, d)) : s && (Ue(), he(s, 1, 1, () => {
        s = null;
      }), Je());
    },
    i(j) {
      _ || (le(s), _ = !0);
    },
    o(j) {
      he(s), _ = !1;
    },
    d(j) {
      j && (U(l), U(a), U(d)), s && s.d(j), I = !1, Xe(b);
    }
  };
}
function Ve(t, e) {
  let l, n, i, u, h, m, p, F, w = (
    /*labelForItem*/
    e[8](
      /*item*/
      e[13]
    ) + ""
  ), v, o, f, a, g, d;
  function _() {
    return (
      /*click_handler_1*/
      e[11](
        /*item*/
        e[13]
      )
    );
  }
  function I(...b) {
    return (
      /*keydown_handler_1*/
      e[12](
        /*item*/
        e[13],
        ...b
      )
    );
  }
  return {
    key: t,
    first: null,
    c() {
      l = te("div"), n = te("span"), i = q(), u = B("svg"), h = B("path"), m = B("path"), p = q(), F = te("span"), v = $e(w), o = q(), y(n, "class", "hierarchy-twist-spacer svelte-1p05czg"), y(h, "d", "M5.25 2.75h6.05L15.75 7.2v10.05H5.25V2.75Z"), y(m, "d", "M11.25 2.95V7.3h4.3"), y(u, "class", "hierarchy-icon hierarchy-item-icon svelte-1p05czg"), y(u, "viewBox", "0 0 20 20"), y(u, "aria-hidden", "true"), y(F, "class", "hierarchy-name svelte-1p05czg"), y(l, "class", "hierarchy-row hierarchy-item svelte-1p05czg"), y(l, "title", f = /*labelForItem*/
      e[8](
        /*item*/
        e[13]
      )), y(l, "role", "button"), y(l, "tabindex", "0"), y(l, "aria-pressed", a = /*selected*/
      e[14]), ce(
        l,
        "hierarchy-item-selected",
        /*selected*/
        e[14]
      ), fe(l, "padding-left", `${/*depth*/
      e[2] * 18 + 6}px`), this.first = l;
    },
    m(b, c) {
      X(b, l, c), D(l, n), D(l, i), D(l, u), D(u, h), D(u, m), D(l, p), D(l, F), D(F, v), D(l, o), g || (d = [
        ue(l, "click", _),
        ue(l, "keydown", I)
      ], g = !0);
    },
    p(b, c) {
      e = b, c & /*labelForItem, items*/
      258 && w !== (w = /*labelForItem*/
      e[8](
        /*item*/
        e[13]
      ) + "") && Ye(v, w), c & /*labelForItem, items*/
      258 && f !== (f = /*labelForItem*/
      e[8](
        /*item*/
        e[13]
      )) && y(l, "title", f), c & /*value, valueForItem, items*/
      146 && a !== (a = /*selected*/
      e[14]) && y(l, "aria-pressed", a), c & /*value, valueForItem, items*/
      146 && ce(
        l,
        "hierarchy-item-selected",
        /*selected*/
        e[14]
      ), c & /*depth*/
      4 && fe(l, "padding-left", `${/*depth*/
      e[2] * 18 + 6}px`);
    },
    d(b) {
      b && U(l), g = !1, Xe(d);
    }
  };
}
function Lt(t) {
  let e = [], l = /* @__PURE__ */ new Map(), n, i = [], u = /* @__PURE__ */ new Map(), h, m, p = se(
    /*folders*/
    t[0]
  );
  const F = (o) => xe(
    /*folder*/
    o[17]
  );
  for (let o = 0; o < p.length; o += 1) {
    let f = Ce(t, p, o), a = F(f);
    l.set(a, e[o] = Pe(a, f));
  }
  let w = se(
    /*items*/
    t[1]
  );
  const v = (o) => (
    /*valueForItem*/
    o[7](
      /*item*/
      o[13]
    )
  );
  for (let o = 0; o < w.length; o += 1) {
    let f = He(t, w, o), a = v(f);
    u.set(a, i[o] = Ve(a, f));
  }
  return {
    c() {
      for (let o = 0; o < e.length; o += 1)
        e[o].c();
      n = q();
      for (let o = 0; o < i.length; o += 1)
        i[o].c();
      h = Qe();
    },
    m(o, f) {
      for (let a = 0; a < e.length; a += 1)
        e[a] && e[a].m(o, f);
      X(o, n, f);
      for (let a = 0; a < i.length; a += 1)
        i[a] && i[a].m(o, f);
      X(o, h, f), m = !0;
    },
    p(o, [f]) {
      f & /*folders, depth, expanded, value, toggleItem, toggleFolder, valueForItem, labelForItem, folderPath, folderLabel*/
      509 && (p = se(
        /*folders*/
        o[0]
      ), Ue(), e = Le(e, f, F, 1, o, p, l, n.parentNode, zt, Pe, n, Ce), Je()), f & /*labelForItem, items, value, valueForItem, depth, toggleItem*/
      438 && (w = se(
        /*items*/
        o[1]
      ), i = Le(i, f, v, 1, o, w, u, h.parentNode, Ft, Ve, h, He));
    },
    i(o) {
      if (!m) {
        for (let f = 0; f < p.length; f += 1)
          le(e[f]);
        m = !0;
      }
    },
    o(o) {
      for (let f = 0; f < e.length; f += 1)
        he(e[f]);
      m = !1;
    },
    d(o) {
      o && (U(n), U(h));
      for (let f = 0; f < e.length; f += 1)
        e[f].d(o);
      for (let f = 0; f < i.length; f += 1)
        i[f].d(o);
    }
  };
}
function oe(t) {
  return String(t.name || t.path || "");
}
function xe(t) {
  return String(t.path || t.name || "");
}
function Ht(t, e, l) {
  let { folders: n = [] } = e, { items: i = [] } = e, { depth: u = 0 } = e, { expanded: h } = e, { value: m } = e, { toggleItem: p } = e, { toggleFolder: F } = e, { valueForItem: w } = e, { labelForItem: v } = e;
  const o = (d) => F(d), f = (d, _) => {
    (_.key === "Enter" || _.key === " ") && (_.preventDefault(), F(d));
  }, a = (d) => p(d), g = (d, _) => {
    (_.key === "Enter" || _.key === " ") && (_.preventDefault(), p(d));
  };
  return t.$$set = (d) => {
    "folders" in d && l(0, n = d.folders), "items" in d && l(1, i = d.items), "depth" in d && l(2, u = d.depth), "expanded" in d && l(3, h = d.expanded), "value" in d && l(4, m = d.value), "toggleItem" in d && l(5, p = d.toggleItem), "toggleFolder" in d && l(6, F = d.toggleFolder), "valueForItem" in d && l(7, w = d.valueForItem), "labelForItem" in d && l(8, v = d.labelForItem);
  }, [
    n,
    i,
    u,
    h,
    m,
    p,
    F,
    w,
    v,
    o,
    f,
    a,
    g
  ];
}
class et extends kt {
  constructor(e) {
    super(), St(this, e, Ht, Lt, Dt, {
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
    this.$$set({ folders: e }), V();
  }
  get items() {
    return this.$$.ctx[1];
  }
  set items(e) {
    this.$$set({ items: e }), V();
  }
  get depth() {
    return this.$$.ctx[2];
  }
  set depth(e) {
    this.$$set({ depth: e }), V();
  }
  get expanded() {
    return this.$$.ctx[3];
  }
  set expanded(e) {
    this.$$set({ expanded: e }), V();
  }
  get value() {
    return this.$$.ctx[4];
  }
  set value(e) {
    this.$$set({ value: e }), V();
  }
  get toggleItem() {
    return this.$$.ctx[5];
  }
  set toggleItem(e) {
    this.$$set({ toggleItem: e }), V();
  }
  get toggleFolder() {
    return this.$$.ctx[6];
  }
  set toggleFolder(e) {
    this.$$set({ toggleFolder: e }), V();
  }
  get valueForItem() {
    return this.$$.ctx[7];
  }
  set valueForItem(e) {
    this.$$set({ valueForItem: e }), V();
  }
  get labelForItem() {
    return this.$$.ctx[8];
  }
  set labelForItem(e) {
    this.$$set({ labelForItem: e }), V();
  }
}
const {
  SvelteComponent: Ct,
  action_destroyer: Et,
  append: H,
  attr: k,
  binding_callbacks: me,
  check_outros: tt,
  create_component: Pt,
  destroy_component: Vt,
  destroy_each: At,
  detach: Z,
  element: E,
  empty: Bt,
  ensure_array_like: Ae,
  flush: C,
  group_outros: lt,
  init: Nt,
  insert: G,
  listen: A,
  mount_component: Tt,
  null_to_empty: Be,
  run_all: nt,
  safe_not_equal: qt,
  set_data: de,
  space: J,
  stop_propagation: Rt,
  text: _e,
  toggle_class: Q,
  transition_in: R,
  transition_out: ne
} = window.__gradio__svelte__internal, { onDestroy: Zt, onMount: Gt, tick: ge } = window.__gradio__svelte__internal;
function Ne(t, e, l) {
  const n = t.slice();
  return n[52] = e[l], n[54] = l, n;
}
function Te(t) {
  let e, l, n, i, u, h, m, p, F, w, v, o, f, a = (
    /*show_label*/
    t[4] && /*label*/
    t[2] && qe(t)
  ), g = (
    /*selectedValue*/
    t[7].length === 0 && Re(t)
  ), d = Ae(
    /*selectedValue*/
    t[7]
  ), _ = [];
  for (let c = 0; c < d.length; c += 1)
    _[c] = Ze(Ne(t, d, c));
  let I = (
    /*open*/
    t[6] && Ge(t)
  ), b = (
    /*info*/
    t[3] && Ke(t)
  );
  return {
    c() {
      e = E("div"), l = E("div"), a && a.c(), n = J(), i = E("div"), u = E("div"), g && g.c(), h = J();
      for (let c = 0; c < _.length; c += 1)
        _[c].c();
      p = J(), I && I.c(), F = J(), b && b.c(), k(u, "class", "hierarchy-selector-chips svelte-dplsji"), k(i, "class", "hierarchy-selector-input svelte-dplsji"), k(i, "role", "button"), k(i, "tabindex", m = /*interactive*/
      t[5] ? 0 : -1), k(i, "aria-haspopup", "tree"), k(
        i,
        "aria-expanded",
        /*open*/
        t[6]
      ), Q(i, "hierarchy-selector-disabled", !/*interactive*/
      t[5]), k(l, "class", "hierarchy-selector-field svelte-dplsji"), k(
        e,
        "id",
        /*elem_id*/
        t[0]
      ), k(e, "class", w = Be(
        /*classes*/
        t[17]
      ) + " svelte-dplsji"), k(
        e,
        "style",
        /*style*/
        t[16]
      );
    },
    m(c, M) {
      G(c, e, M), H(e, l), a && a.m(l, null), H(l, n), H(l, i), H(i, u), g && g.m(u, null), H(u, h);
      for (let s = 0; s < _.length; s += 1)
        _[s] && _[s].m(u, null);
      t[43](i), H(l, p), I && I.m(l, null), H(e, F), b && b.m(e, null), t[45](e), v = !0, o || (f = [
        A(
          i,
          "click",
          /*openPanel*/
          t[22]
        ),
        A(
          i,
          "keydown",
          /*onInputKeydown*/
          t[26]
        )
      ], o = !0);
    },
    p(c, M) {
      if (/*show_label*/
      c[4] && /*label*/
      c[2] ? a ? a.p(c, M) : (a = qe(c), a.c(), a.m(l, n)) : a && (a.d(1), a = null), /*selectedValue*/
      c[7].length === 0 ? g ? g.p(c, M) : (g = Re(c), g.c(), g.m(u, h)) : g && (g.d(1), g = null), M[0] & /*interactive, draggedIndex, dragOverIndex, onDragStart, onDragEnd, onDrop, removeValue, displayValue, selectedValue*/
      60055712) {
        d = Ae(
          /*selectedValue*/
          c[7]
        );
        let s;
        for (s = 0; s < d.length; s += 1) {
          const j = Ne(c, d, s);
          _[s] ? _[s].p(j, M) : (_[s] = Ze(j), _[s].c(), _[s].m(u, null));
        }
        for (; s < _.length; s += 1)
          _[s].d(1);
        _.length = d.length;
      }
      (!v || M[0] & /*interactive*/
      32 && m !== (m = /*interactive*/
      c[5] ? 0 : -1)) && k(i, "tabindex", m), (!v || M[0] & /*open*/
      64) && k(
        i,
        "aria-expanded",
        /*open*/
        c[6]
      ), (!v || M[0] & /*interactive*/
      32) && Q(i, "hierarchy-selector-disabled", !/*interactive*/
      c[5]), /*open*/
      c[6] ? I ? (I.p(c, M), M[0] & /*open*/
      64 && R(I, 1)) : (I = Ge(c), I.c(), R(I, 1), I.m(l, null)) : I && (lt(), ne(I, 1, 1, () => {
        I = null;
      }), tt()), /*info*/
      c[3] ? b ? b.p(c, M) : (b = Ke(c), b.c(), b.m(e, null)) : b && (b.d(1), b = null), (!v || M[0] & /*elem_id*/
      1) && k(
        e,
        "id",
        /*elem_id*/
        c[0]
      ), (!v || M[0] & /*classes*/
      131072 && w !== (w = Be(
        /*classes*/
        c[17]
      ) + " svelte-dplsji")) && k(e, "class", w), (!v || M[0] & /*style*/
      65536) && k(
        e,
        "style",
        /*style*/
        c[16]
      );
    },
    i(c) {
      v || (R(I), v = !0);
    },
    o(c) {
      ne(I), v = !1;
    },
    d(c) {
      c && Z(e), a && a.d(), g && g.d(), At(_, c), t[43](null), I && I.d(), b && b.d(), t[45](null), o = !1, nt(f);
    }
  };
}
function qe(t) {
  let e, l;
  return {
    c() {
      e = E("span"), l = _e(
        /*label*/
        t[2]
      ), k(e, "class", "hierarchy-selector-label svelte-dplsji");
    },
    m(n, i) {
      G(n, e, i), H(e, l);
    },
    p(n, i) {
      i[0] & /*label*/
      4 && de(
        l,
        /*label*/
        n[2]
      );
    },
    d(n) {
      n && Z(e);
    }
  };
}
function Re(t) {
  let e, l;
  return {
    c() {
      e = E("span"), l = _e(
        /*label*/
        t[2]
      ), k(e, "class", "hierarchy-selector-placeholder svelte-dplsji");
    },
    m(n, i) {
      G(n, e, i), H(e, l);
    },
    p(n, i) {
      i[0] & /*label*/
      4 && de(
        l,
        /*label*/
        n[2]
      );
    },
    d(n) {
      n && Z(e);
    }
  };
}
function Ze(t) {
  let e, l, n = (
    /*displayValue*/
    t[18](
      /*selected*/
      t[52]
    ) + ""
  ), i, u, h, m, p, F;
  function w() {
    return (
      /*click_handler*/
      t[38](
        /*index*/
        t[54]
      )
    );
  }
  function v(...a) {
    return (
      /*dragstart_handler*/
      t[39](
        /*index*/
        t[54],
        ...a
      )
    );
  }
  function o(...a) {
    return (
      /*dragover_handler*/
      t[40](
        /*index*/
        t[54],
        ...a
      )
    );
  }
  function f(...a) {
    return (
      /*drop_handler*/
      t[42](
        /*index*/
        t[54],
        ...a
      )
    );
  }
  return {
    c() {
      e = E("span"), l = E("span"), i = _e(n), u = J(), h = E("button"), h.textContent = "x", m = J(), k(l, "class", "hierarchy-selector-chip-text svelte-dplsji"), k(h, "type", "button"), k(h, "class", "hierarchy-selector-remove svelte-dplsji"), k(h, "aria-label", "Remove"), k(e, "class", "hierarchy-selector-chip svelte-dplsji"), k(e, "role", "listitem"), k(
        e,
        "draggable",
        /*interactive*/
        t[5]
      ), Q(
        e,
        "hierarchy-selector-chip-dragging",
        /*draggedIndex*/
        t[13] === /*index*/
        t[54]
      ), Q(
        e,
        "hierarchy-selector-chip-over",
        /*dragOverIndex*/
        t[14] === /*index*/
        t[54]
      );
    },
    m(a, g) {
      G(a, e, g), H(e, l), H(l, i), H(e, u), H(e, h), H(e, m), p || (F = [
        A(h, "click", Rt(w)),
        A(e, "dragstart", v),
        A(
          e,
          "dragend",
          /*onDragEnd*/
          t[24]
        ),
        A(e, "dragover", o),
        A(
          e,
          "dragleave",
          /*dragleave_handler*/
          t[41]
        ),
        A(e, "drop", f)
      ], p = !0);
    },
    p(a, g) {
      t = a, g[0] & /*selectedValue*/
      128 && n !== (n = /*displayValue*/
      t[18](
        /*selected*/
        t[52]
      ) + "") && de(i, n), g[0] & /*interactive*/
      32 && k(
        e,
        "draggable",
        /*interactive*/
        t[5]
      ), g[0] & /*draggedIndex*/
      8192 && Q(
        e,
        "hierarchy-selector-chip-dragging",
        /*draggedIndex*/
        t[13] === /*index*/
        t[54]
      ), g[0] & /*dragOverIndex*/
      16384 && Q(
        e,
        "hierarchy-selector-chip-over",
        /*dragOverIndex*/
        t[14] === /*index*/
        t[54]
      );
    },
    d(a) {
      a && Z(e), p = !1, nt(F);
    }
  };
}
function Ge(t) {
  let e, l, n, i, u;
  return l = new et({
    props: {
      folders: (
        /*normalizedHierarchy*/
        t[8].folders || []
      ),
      items: (
        /*normalizedHierarchy*/
        t[8].items || []
      ),
      depth: 0,
      expanded: (
        /*expanded*/
        t[12]
      ),
      value: (
        /*selectedValue*/
        t[7]
      ),
      toggleItem: (
        /*toggleItem*/
        t[19]
      ),
      toggleFolder: (
        /*toggleFolder*/
        t[21]
      ),
      valueForItem: be,
      labelForItem: ve
    }
  }), {
    c() {
      e = E("div"), Pt(l.$$.fragment), k(e, "class", "hierarchy-selector-panel svelte-dplsji"), k(
        e,
        "style",
        /*panelStyle*/
        t[15]
      );
    },
    m(h, m) {
      G(h, e, m), Tt(l, e, null), t[44](e), n = !0, i || (u = Et(Xt.call(null, e)), i = !0);
    },
    p(h, m) {
      const p = {};
      m[0] & /*normalizedHierarchy*/
      256 && (p.folders = /*normalizedHierarchy*/
      h[8].folders || []), m[0] & /*normalizedHierarchy*/
      256 && (p.items = /*normalizedHierarchy*/
      h[8].items || []), m[0] & /*expanded*/
      4096 && (p.expanded = /*expanded*/
      h[12]), m[0] & /*selectedValue*/
      128 && (p.value = /*selectedValue*/
      h[7]), l.$set(p), (!n || m[0] & /*panelStyle*/
      32768) && k(
        e,
        "style",
        /*panelStyle*/
        h[15]
      );
    },
    i(h) {
      n || (R(l.$$.fragment, h), n = !0);
    },
    o(h) {
      ne(l.$$.fragment, h), n = !1;
    },
    d(h) {
      h && Z(e), Vt(l), t[44](null), i = !1, u();
    }
  };
}
function Ke(t) {
  let e, l;
  return {
    c() {
      e = E("div"), l = _e(
        /*info*/
        t[3]
      ), k(e, "class", "hierarchy-selector-info svelte-dplsji");
    },
    m(n, i) {
      G(n, e, i), H(e, l);
    },
    p(n, i) {
      i[0] & /*info*/
      8 && de(
        l,
        /*info*/
        n[3]
      );
    },
    d(n) {
      n && Z(e);
    }
  };
}
function Kt(t) {
  let e, l, n = (
    /*visible*/
    t[1] && Te(t)
  );
  return {
    c() {
      n && n.c(), e = Bt();
    },
    m(i, u) {
      n && n.m(i, u), G(i, e, u), l = !0;
    },
    p(i, u) {
      /*visible*/
      i[1] ? n ? (n.p(i, u), u[0] & /*visible*/
      2 && R(n, 1)) : (n = Te(i), n.c(), R(n, 1), n.m(e.parentNode, e)) : n && (lt(), ne(n, 1, 1, () => {
        n = null;
      }), tt());
    },
    i(i) {
      l || (R(n), l = !0);
    },
    o(i) {
      ne(n), l = !1;
    },
    d(i) {
      i && Z(e), n && n.d(i);
    }
  };
}
const Oe = 32, Ot = 8, ae = 6, ee = 8;
function pe(t) {
  return Array.isArray(t) ? t.map((e) => String(e)) : t == null || t === "" ? [] : [String(t)];
}
function Wt(t) {
  return {
    folders: it((t == null ? void 0 : t.folders) || []),
    items: rt((t == null ? void 0 : t.items) || [])
  };
}
function it(t) {
  return t.map((e) => ({
    ...e,
    folders: it(e.folders || []),
    items: rt(e.items || [])
  })).sort((e, l) => We(e).localeCompare(We(l), void 0, { sensitivity: "base" }));
}
function rt(t) {
  return t.map((e) => ({ ...e })).sort((e, l) => ve(e).localeCompare(ve(l), void 0, { sensitivity: "base" }));
}
function We(t) {
  return String(t.name || t.path || "");
}
function ve(t) {
  return String(t.name || t.path || t.value || "");
}
function Jt(t) {
  return String(t.path || t.name || t.value || "");
}
function be(t) {
  return String(t.value || t.path || t.name || "");
}
function Qt(t) {
  const e = {};
  function l(n) {
    for (const i of n.items || []) e[be(i)] = Jt(i);
    for (const i of n.folders || []) l(i);
  }
  return l(t), e;
}
function Ut(t) {
  return t.replace(/\\/g, "/").replace(/\.[^/.]+$/, "");
}
function Xt(t) {
  return document.body.appendChild(t), {
    destroy() {
      t.remove();
    }
  };
}
function Yt(t, e, l) {
  let n, i, u, h, m, p, { elem_id: F = "" } = e, { elem_classes: w = [] } = e, { visible: v = !0 } = e, { value: o = [] } = e, { hierarchy: f = { folders: [], items: [] } } = e, { height: a = 10 } = e, { label: g = "Hierarchy Selector" } = e, { info: d = void 0 } = e, { show_label: _ = !0 } = e, { container: I = !0 } = e, { scale: b = null } = e, { min_width: c = void 0 } = e, { interactive: M = !0 } = e, { gradio: s } = e, j, z, K, O = !1, Y = !1, W = /* @__PURE__ */ new Set(), N = null, $ = null, we = {}, ye = "";
  function st(r) {
    return we[r] || Ut(r);
  }
  function ie(r, S = "change", L = void 0) {
    var P, x, re;
    l(27, o = pe(r)), (P = s == null ? void 0 : s.dispatch) == null || P.call(s, "input"), S === "select" && ((x = s == null ? void 0 : s.dispatch) == null || x.call(s, "select", L)), (re = s == null ? void 0 : s.dispatch) == null || re.call(s, "change"), ge().then(T);
  }
  function ot(r) {
    if (!M) return;
    const S = be(r);
    n.includes(S) ? ie(n.filter((L) => L !== S), "select", { value: S, selected: !1 }) : ie([...n, S], "select", { value: S, selected: !0 });
  }
  function ke(r) {
    if (!M) return;
    const S = n.filter((L, P) => P !== r);
    ie(S, "select", {
      value: n[r],
      selected: !1
    });
  }
  function at(r) {
    W.has(r) ? W.delete(r) : W.add(r), l(12, W = new Set(W));
  }
  function T() {
    if (!z || !O) return;
    const r = z.getBoundingClientRect(), S = h, L = r.top - ee - ae, P = window.innerHeight - r.bottom - ee - ae, x = L < S && P > L, re = Math.max(Oe * 4, x ? P : L), ze = Math.min(S, re), wt = x ? r.bottom + ae : Math.max(ee, r.top - ze - ae), De = Math.max(ee, r.left), yt = Math.max(240, Math.min(r.width, window.innerWidth - De - ee));
    l(15, ye = `top:${Math.round(wt)}px;left:${Math.round(De)}px;width:${Math.round(yt)}px;height:${Math.round(ze)}px;`);
  }
  function Ie() {
    var r;
    M && (l(6, O = !0), l(36, Y = !0), (r = s == null ? void 0 : s.dispatch) == null || r.call(s, "focus"), ge().then(() => {
      z == null || z.focus(), T();
    }));
  }
  function Fe() {
    var r;
    !O && !Y || (l(6, O = !1), l(36, Y = !1), (r = s == null ? void 0 : s.dispatch) == null || r.call(s, "blur"));
  }
  function Me(r) {
    j != null && j.contains(r.target) || K != null && K.contains(r.target) || Fe();
  }
  function Se(r, S) {
    var L;
    M && (l(13, N = r), (L = S.dataTransfer) == null || L.setData("text/plain", String(r)), S.dataTransfer && (S.dataTransfer.effectAllowed = "move"));
  }
  function ut() {
    l(13, N = null), l(14, $ = null);
  }
  function je(r, S) {
    if (S.preventDefault(), N === null || N === r) return;
    const L = [...n], [P] = L.splice(N, 1);
    L.splice(r, 0, P), l(13, N = null), l(14, $ = null), ie(L);
  }
  function ft(r) {
    r.key === "Escape" ? Fe() : (r.key === "Enter" || r.key === " " || r.key === "ArrowDown") && (r.preventDefault(), Ie());
  }
  function ct() {
    return pe(o);
  }
  Gt(() => {
    document.addEventListener("pointerdown", Me, !0), window.addEventListener("resize", T), window.addEventListener("scroll", T, !0);
  }), Zt(() => {
    document.removeEventListener("pointerdown", Me, !0), window.removeEventListener("resize", T), window.removeEventListener("scroll", T, !0);
  });
  const ht = (r) => ke(r), dt = (r, S) => Se(r, S), _t = (r, S) => {
    S.preventDefault(), l(14, $ = r);
  }, mt = () => l(14, $ = null), gt = (r, S) => je(r, S);
  function pt(r) {
    me[r ? "unshift" : "push"](() => {
      z = r, l(10, z);
    });
  }
  function vt(r) {
    me[r ? "unshift" : "push"](() => {
      K = r, l(11, K);
    });
  }
  function bt(r) {
    me[r ? "unshift" : "push"](() => {
      j = r, l(9, j);
    });
  }
  return t.$$set = (r) => {
    "elem_id" in r && l(0, F = r.elem_id), "elem_classes" in r && l(28, w = r.elem_classes), "visible" in r && l(1, v = r.visible), "value" in r && l(27, o = r.value), "hierarchy" in r && l(29, f = r.hierarchy), "height" in r && l(30, a = r.height), "label" in r && l(2, g = r.label), "info" in r && l(3, d = r.info), "show_label" in r && l(4, _ = r.show_label), "container" in r && l(31, I = r.container), "scale" in r && l(32, b = r.scale), "min_width" in r && l(33, c = r.min_width), "interactive" in r && l(5, M = r.interactive), "gradio" in r && l(34, s = r.gradio);
  }, t.$$.update = () => {
    t.$$.dirty[0] & /*value*/
    134217728 && l(7, n = pe(o)), t.$$.dirty[0] & /*hierarchy*/
    536870912 && l(8, i = Wt(f)), t.$$.dirty[0] & /*normalizedHierarchy*/
    256 && (we = Qt(i)), t.$$.dirty[0] & /*height*/
    1073741824 && l(37, u = Math.max(10, Number(a) || 10)), t.$$.dirty[1] & /*panelRows*/
    64 && (h = u * Oe + Ot), t.$$.dirty[0] & /*elem_classes*/
    268435456 | t.$$.dirty[1] & /*container, focused*/
    33 && l(17, m = [
      "hierarchy-selector",
      I ? "hierarchy-selector-container" : "",
      Y ? "hierarchy-selector-focused" : "",
      ...Array.isArray(w) ? w : []
    ].filter(Boolean).join(" ")), t.$$.dirty[1] & /*scale, min_width*/
    6 && l(16, p = [
      b !== null ? `flex-grow:${b};` : "",
      c !== void 0 ? `min-width:${c}px;` : ""
    ].join("")), t.$$.dirty[0] & /*open, selectedValue*/
    192 && O && n && ge().then(T);
  }, [
    F,
    v,
    g,
    d,
    _,
    M,
    O,
    n,
    i,
    j,
    z,
    K,
    W,
    N,
    $,
    ye,
    p,
    m,
    st,
    ot,
    ke,
    at,
    Ie,
    Se,
    ut,
    je,
    ft,
    o,
    w,
    f,
    a,
    I,
    b,
    c,
    s,
    ct,
    Y,
    u,
    ht,
    dt,
    _t,
    mt,
    gt,
    pt,
    vt,
    bt
  ];
}
class $t extends Ct {
  constructor(e) {
    super(), Nt(
      this,
      e,
      Yt,
      Kt,
      qt,
      {
        elem_id: 0,
        elem_classes: 28,
        visible: 1,
        value: 27,
        hierarchy: 29,
        height: 30,
        label: 2,
        info: 3,
        show_label: 4,
        container: 31,
        scale: 32,
        min_width: 33,
        interactive: 5,
        gradio: 34,
        get_value: 35
      },
      null,
      [-1, -1]
    );
  }
  get elem_id() {
    return this.$$.ctx[0];
  }
  set elem_id(e) {
    this.$$set({ elem_id: e }), C();
  }
  get elem_classes() {
    return this.$$.ctx[28];
  }
  set elem_classes(e) {
    this.$$set({ elem_classes: e }), C();
  }
  get visible() {
    return this.$$.ctx[1];
  }
  set visible(e) {
    this.$$set({ visible: e }), C();
  }
  get value() {
    return this.$$.ctx[27];
  }
  set value(e) {
    this.$$set({ value: e }), C();
  }
  get hierarchy() {
    return this.$$.ctx[29];
  }
  set hierarchy(e) {
    this.$$set({ hierarchy: e }), C();
  }
  get height() {
    return this.$$.ctx[30];
  }
  set height(e) {
    this.$$set({ height: e }), C();
  }
  get label() {
    return this.$$.ctx[2];
  }
  set label(e) {
    this.$$set({ label: e }), C();
  }
  get info() {
    return this.$$.ctx[3];
  }
  set info(e) {
    this.$$set({ info: e }), C();
  }
  get show_label() {
    return this.$$.ctx[4];
  }
  set show_label(e) {
    this.$$set({ show_label: e }), C();
  }
  get container() {
    return this.$$.ctx[31];
  }
  set container(e) {
    this.$$set({ container: e }), C();
  }
  get scale() {
    return this.$$.ctx[32];
  }
  set scale(e) {
    this.$$set({ scale: e }), C();
  }
  get min_width() {
    return this.$$.ctx[33];
  }
  set min_width(e) {
    this.$$set({ min_width: e }), C();
  }
  get interactive() {
    return this.$$.ctx[5];
  }
  set interactive(e) {
    this.$$set({ interactive: e }), C();
  }
  get gradio() {
    return this.$$.ctx[34];
  }
  set gradio(e) {
    this.$$set({ gradio: e }), C();
  }
  get get_value() {
    return this.$$.ctx[35];
  }
}
export {
  $t as default
};
