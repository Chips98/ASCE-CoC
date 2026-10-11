/* ==========================================================================
   ASCE-CoC 仿真监控工作台 · 前端
   全中文界面。SSE 增量渲染，避免整块重排。
   * 卡片支持「概览 / 展开认知链」两种阅读方式
   * 详情用宽侧栏按「看到 → 认知链 → 动作 → 原文」组织，可导出 PDF
   * 支持轮次导航与回放：仿真结束后可逐轮回看历史
   ========================================================================== */

(function () {
  "use strict";

  var MAX_CARDS = 40;
  var PLAY_INTERVAL = 2600;

  var DIM_ORDER = ["情绪", "情感", "思维", "立场", "观点", "意图", "一致性"];
  var DIM_CLASS = {
    "情绪": "d-emotion", "情感": "d-mood", "思维": "d-thinking",
    "立场": "d-stance", "观点": "d-opinion", "意图": "d-intention",
    "一致性": "d-integration"
  };
  var ACTION_CN = {
    like_post: "点赞帖子", dislike_post: "点踩帖子", create_comment: "发表评论",
    reply_to_comment: "回复评论", share_comment: "转发评论", like_comment: "点赞评论",
    dislike_comment: "点踩评论", create_post: "发布新帖", follow: "关注用户",
    unfollow: "取消关注", do_nothing: "不做动作", refresh: "刷新信息流",
    trend: "查看热门", search_posts: "搜索帖子", search_user: "搜索用户",
    get_trending_comments: "查看热门评论", get_post_detail: "查看帖子详情",
    get_user_profile: "查看用户资料"
  };

  var state = {
    latestTick: 0,      // 数据库里的最新轮次
    viewTick: null,     // 正在查看的轮次；null = 跟随最新
    totalTicks: 0,
    numAgents: 0,
    live: true,         // 是否跟随最新
    paused: false,      // 是否暂停头插
    playing: false,
    playTimer: null,
    pollTimer: null,
    loadedTick: null,   // 当前真正渲染在页面上的轮次
    seen: Object.create(null),
    cards: [],
    expanded: Object.create(null),
    pending: 0,
    es: null,
    detail: null,
    detailTab: "seen"
  };

  function $(id) { return document.getElementById(id); }

  // ─────────────────────────────────────────────── DOM 工具
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  function fmtNum(n) {
    n = Number(n) || 0;
    if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
    if (n >= 1e4) return (n / 1e3).toFixed(1) + "k";
    return String(n);
  }
  function fmtMs(ms) {
    ms = Number(ms) || 0;
    return ms >= 1000 ? (ms / 1000).toFixed(1) + "s" : Math.round(ms) + "ms";
  }
  function actionLabel(name) {
    if (ACTION_CN[name]) return ACTION_CN[name];
    return name + "（模型自创，未执行）";
  }
  function dimClass(label) { return DIM_CLASS[label] || ""; }
  function orderDims(labels) {
    return labels.slice().sort(function (a, b) {
      var ia = DIM_ORDER.indexOf(a), ib = DIM_ORDER.indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });
  }
  function showToast(msg) {
    var t = $("toast");
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(showToast._t);
    showToast._t = setTimeout(function () { t.hidden = true; }, 2800);
  }

  // 当前查看的轮次（回放时是选中的轮次，否则是最新轮次）
  function currentTick() {
    return state.viewTick === null ? state.latestTick : state.viewTick;
  }
  function isReplay() {
    return state.viewTick !== null && state.viewTick !== state.latestTick;
  }

  // ─────────────────────────────────────────────── 状态栏 + 指标
  function renderOverview(ov) {
    state.totalTicks = Number(ov.total_ticks) || 0;
    var newLatest = Number(ov.current_tick) || 0;
    var latestChanged = newLatest !== state.latestTick;
    state.latestTick = newLatest;
    state.numAgents = Number(ov.num_agents) || 0;

    var topic = ov.topic || "—";
    $("topic-chip").textContent = topic === "cut" ? "中美关税战" : topic;
    $("profile-chip").textContent = (ov.chain_profile || "—") + " 档";

    var viewed = currentTick();
    var pct = state.totalTicks ? Math.min(100, viewed / state.totalTicks * 100) : 0;
    $("progress-bar").style.width = pct.toFixed(1) + "%";
    $("progress-text").textContent =
      "第 " + viewed + " / " + state.totalTicks + " 轮" +
      (isReplay() ? "（回放）" : "");

    var t = ov.totals || {};
    var ticks = ov.ticks || [];
    var row = null;
    for (var i = 0; i < ticks.length; i++) {
      if (Number(ticks[i].tick) === viewed) row = ticks[i];
    }
    if (!row && ticks.length) row = ticks[ticks.length - 1];

    $("kpi-tick").textContent = viewed;
    $("kpi-tick-unit").textContent = "/ " + state.totalTicks;
    $("kpi-active").textContent = row ? (row.active_count || 0) : 0;
    $("kpi-active-unit").textContent = "/ " + state.numAgents;
    $("kpi-posts").textContent = fmtNum(t.all_posts);
    $("kpi-comments").textContent = fmtNum(t.comments);
    $("kpi-tools").textContent = fmtNum(t.tool_calls);
    $("kpi-tokens").textContent = fmtNum(t.tokens);

    if (row) {
      var miss = Number(row.chain_missing_count) || 0;
      var mv = $("kpi-missing");
      mv.textContent = miss;
      mv.className = "m-value" + (miss === 0 ? " is-ok" : miss < 10 ? " is-warn" : " is-bad");
      $("kpi-missing-unit").textContent = miss === 0 ? "全部写出" : "未写出";
    }

    $("p-posts").textContent = fmtNum(t.all_posts);
    $("p-comments").textContent = fmtNum(t.comments);
    $("p-likes").textContent = fmtNum(t.likes);
    $("p-follows").textContent = fmtNum(t.follows);
    $("p-turns").textContent = fmtNum(t.turns);
    $("p-pool").textContent = fmtNum(ov.pool_size);

    $("m-topic").textContent = topic;
    $("m-profile").textContent = ov.chain_profile || "—";
    $("m-model").textContent = ov.model || "—";

    if (latestChanged || !$("tick-select").options.length) renderTickNav();
    updateReplayBanner();
  }

  function setConn(on, text) {
    $("conn").className = "conn " + (on ? "is-on" : "is-off");
    $("conn-text").textContent = text || (on ? "已连接" : "已断开");
  }

  // ─────────────────────────────────────────────── 轮次导航
  function renderTickNav() {
    var sel = $("tick-select");
    var keep = sel.value;
    clear(sel);
    for (var i = 0; i <= state.totalTicks; i++) {
      var o = el("option");
      o.value = String(i);
      o.textContent = "第 " + i + " 轮" + (i === state.latestTick ? " · 最新" : "");
      sel.appendChild(o);
    }
    sel.value = String(currentTick());
    if (keep && sel.querySelector('option[value="' + keep + '"]')) sel.value = keep;
    syncNavButtons();
  }

  function syncNavButtons() {
    var v = currentTick();
    $("tick-prev").disabled = v <= 0;
    $("tick-next").disabled = v >= state.latestTick;
    $("back-live").hidden = !isReplay();
    $("tick-select").value = String(v);
  }

  function updateReplayBanner() {
    var b = $("replay-banner");
    if (isReplay()) {
      b.hidden = false;
      $("replay-text").textContent =
        "正在回放第 " + currentTick() + " 轮（最新为第 " + state.latestTick + " 轮）";
    } else {
      b.hidden = true;
    }
  }

  function goTick(tick) {
    tick = Math.max(0, Math.min(state.latestTick, Number(tick) || 0));
    state.viewTick = tick === state.latestTick ? null : tick;
    stopPlay();
    syncNavButtons();
    updateReplayBanner();
    loadTick(currentTick());
    refreshOverview();
  }

  function backToLive() {
    state.viewTick = null;
    stopPlay();
    syncNavButtons();
    updateReplayBanner();
    state.pending = 0;
    $("new-pill").hidden = true;
    loadTick(state.latestTick);
  }

  function togglePlay() {
    if (state.playing) { stopPlay(); return; }
    if (state.latestTick <= 0) return;
    // 从最早一轮开始播
    state.playing = true;
    $("tick-play").textContent = "❚❚";
    $("tick-play").title = "暂停回放";
    if (currentTick() >= state.latestTick) goTick(0);
    state.playTimer = setInterval(function () {
      var next = currentTick() + 1;
      if (next > state.latestTick) { stopPlay(); return; }
      goTick(next);
      if (next >= state.latestTick) stopPlay();
    }, PLAY_INTERVAL);
  }

  function stopPlay() {
    if (state.playTimer) { clearInterval(state.playTimer); state.playTimer = null; }
    state.playing = false;
    var btn = $("tick-play");
    if (btn) { btn.textContent = "▶▶"; btn.title = "自动回放"; }
  }

  // ─────────────────────────────────────────────── 活动卡片
  function buildCard(a) {
    var key = a.agent_id + ":" + a.tick;
    var card = el("div", "act" + (a.status === "failed" ? " is-failed"
                                : a.status === "fallback" ? " is-fallback" : ""));

    var head = el("div", "act-head");
    head.appendChild(el("span", "act-agent", a.agent_name || ("用户" + a.agent_id)));
    head.appendChild(el("span", "act-tick", "第 " + a.tick + " 轮 · ID " + a.agent_id));

    var muted = !a.primary_action || a.primary_action === "do_nothing";
    var actEl = el("span", "act-action" + (a.primary_action ? (muted ? " is-none" : "") : " is-none"));
    actEl.appendChild(document.createTextNode(
      a.primary_action ? actionLabel(a.primary_action) : "未采取动作"));
    if (a.primary_target_id) actEl.appendChild(el("span", "act-target", "#" + a.primary_target_id));
    head.appendChild(actEl);
    card.appendChild(head);

    if (a.reason) card.appendChild(el("div", "act-reason", a.reason));

    var foot = el("div", "act-foot");
    var byTool = {};
    (a.actions || []).forEach(function (c) {
      var n = c.tool_cn || c.tool || "?";
      byTool[n] = byTool[n] || { n: 0, fail: 0 };
      byTool[n].n++;
      if (!c.success) byTool[n].fail++;
    });
    Object.keys(byTool).forEach(function (k) {
      var v = byTool[k];
      var chip = el("span", "tool-chip" + (v.fail ? " is-fail" : ""));
      chip.appendChild(document.createTextNode(k));
      chip.appendChild(el("span", "tc-n", "×" + v.n + (v.fail ? " (" + v.fail + "✗)" : "")));
      foot.appendChild(chip);
    });

    var steps = a.chain || [];
    var total = steps.length;
    var missing = steps.filter(function (s) { return !s.present; }).length;
    var inherited = steps.filter(function (s) { return s.present && s.inherited; }).length;
    var statTxt = "节点 " + (total - missing) + "/" + total;
    if (missing) statTxt += " · 缺 " + missing;
    if (inherited) statTxt += " · 兜底 " + inherited;
    foot.appendChild(el("span", "act-stat" + (missing ? " is-warn" : ""), statTxt));
    foot.appendChild(el("span", "act-stat",
      "思考 " + fmtMs((a.stage1_ms || 0) + (a.stage2_ms || 0)) +
      " · token " + fmtNum((a.prompt_tokens || 0) + (a.completion_tokens || 0))));

    var btns = el("div", "act-btns");
    if (total) {
      var toggle = el("button", "btn", state.expanded[key] ? "收起认知链" : "展开认知链");
      toggle.addEventListener("click", function () {
        var body = card.querySelector(".chain");
        if (state.expanded[key]) {
          delete state.expanded[key];
          if (body) body.remove();
          toggle.textContent = "展开认知链";
        } else {
          state.expanded[key] = true;
          card.appendChild(buildChain(a));
          toggle.textContent = "收起认知链";
        }
      });
      btns.appendChild(toggle);
    }
    var detailBtn = el("button", "btn", "查看完整记录");
    detailBtn.addEventListener("click", function () { openDrawer(a.agent_id, a.tick); });
    btns.appendChild(detailBtn);
    foot.appendChild(btns);
    card.appendChild(foot);

    if (state.expanded[key] && total) card.appendChild(buildChain(a));

    card.dataset.key = key;
    card.dataset.tick = a.tick;
    return card;
  }

  function buildChain(a) {
    var box = el("div", "chain");
    var groups = {};
    (a.chain || []).forEach(function (s) { (groups[s.dim] = groups[s.dim] || []).push(s); });
    orderDims(Object.keys(groups)).forEach(function (dim) {
      var g = el("div", "chain-group");
      g.appendChild(el("div", "chain-dim " + dimClass(dim), dim));
      var nodes = el("div", "chain-nodes");
      groups[dim].forEach(function (s) {
        var node = el("div", "chain-node" + (s.present ? "" : " is-missing"));
        node.appendChild(el("span", "chain-node-key", s.key || ""));
        node.appendChild(el("span", "chain-node-text",
          s.present ? s.text : "（模型未写出该节点）"));
        nodes.appendChild(node);
      });
      g.appendChild(nodes);
      box.appendChild(g);
    });
    return box;
  }

  function pushActivity(a, animate) {
    var key = a.agent_id + ":" + a.tick;
    if (state.seen[key]) return;
    state.seen[key] = true;

    var feed = $("feed");
    var st = feed.querySelector(".state");
    if (st) st.remove();

    var card = buildCard(a);
    // 只有实时新增才播入场动画；整轮批量载入时若也播，几十张卡会同时淡入，
    // 截图和快速浏览时看起来"整页发灰"。
    if (animate) card.classList.add("is-enter");
    // 回放按轮次从早到晚铺开；实时模式头插最新的
    if (isReplay()) feed.appendChild(card);
    else feed.insertBefore(card, feed.firstChild);
    state.cards.push(card);

    while (state.cards.length > MAX_CARDS) {
      var old = state.cards.shift();
      if (old && old.parentNode) old.parentNode.removeChild(old);
    }
  }

  function makeState(title, hint, cls) {
    var box = el("div", "state" + (cls ? " " + cls : ""));
    if (cls === "state-loading") box.appendChild(el("div", "spinner"));
    box.appendChild(el("p", null, title));
    if (hint) box.appendChild(el("p", "state-hint", hint));
    return box;
  }

  function emptyFeed(title, hint) {
    var feed = $("feed");
    clear(feed);
    state.seen = Object.create(null);
    state.cards = [];
    state.pending = 0;
    $("new-pill").hidden = true;
    feed.appendChild(makeState(title || "暂无活动",
      hint || "仿真开始后，每个智能体的认知链与动作会实时出现在这里。"));
  }

  // ─────────────────────────────────────────────── 右侧：认知链健康度
  // 偏离阈值：健康运行下节点写出率在 98%~100% 之间波动，若把 98% 也标黄，
  // 整个面板会全是琥珀色，"突出异常"就失去了意义。只有明显缺口才提示。
  var WARN_RATE = 0.95;
  var BAD_RATE = 0.80;

  function rateClass(rate) {
    if (rate <= 0) return "is-zero";
    if (rate < BAD_RATE) return "is-bad";
    if (rate < WARN_RATE) return "is-low";
    return "";
  }

  function renderChainHealth(nodes) {
    var box = $("chain-groups");
    clear(box);
    if (!nodes || !nodes.length) {
      box.appendChild(makeState("暂无数据", null, "state-mini"));
      return;
    }
    var groups = {};
    nodes.forEach(function (n) {
      var d = n.dim || "未分组";
      (groups[d] = groups[d] || []).push(n);
    });
    var dims = Object.keys(groups);
    dims.sort(function (a, b) {
      var ma = groups[a].filter(function (x) {
        return (x.explicit_rate === undefined ? x.rate : x.explicit_rate) < 1;
      }).length;
      var mb = groups[b].filter(function (x) {
        return (x.explicit_rate === undefined ? x.rate : x.explicit_rate) < 1;
      }).length;
      if ((ma > 0) !== (mb > 0)) return ma > 0 ? -1 : 1;
      var ord = orderDims([a, b]);
      return ord.indexOf(a) - ord.indexOf(b);
    });

    dims.forEach(function (dim) {
      var items = groups[dim];
      var g = el("div", "dim-group");
      var hasAlert = items.some(function (x) {
        return (x.explicit_rate === undefined ? x.rate : x.explicit_rate) < WARN_RATE;
      });
      if (hasAlert) g.classList.add("is-alert");

      var head = el("div", "dim-head");
      head.appendChild(el("span", "dim-name " + dimClass(dim), dim));
      var avgP = items.reduce(function (s, x) { return s + (x.rate || 0); }, 0) / items.length;
      var avgE = items.reduce(function (s, x) {
        return s + (x.explicit_rate === undefined ? x.rate : x.explicit_rate);
      }, 0) / items.length;
      var gap = avgP - avgE > 0.005;
      var rate = el("span", "dim-rate" + (gap ? " is-warn" : ""),
        Math.round(avgP * 100) + "%");
      if (gap) rate.textContent += " · 显式 " + Math.round(avgE * 100) + "%";
      rate.title = "有内容比例" + (gap ? " / 显式标注比例" : "（与显式标注一致）");
      head.appendChild(rate);
      g.appendChild(head);

      items.forEach(function (x) {
        var present = x.rate || 0;
        var explicit = (x.explicit_rate === undefined ? x.rate : x.explicit_rate);
        var gapN = present - explicit > 0.005;

        var row = el("div", "node-row");
        var h = el("div", "node-row-head");
        h.appendChild(el("span", "node-name", (x.key || "") + " " + (x.node || "")));
        var nums = el("span", "node-nums");
        nums.appendChild(el("span", "n-present",
          (x.present || 0) + "/" + (x.total || 0)));
        if (gapN) {
          nums.appendChild(el("span", "n-sep", "·"));
          nums.appendChild(el("span", "n-explicit " + rateClass(explicit),
            "显式 " + Math.round(explicit * 100) + "%"));
        }
        h.appendChild(nums);
        h.title = (x.node || x.key) + "：有内容 " + (x.present || 0) + "/" + (x.total || 0) +
                  "，其中模型显式标注 " + Math.round(explicit * 100) + "%";
        row.appendChild(h);

        var track = el("div", "bar-track");
        var fill = el("div", "bar-fill " + (explicit <= 0 ? "is-bad"
                                            : explicit < WARN_RATE ? "is-warn" : "is-ok"));
        fill.style.width = Math.round(present * 100) + "%";
        track.appendChild(fill);
        row.appendChild(track);
        g.appendChild(row);
      });
      box.appendChild(g);
    });
  }

  // ─────────────────────────────────────────────── 右侧：动作分布
  function renderTools(items) {
    var box = $("tool-list");
    clear(box);
    if (!items || !items.length) {
      box.appendChild(makeState("本轮还没有动作", null, "state-mini"));
      return;
    }
    var maxN = items.reduce(function (m, x) { return Math.max(m, x.n || 0); }, 1);
    items.forEach(function (x) {
      var row = el("div", "tool-row");
      row.appendChild(el("span", "tool-name", x.cn || x.name));
      var counts = el("span", "tool-counts");
      counts.appendChild(el("span", "t-ok", x.n));
      var fail = (x.n || 0) - (x.ok || 0);
      if (fail > 0) counts.appendChild(el("span", "t-fail", "✗" + fail));
      row.appendChild(counts);
      var bar = el("div", "tool-bar");
      var span = el("span");
      span.style.width = Math.round((x.n || 0) / maxN * 100) + "%";
      bar.appendChild(span);
      row.appendChild(bar);
      box.appendChild(row);
    });
  }

  // ─────────────────────────────────────────────── 载入某一轮
  function loadTick(tick) {
    state.loadedTick = Number(tick);
    state.seen = Object.create(null);
    state.cards = [];
    var feed = $("feed");
    clear(feed);
    feed.appendChild(makeState("正在读取第 " + tick + " 轮…", null, "state-loading"));

    var q = "tick=" + tick;
    return Promise.all([
      fetch("/api/activities?" + q + "&limit=" + MAX_CARDS).then(function (r) { return r.json(); }),
      fetch("/api/chain-health?" + q).then(function (r) { return r.json(); }),
      fetch("/api/tool-stats?" + q).then(function (r) { return r.json(); })
    ]).then(function (res) {
      var list = res[0] || [];
      clear(feed);
      state.seen = Object.create(null);
      state.cards = [];
      if (!list.length) {
        if (tick === 0) {
          feed.appendChild(makeState("第 0 轮是初始化轮",
            "这一轮只导入历史语料并建立初始关注网络，没有智能体决策记录。", null));
        } else {
          feed.appendChild(makeState("第 " + tick + " 轮没有活动记录",
            "该轮可能尚未落库，或这一轮没有任何智能体被激活。", null));
        }
      } else {
        // 接口按 tick DESC / agent ASC；回放时按 agent 顺序自上而下展示
        list.slice().reverse().forEach(function (a) { pushActivity(a, false); });
      }
      renderChainHealth(res[1]);
      renderTools(res[2]);
    }).catch(function (err) {
      clear(feed);
      feed.appendChild(makeState("读取失败：" + err.message,
        "请确认监控服务仍在运行，或刷新页面重试。", "state-error"));
    });
  }

  function refreshOverview() {
    return fetch("/api/overview")
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(renderOverview);
  }

  // ─────────────────────────────────────────────── 宽侧栏详情
  function syncUrl(agentId, tick, tab) {
    if (agentId === null) {
      history.replaceState(null, "", location.pathname);
    } else {
      var q = "?agent=" + agentId + "&tick=" + tick;
      if (tab && tab !== "seen") q += "&tab=" + tab;
      history.replaceState(null, "", q);
    }
  }

  var VALID_TABS = { seen: 1, chain: 1, actions: 1, raw: 1 };

  function openDrawer(agentId, tick, tab) {
    state.detailTab = VALID_TABS[tab] ? tab : "seen";
    syncUrl(agentId, tick, state.detailTab);
    $("drawer").hidden = false;
    $("drawer-title").textContent = "加载中…";
    $("drawer-sub").textContent = "正在读取第 " + tick + " 轮记录";
    $("btn-export").disabled = true;
    setTabs();
    var body = $("drawer-body");
    clear(body);
    body.appendChild(makeState("正在加载记录…", null, "state-loading"));

    fetch("/api/turn?agent_id=" + encodeURIComponent(agentId) + "&tick=" + encodeURIComponent(tick))
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (data) {
        if (!data || data.error) throw new Error((data && data.error) || "未找到记录");
        state.detail = data;
        $("drawer-title").textContent = data.agent_name || ("用户 " + agentId);
        $("drawer-sub").textContent =
          "第 " + tick + " 轮 · " + (data.status === "ok" ? "正常" : data.status) +
          " · 思考 " + fmtMs((data.stage1_ms || 0) + (data.stage2_ms || 0)) +
          " · token " + fmtNum((data.prompt_tokens || 0) + (data.completion_tokens || 0));
        $("btn-export").disabled = false;
        renderDetailTab();
      })
      .catch(function (err) {
        $("drawer-title").textContent = "读取失败";
        $("drawer-sub").textContent = "用户 " + agentId + " · 第 " + tick + " 轮";
        var b = $("drawer-body");
        clear(b);
        b.appendChild(makeState("无法读取这条记录：" + err.message,
          "可能是该轮次尚未落库，或仿真进程正在写入。稍后重试。", "state-error"));
      });
  }

  function setTabs() {
    Array.prototype.forEach.call($("drawer-tabs").children, function (btn) {
      btn.classList.toggle("is-active", btn.dataset.tab === state.detailTab);
    });
  }

  function renderDetailTab() {
    var d = state.detail;
    if (!d) return;
    var body = $("drawer-body");
    clear(body);
    if (state.detailTab === "seen") body.appendChild(secSeen(d));
    else if (state.detailTab === "chain") body.appendChild(secChain(d));
    else if (state.detailTab === "actions") body.appendChild(secActions(d));
    else body.appendChild(secRaw(d));
  }

  function sec(title, note) {
    var s = el("section", "sec");
    var h = el("h3", "sec-title", title);
    if (note) h.appendChild(el("span", "sec-n", note));
    s.appendChild(h);
    return s;
  }

  function secSeen(d) {
    var posts = d.feed_posts || [];
    var s = sec("看到的内容", posts.length + " 条帖子");
    if (!posts.length) {
      s.appendChild(makeState("本轮没有刷到内容", null, "state-mini"));
      return s;
    }
    posts.forEach(function (p) {
      var w = el("div", "seen-post");
      var head = el("div", "seen-post-head");
      head.appendChild(el("span", "sp-id", "帖子 " + p.post_id));
      head.appendChild(el("span", null, p.author_name || ("用户" + p.user_id)));
      head.appendChild(el("span", null, "赞 " + (p.num_likes || 0) + " · 踩 " + (p.num_dislikes || 0)));
      w.appendChild(head);
      w.appendChild(el("div", "seen-post-body", p.content || ""));
      s.appendChild(w);
    });
    return s;
  }

  function secChain(d) {
    var steps = (d.chain && d.chain.steps) || [];
    var missing = (d.chain && d.chain.missing) || [];
    var s = sec("认知链", steps.length + " 个节点" + (missing.length ? " · 缺 " + missing.length : ""));
    if (!steps.length) {
      s.appendChild(makeState("这一轮没有认知链记录", null, "state-mini"));
      return s;
    }
    var groups = {};
    steps.forEach(function (st) { (groups[st.dim] = groups[st.dim] || []).push(st); });
    var wrap = el("div", "detail-chain");
    orderDims(Object.keys(groups)).forEach(function (dim) {
      var g = el("div", "detail-dim");
      g.appendChild(el("div", "detail-dim-head " + dimClass(dim), dim));
      groups[dim].forEach(function (st) {
        var node = el("div", "detail-node" + (st.present ? "" : " is-missing"));
        var line = el("div");
        line.appendChild(el("span", "detail-node-key", st.key || ""));
        line.appendChild(el("span", "detail-node-label", st.node || ""));
        if (st.present && st.inherited) line.appendChild(el("span", "chain-inherited", "来自维度段"));
        node.appendChild(line);
        node.appendChild(el("div", "detail-node-text",
          st.present ? st.text : "（模型未写出该节点）"));
        g.appendChild(node);
      });
      wrap.appendChild(g);
    });
    s.appendChild(wrap);
    return s;
  }

  function secActions(d) {
    var acts = d.actions || [];
    var s = sec("实际动作", acts.length + " 次工具调用");
    if (!acts.length) {
      s.appendChild(makeState("这一轮没有调用任何工具", null, "state-mini"));
      return s;
    }
    acts.forEach(function (c) {
      var row = el("div", "act-row " + (c.success ? "is-ok" : "is-err"));
      var head = el("div", "act-row-head");
      head.appendChild(el("span", "act-row-order", "#" + (c.order || "")));
      head.appendChild(el("span", null, (c.tool_cn || c.tool) + (c.success ? "" : " · 未执行")));
      row.appendChild(head);
      var args = el("div", "act-row-args", "参数：");
      args.appendChild(el("code", null, JSON.stringify(c.args || {})));
      row.appendChild(args);
      if (!c.success && c.error) row.appendChild(el("div", "act-row-err", "拒绝原因：" + c.error));
      s.appendChild(row);
    });
    return s;
  }

  function secRaw(d) {
    var s = sec("两阶段原文");
    var b1 = el("div", "raw-block");
    b1.appendChild(el("h4", null, "第一阶段 · 感受与自我（情绪 / 情感 / 思维 / 立场）"));
    b1.appendChild(el("div", "raw-text", d.stage1_text || "（无输出）"));
    s.appendChild(b1);

    var b2 = el("div", "raw-block");
    b2.appendChild(el("h4", null, "第二阶段 · 判断与行动（观点 / 意图 / 一致性）"));
    b2.appendChild(el("div", "raw-text", d.stage2_text || "（无输出）"));
    s.appendChild(b2);

    if (d.error_text) {
      var b3 = el("div", "raw-block");
      b3.appendChild(el("h4", null, "执行异常"));
      b3.appendChild(el("div", "raw-text", d.error_text));
      s.appendChild(b3);
    }
    return s;
  }

  function closeDrawer() {
    $("drawer").hidden = true;
    state.detail = null;
    syncUrl(null, null);
  }

  // ─────────────────────────────────────────────── 跟随控制
  function setPaused(paused) {
    state.paused = paused;
    $("follow-live").checked = !paused;
    if (!paused) {
      state.pending = 0;
      $("new-pill").hidden = true;
      $("feed").scrollTop = 0;
    }
  }

  function bumpPending() {
    state.pending++;
    $("new-pill-text").textContent = "有 " + state.pending + " 条新活动";
    $("new-pill").hidden = false;
  }

  // ─────────────────────────────────────────────── SSE
  // 断连时的轮询兜底：SSE 不可用（例如被反向代理缓冲、连接数受限）时，
  // 页面仍然可用，只是刷新粒度从"每轮"退化为"每几秒"。
  function startPolling() {
    if (NO_STREAM || state.pollTimer) return;
    state.pollTimer = setInterval(function () {
      refreshOverview().then(function () {
        if (state.viewTick === null && !state.playing) {
          if (currentTick() !== state.loadedTick) loadTick(currentTick());
        }
      }).catch(function () { /* 继续等下一轮 */ });
    }, 4000);
  }

  function stopPolling() {
    if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
  }

  // ?no_stream=1：不建立 SSE 长连接，也不启动轮询，只做一次 REST 加载。
  // 用途：把页面嵌入离线报告、生成归档截图、或在禁止长连接的代理后面查看。
  // 正常使用时不要带这个参数。
  var NO_STREAM = new URLSearchParams(location.search).get("no_stream") === "1";

  function connect() {
    if (NO_STREAM) {
      setConn(true, "静态快照模式");
      return;
    }
    setConn(false, "连接中…");

    var es = new EventSource("/api/stream");
    state.es = es;

    es.addEventListener("open", function () {
      setConn(true, "已连接");
      stopPolling();
    });

    es.addEventListener("error", function () {
      setConn(false, "连接中断 · 已切换为轮询刷新");
      startPolling();
    });

    es.addEventListener("snapshot", function (e) {
      var ov = JSON.parse(e.data);
      renderOverview(ov);
      // 跟随最新：出现新轮次时自动切换（回放/播放中不打断）
      if (state.viewTick === null && !state.playing && currentTick() !== state.loadedTick) {
        loadTick(currentTick());
      }
    });

    es.addEventListener("activity", function (e) {
      var a = JSON.parse(e.data);
      if (isReplay() || state.playing) return;      // 回放中不打断
      if (Number(a.tick) !== currentTick()) {
        // 属于更新的轮次：快照事件会处理整轮切换
        if (state.paused) bumpPending();
        return;
      }
      if (state.paused) { bumpPending(); return; }
      pushActivity(a, true);
    });

    es.addEventListener("chain", function (e) {
      var d = JSON.parse(e.data);
      if (isReplay() || state.playing) return;
      if (Number(d.tick) !== currentTick()) return;
      renderChainHealth(d.nodes);
    });

    es.addEventListener("tools", function (e) {
      var d = JSON.parse(e.data);
      if (isReplay() || state.playing) return;
      if (Number(d.tick) !== currentTick()) return;
      renderTools(d.items);
    });
  }

  // ─────────────────────────────────────────────── 交互绑定
  function bindUI() {
    $("follow-live").addEventListener("change", function (e) {
      setPaused(!e.target.checked);
      if (!state.paused) backToLive();
    });

    $("new-pill").addEventListener("click", function () {
      setPaused(false);
      backToLive();
    });

    $("tick-prev").addEventListener("click", function () { goTick(currentTick() - 1); });
    $("tick-next").addEventListener("click", function () { goTick(currentTick() + 1); });
    $("tick-play").addEventListener("click", togglePlay);
    $("tick-select").addEventListener("change", function (e) { goTick(e.target.value); });
    $("back-live").addEventListener("click", backToLive);

    $("drawer-tabs").addEventListener("click", function (e) {
      var btn = e.target.closest(".tab");
      if (!btn) return;
      state.detailTab = btn.dataset.tab;
      setTabs();
      renderDetailTab();
      if (state.detail && !$("drawer").hidden) {
        syncUrl(state.detail.agent_id, state.detail.tick, state.detailTab);
      }
    });

    Array.prototype.forEach.call(document.querySelectorAll("[data-drawer-close]"), function (n) {
      n.addEventListener("click", closeDrawer);
    });

    $("btn-export").addEventListener("click", function () {
      var d = state.detail;
      if (!d) return;
      var url = "/print/turn?agent_id=" + encodeURIComponent(d.agent_id) +
                "&tick=" + encodeURIComponent(d.tick);
      var w = window.open(url, "_blank");
      if (!w) showToast("浏览器拦截了新窗口，请允许弹出窗口后重试");
    });

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") {
        if (!$("drawer").hidden) closeDrawer();
        else if (state.playing) stopPlay();
      }
      if ($("drawer").hidden) {
        if (e.key === "ArrowLeft") goTick(currentTick() - 1);
        if (e.key === "ArrowRight") goTick(currentTick() + 1);
      }
    });
  }

  // ─────────────────────────────────────────────── 启动
  bindUI();
  connect();

  // 首屏走 REST，而不是等 SSE 的第一帧：
  // 这样即使 SSE 被中间层阻断，页面依然能正常显示（只是不再自动刷新）。
  emptyFeed("正在连接监控服务…", "监控页与仿真共用一个数据库；仿真启动后会自动出现内容。");
  var boot = $("feed").querySelector(".state");
  if (boot) boot.className = "state state-loading";

  // 深链：?agent=5&tick=2 直接打开某条记录，便于分享与复核
  var params = new URLSearchParams(location.search);
  var deepAgent = parseInt(params.get("agent"), 10);
  var deepTick = parseInt(params.get("tick"), 10);
  var deepTab = params.get("tab");

  refreshOverview()
    .then(function () {
      var t = isNaN(deepTick) ? currentTick() : deepTick;
      if (!isNaN(deepAgent)) {
        state.viewTick = t === state.latestTick ? null : t;
        syncNavButtons();
        updateReplayBanner();
      }
      return loadTick(t);
    })
    .then(function () {
      if (!isNaN(deepAgent)) {
        openDrawer(deepAgent, isNaN(deepTick) ? currentTick() : deepTick, deepTab);
      }
    })
    .catch(function (err) {
      setConn(false, "无法连接监控服务");
      emptyFeed("无法连接监控服务",
        "请确认监控服务正在运行（" + err.message + "）。若服务已启动，刷新页面重试。",
        "state-error");
      startPolling();
    });
})();
