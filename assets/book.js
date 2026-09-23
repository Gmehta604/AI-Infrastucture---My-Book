/* ==========================================================================
   Gradient to Gigawatt
   Shared script for index.html and every chapter.

   It builds the top bar, sidebar, "on this page" links, prev/next pager,
   copy buttons, code highlighting, diagrams (Mermaid), theme toggle and
   reading progress. Chapter files only need their content.

   To add or rename a chapter, edit the CHAPTERS list below.
   ========================================================================== */

(function () {
  "use strict";

  // ---- The table of contents: single source of truth --------------------
  // To add a chapter or deep dive, add a line here and save the file with that name.
  var BOOK_TITLE = "Gradient to Gigawatt";
  var PARTS = [
    { id: "I",   title: "Foundations" },
    { id: "II",  title: "Deep Learning" },
    { id: "III", title: "Transformers and Large Language Models" },
    { id: "IV",  title: "AI Infrastructure" },
    { id: "V",   title: "The Industry and the Frontier" },
    { id: "A",   title: "Appendix" }
  ];

  var CHAPTERS = [
    { n: 1,  part: "I",   file: "01-story.html",        title: "The Story of AI, 1943–2026", desc: "Booms, winters, and the ideas that keep coming back." },
    { n: 2,  part: "I",   file: "02-math.html",         title: "The Math You Actually Need", desc: "Vectors, matmul FLOPs, gradients, probability, cross-entropy, number formats." },
    { n: 3,  part: "I",   file: "03-classical-ml.html", title: "Classical Machine Learning", desc: "Regression, trees, boosting, evaluation, and metrics tied to money." },
    { n: 4,  part: "II",  file: "04-neural-nets.html",  title: "Neural Networks and Backpropagation", desc: "Build an autograd engine from scratch and learn XOR." },
    { n: 5,  part: "II",  file: "05-training.html",     title: "Training Deep Networks Well", desc: "AdamW, schedules, normalisation, mixed precision, debugging." },
    { n: 6,  part: "II",  file: "06-cnn.html",          title: "Convolutional Networks and Computer Vision", desc: "Convolutions, ResNet, ViT, and transfer learning." },
    { n: 7,  part: "II",  file: "07-sequences.html",    title: "Sequence Models: From RNNs to Attention", desc: "RNNs, LSTMs, seq2seq, and the bottleneck attention removed." },
    { n: 8,  part: "III", file: "08-tokens.html",       title: "Tokenization and Embeddings", desc: "BPE from scratch, vocabularies, embeddings, positions." },
    { n: 9,  part: "III", file: "09-transformer.html",  title: "Attention and the Transformer", desc: "A modern Llama-style model written from scratch." },
    { n: 10, part: "III", file: "10-pretraining.html",  title: "Pretraining and Scaling Laws", desc: "Data pipelines, Chinchilla, and frontier training runs." },
    { n: 11, part: "III", file: "11-moe.html",          title: "Mixture of Experts and Modern Architectures", desc: "MoE, long context, SSMs and hybrid attention." },
    { n: 12, part: "III", file: "12-posttraining.html", title: "Post-Training: From Base Model to Assistant", desc: "SFT, RLHF, DPO, Constitutional AI, GRPO." },
    { n: 13, part: "III", file: "13-reasoning.html",    title: "Reasoning Models and Test-Time Compute", desc: "Thinking tokens, test-time scaling, and what it did to costs." },
    { n: 14, part: "III", file: "14-multimodal.html",   title: "Multimodal and Generative Media", desc: "CLIP, vision-language models, speech, diffusion, video." },
    { n: 15, part: "IV",  file: "15-hardware.html",     title: "GPUs and AI Accelerators", desc: "SMs, HBM, rooflines, NVIDIA generations, TPUs and rivals." },
    { n: 16, part: "IV",  file: "16-kernels.html",      title: "The Software Stack: CUDA, Triton, Kernels and Compilers", desc: "Fusion, torch.compile, Triton, FlashAttention, profiling." },
    { n: 17, part: "IV",  file: "17-distributed.html",  title: "Distributed Training", desc: "Collectives, DDP, FSDP/ZeRO, TP, PP, CP, EP." },
    { n: 18, part: "IV",  file: "18-finetuning.html",   title: "Fine-Tuning in Practice", desc: "When to fine-tune, LoRA, QLoRA, multi-LoRA serving." },
    { n: 19, part: "IV",  file: "19-inference.html",    title: "Inference and Serving", desc: "KV cache, batching, PagedAttention, quantization, speculation." },
    { n: 20, part: "IV",  file: "20-gateway.html",      title: "LLM Gateways, Routing and Cost Engineering", desc: "Routing, caching, failover, budgets, and a working gateway." },
    { n: 21, part: "IV",  file: "21-rag.html",          title: "Retrieval, Vector Search and RAG", desc: "Chunking, ANN indexes, hybrid search, reranking, RAG evals." },
    { n: 22, part: "IV",  file: "22-agents.html",       title: "Agents, Tools and MCP", desc: "The agent loop, tool calling, MCP, sandboxes, security." },
    { n: 23, part: "IV",  file: "23-evals.html",        title: "Evaluation and Observability", desc: "Application evals, LLM judges, CI gates, tracing." },
    { n: 24, part: "IV",  file: "24-platform.html",     title: "The AI Platform: Kubernetes, Schedulers and MLOps", desc: "GPUs on Kubernetes, autoscaling, cold starts, FinOps." },
    { n: 25, part: "IV",  file: "25-datacenters.html",  title: "Data Centers, Power and the Economics of Compute", desc: "Racks, power, cooling, TCO, and the token economy." },
    { n: 26, part: "V",   file: "26-industry.html",     title: "The AI Industry Map", desc: "Who builds what, how money flows, and the roles in AI infra." },
    { n: 27, part: "V",   file: "27-safety.html",       title: "Safety, Security, Interpretability and Policy", desc: "Risk categories, safety frameworks, AI security, regulation." },
    { n: 28, part: "V",   file: "28-frontier.html",     title: "The Frontier: September 2026", desc: "The state of the art, with sources." },
    { n: 29, part: "V",   file: "29-path.html",         title: "Your Path: Projects, Plan and Reading List", desc: "The project ladder, a 12-month plan, and what to read." },
    { n: 30, part: "A",   file: "30-glossary.html",     title: "Glossary", desc: "Every key term, with the chapter that explains it." }
  ];

  // Deep dives live in deep-dives/ and hang off a parent chapter, e.g.
  // { parent: 15, file: "15a-gpu-architecture.html", title: "GPU Architecture and the Performance Model" }
  var DEEP_DIVES = [
    { parent: 15, file: "15a-gpu-architecture.html", title: "GPU Architecture and the Performance Model" }
  ];
  // ---- Pinned third-party libraries (loaded from jsDelivr) ---------------
  var HLJS_SRC = "https://cdn.jsdelivr.net/npm/@highlightjs/cdn-assets@11.9.0/highlight.min.js";
  var MERMAID_SRC = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js";

  // ---- Small helpers ------------------------------------------------------
  var STORE_READ = "g2g-read";
  var STORE_THEME = "g2g-theme";
  function load(key, fallback) { try { var v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; } catch (e) { return fallback; } }
  function save(key, val) { try { localStorage.setItem(key, JSON.stringify(val)); } catch (e) {} }
  function el(tag, attrs, html) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (html != null) e.innerHTML = html;
    return e;
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function slug(s) { return s.toLowerCase().replace(/<[^>]+>/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60); }
  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var s = document.createElement("script"); s.src = src; s.onload = resolve; s.onerror = reject; document.head.appendChild(s);
    });
  }

  var body = document.body;
  var isIndex = body.dataset.page === "index";
  var chapNum = parseInt(body.dataset.chapter || "0", 10);
  var deepFile = body.dataset.deep || "";
  var root = isIndex ? "" : "../";
  var chapHref = function (c) { return root + "chapters/" + c.file; };
  var deepHref = function (d) { return root + "deep-dives/" + d.file; };
  var deepsOf = function (n) { return DEEP_DIVES.filter(function (d) { return d.parent === n; }); };
  var myDeep = DEEP_DIVES.filter(function (d) { return d.file === deepFile; })[0];
  if (myDeep) chapNum = 0;
  var read = load(STORE_READ, {});

  // ---- Theme: system -> light -> dark ------------------------------------
  var theme = load(STORE_THEME, "system");
  function applyTheme() {
    if (theme === "system") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
  }
  function isDark() {
    if (theme === "dark") return true;
    if (theme === "light") return false;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  applyTheme();

  // ---- Top bar -------------------------------------------------------------
  var topbar = el("header", { class: "topbar" });
  topbar.innerHTML =
    '<button class="icon-btn menu-btn" id="menuBtn" type="button" aria-expanded="false" aria-controls="sidebar">Contents</button>' +
    '<a class="brand" href="' + root + 'index.html">Gradient <em>to</em> Gigawatt</a>' +
    '<div class="progress" aria-hidden="true"><i id="bar"></i></div>' +
    '<button class="icon-btn" id="themeBtn" type="button"></button>';
  body.insertBefore(topbar, body.firstChild);
  var themeBtn = document.getElementById("themeBtn");
  function paintThemeBtn() { themeBtn.textContent = "Theme: " + theme; }
  paintThemeBtn();
  themeBtn.addEventListener("click", function () {
    theme = theme === "system" ? "light" : theme === "light" ? "dark" : "system";
    save(STORE_THEME, theme); applyTheme(); paintThemeBtn(); renderDiagrams();
  });

  // ---- Layout: sidebar + main ------------------------------------------------
  var content = document.querySelector("main") || document.querySelector("article");
  var layout = el("div", { class: "layout" });
  var sidebar = el("nav", { class: "sidebar", id: "sidebar", "aria-label": "Book navigation" });
  var main = document.querySelector("main");
  if (!main) {
    main = el("main");
    var col = el("div", { class: "col" });
    col.appendChild(content);
    main.appendChild(col);
  } else {
    main.parentNode.removeChild(main);
  }
  layout.appendChild(sidebar);
  layout.appendChild(main);
  body.insertBefore(layout, topbar.nextSibling);

  function chapterListHTML() {
    var html = "";
    PARTS.forEach(function (p) {
      html += '<div class="side-label">Part ' + p.id + " · " + esc(p.title) + '</div><ul class="side-list">';
      CHAPTERS.filter(function (c) { return c.part === p.id; }).forEach(function (c) {
        var cls = (c.n === chapNum ? "current " : "") + (read[c.n] ? "read" : "");
        html += '<li><a class="' + cls + '" href="' + chapHref(c) + '"><span class="n">' + (c.part === "A" ? "A" : c.n) + '</span><span class="t">' + esc(c.title) + '</span><span class="tick"></span></a>';
        deepsOf(c.n).forEach(function (d) {
          var dc = (myDeep === d ? "current " : "") + (read["d:" + d.file] ? "read" : "");
          html += '<a class="deep ' + dc + '" href="' + deepHref(d) + '"><span class="n">↳</span><span class="t">Deep dive: ' + esc(d.title) + '</span><span class="tick"></span></a>';
          if (myDeep === d) html += '<ul class="otp" id="otp"></ul>';
        });
        if (c.n === chapNum) html += '<ul class="otp" id="otp"></ul>';
        html += "</li>";
      });
      html += "</ul>";
    });
    return html;
  }
  sidebar.innerHTML = chapterListHTML();

  var menuBtn = document.getElementById("menuBtn");
  menuBtn.addEventListener("click", function () {
    var open = sidebar.classList.toggle("open"); menuBtn.setAttribute("aria-expanded", open);
  });
  sidebar.addEventListener("click", function (e) {
    if (e.target.closest("a")) { sidebar.classList.remove("open"); menuBtn.setAttribute("aria-expanded", "false"); }
  });

  // ---- Chapter page enhancements ---------------------------------------------
  var article = document.querySelector("article.chapter");
  var readKey = myDeep ? "d:" + myDeep.file : chapNum;
  if (article && (chapNum || myDeep)) {
    var me = myDeep ? CHAPTERS.filter(function (c) { return c.n === myDeep.parent; })[0]
                    : CHAPTERS.filter(function (c) { return c.n === chapNum; })[0];

    // Heading block
    var h1 = article.querySelector("h1");
    if (h1 && me && !article.querySelector(".chap-eyebrow")) {
      var part = PARTS.filter(function (p) { return p.id === me.part; })[0];
      var label = myDeep ? "Deep dive · Chapter " + me.n + " · " + esc(me.title)
                : part.id === "A" ? "Appendix" : "Chapter " + me.n + " · Part " + part.id + " · " + esc(part.title);
      var eyebrow = el("div", { class: "chap-eyebrow" }, label);
      h1.parentNode.insertBefore(eyebrow, h1);
    }
    document.title = (myDeep ? myDeep.title : me.title) + " · " + BOOK_TITLE;

    // On a chapter page, point to its deep dives
    if (!myDeep && deepsOf(me.n).length && h1) {
      var box = el("div", { class: "callout industry" }, "<strong>Go deeper</strong>" + deepsOf(me.n).map(function (d) {
        return '<p><a href="' + deepHref(d) + '">Deep dive: ' + esc(d.title) + "</a></p>"; }).join(""));
      var lede = article.querySelector(".lede") || h1;
      lede.parentNode.insertBefore(box, lede.nextSibling);
    }

    // IDs for headings + "on this page" list
    var otp = document.getElementById("otp");
    var seen = {};
    article.querySelectorAll("h2, h3").forEach(function (h) {
      if (!h.id) { var s = slug(h.textContent) || "section"; while (seen[s]) s += "-x"; h.id = s; }
      seen[h.id] = true;
      if (h.tagName === "H2" && otp) otp.appendChild(el("li", null, '<a href="#' + h.id + '">' + esc(h.textContent) + "</a>"));
    });

    // Footer: mark as read + prev/next
    var idx = CHAPTERS.indexOf(me);
    var prev = myDeep ? null : CHAPTERS[idx - 1], next = myDeep ? me : CHAPTERS[idx + 1];
    var foot = el("footer", { class: "chap-foot" });
    foot.innerHTML =
      '<button class="done-btn" type="button" id="doneBtn"></button>' +
      '<div class="pager">' +
      (prev ? '<a class="prev" href="' + chapHref(prev) + '"><small>Previous · Chapter ' + prev.n + "</small>" + esc(prev.title) + "</a>" : "<span></span>") +
      (next ? '<a class="next" href="' + chapHref(next) + '"><small>' + (myDeep ? "Back to" : "Next") + " · Chapter " + next.n + "</small>" + esc(next.title) + "</a>" : '<a class="next" href="../index.html"><small>Finished</small>Back to contents</a>') +
      "</div>";
    article.appendChild(foot);
    var doneBtn = document.getElementById("doneBtn");
    function paintDone() {
      var on = !!read[readKey];
      doneBtn.setAttribute("aria-pressed", on);
      doneBtn.textContent = on ? "Read ✓ (click to undo)" : "Mark chapter as read";
      var link = sidebar.querySelector("a.current"); if (link) link.classList.toggle("read", on);
    }
    doneBtn.addEventListener("click", function () { read[readKey] = !read[readKey]; save(STORE_READ, read); paintDone(); });
    paintDone();

    // Highlight the current section in "on this page"
    var h2s = [].slice.call(article.querySelectorAll("h2"));
    var otpLinks = otp ? [].slice.call(otp.querySelectorAll("a")) : [];
    var ticking = false;
    function onScroll() {
      ticking = false;
      var d = document.documentElement;
      document.getElementById("bar").style.width = Math.min(100, (d.scrollTop / Math.max(1, d.scrollHeight - d.clientHeight)) * 100) + "%";
      var cur = -1;
      for (var i = 0; i < h2s.length; i++) { if (h2s[i].getBoundingClientRect().top < 140) cur = i; else break; }
      otpLinks.forEach(function (a, i) { a.classList.toggle("active", i === cur); });
    }
    window.addEventListener("scroll", function () { if (!ticking) { ticking = true; requestAnimationFrame(onScroll); } }, { passive: true });
    onScroll();
  }

  // ---- Index page: build the table of contents -----------------------------
  var tocMount = document.getElementById("toc");
  if (isIndex && tocMount) {
    var html = "";
    PARTS.forEach(function (p) {
      html += '<section class="part-block"><h2>Part ' + p.id + '</h2><div class="part-title">' + esc(p.title) + '</div><ul class="toc-list">';
      CHAPTERS.filter(function (c) { return c.part === p.id; }).forEach(function (c) {
        html += '<li><a class="' + (read[c.n] ? "read" : "") + '" href="chapters/' + c.file + '"><span class="n">' + (c.part === "A" ? "A" : String(c.n).padStart(2, "0")) +
          '</span><span class="t">' + esc(c.title) + '<span class="d">' + esc(c.desc) + '</span></span><span class="s">' + (read[c.n] ? "Read ✓" : "") + "</span></a>";
        deepsOf(c.n).forEach(function (d) {
          var r = read["d:" + d.file];
          html += '<a class="deep ' + (r ? "read" : "") + '" href="deep-dives/' + d.file + '"><span class="n">↳</span><span class="t">Deep dive: ' + esc(d.title) + '</span><span class="s">' + (r ? "Read ✓" : "") + "</span></a>";
        });
        html += "</li>";
      });
      html += "</ul></section>";
    });
    tocMount.innerHTML = html;
    var count = document.getElementById("readCount");
    if (count) count.textContent = CHAPTERS.filter(function (c) { return read[c.n]; }).length + " / " + CHAPTERS.length;
    var bar = document.getElementById("bar"); if (bar) bar.style.width = "0";
  }

  // ---- Tables scroll sideways on small screens --------------------------------
  document.querySelectorAll("article table, main table").forEach(function (t) {
    if (t.parentNode.classList && t.parentNode.classList.contains("tbl")) return;
    var w = el("div", { class: "tbl" }); t.parentNode.insertBefore(w, t); w.appendChild(t);
  });

  // ---- Copy buttons on code blocks ---------------------------------------------
  document.querySelectorAll("pre > code").forEach(function (code) {
    var pre = code.parentNode;
    var btn = el("button", { class: "copy-btn", type: "button" }, "Copy");
    btn.addEventListener("click", function () {
      var text = code.innerText;
      var done = function () { btn.textContent = "Copied"; setTimeout(function () { btn.textContent = "Copy"; }, 1400); };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { selectText(code); });
      } else { selectText(code); }
    });
    pre.appendChild(btn);
  });
  function selectText(node) { var r = document.createRange(); r.selectNodeContents(node); var s = getSelection(); s.removeAllRanges(); s.addRange(r); }

  // ---- Syntax highlighting (needs internet the first time) -------------------------
  if (document.querySelector('pre code[class*="language-"]')) {
    loadScript(HLJS_SRC).then(function () {
      document.querySelectorAll('pre code[class*="language-"]').forEach(function (c) {
        if (c.classList.contains("language-cuda")) { c.classList.remove("language-cuda"); c.classList.add("language-cpp"); }
        try { window.hljs.highlightElement(c); } catch (e) {}
      });
    }).catch(function () { /* offline: code stays readable, just uncoloured */ });
  }

  // ---- Diagrams (Mermaid) ---------------------------------------------------------
  var diagrams = [].slice.call(document.querySelectorAll("pre.mermaid"));
  diagrams.forEach(function (d) { d.dataset.src = d.textContent; });
  var mermaidReady = null;
  function renderDiagrams() {
    if (!diagrams.length) return;
    mermaidReady = mermaidReady || loadScript(MERMAID_SRC);
    mermaidReady.then(function () {
      var m = window.mermaid;
      m.initialize({
        startOnLoad: false, securityLevel: "strict",
        theme: isDark() ? "dark" : "neutral",
        fontFamily: '"IBM Plex Sans", system-ui, sans-serif',
        flowchart: { htmlLabels: true, curve: "basis" }
      });
      diagrams.forEach(function (d) { d.removeAttribute("data-processed"); d.innerHTML = ""; d.textContent = d.dataset.src; });
      return m.run({ nodes: diagrams, suppressErrors: true });
    }).catch(function () {
      diagrams.forEach(function (d) {
        if (!d.querySelector("svg")) d.insertAdjacentHTML("beforebegin", '<p class="figcaption">Diagram needs an internet connection to render. Source shown below.</p>');
      });
    });
  }
  renderDiagrams();
  if (window.matchMedia) {
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    var onChange = function () { if (theme === "system") renderDiagrams(); };
    if (mq.addEventListener) mq.addEventListener("change", onChange); else if (mq.addListener) mq.addListener(onChange);
  }
})();
