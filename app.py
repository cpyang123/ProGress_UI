"""app.py – ProGress Music Generation Demo"""
from __future__ import annotations

# ZeroGPU: import `spaces` before torch so its CUDA-emulation hooks install
# first.  No-op when not on ZeroGPU; absent (and skipped) in local dev.
try:
    import spaces  # noqa: F401
except Exception:
    pass

import os
import random
import sys
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import backend

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;550;600;650;700;750&display=swap');

:root {
  --pg-ink:        #0f172a;   /* headings / primary text */
  --pg-body:       #334155;   /* body text                */
  --pg-muted:      #64748b;   /* secondary / captions     */
  --pg-line:       #e2e8f0;   /* hairline borders         */
  --pg-line-soft:  #eef2f6;   /* very light dividers      */
  --pg-surface:    #ffffff;   /* card surface             */
  --pg-tint:       #f8fafc;   /* page / subtle fill       */
  --pg-accent:     #2563eb;   /* brand blue               */
  --pg-accent-dk:  #1d4ed8;
  --pg-shadow:     0 1px 2px rgba(15,23,42,.04), 0 1px 3px rgba(15,23,42,.06);
  --pg-shadow-md:  0 4px 12px rgba(15,23,42,.08);
}

/* Center the whole app in a constrained column instead of edge-to-edge. */
.gradio-container, .gradio-container input, .gradio-container button,
.gradio-container textarea, .gradio-container select {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
}
.gradio-container {
  font-size: 16px !important;
  max-width: 940px !important;
  margin: 0 auto !important;
  padding: 8px 28px 56px !important;
  color: var(--pg-body);
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

/* Typographic scale — restrained, consistent. */
.gr-markdown p, .gr-markdown li, .prose p, .prose li { font-size: 1rem; line-height: 1.6; }
.gr-markdown ol li, .prose ol li { margin-bottom: 6px; }
.gr-markdown h2, .prose h2 {
  font-size: 1.45rem !important; font-weight: 650; color: var(--pg-ink);
  letter-spacing: -.01em; margin: 4px 0 14px; padding-bottom: 10px;
  border-bottom: 1px solid var(--pg-line-soft);
}
.gr-markdown h3, .prose h3 { font-size: 1.1rem !important; font-weight: 600; color: var(--pg-ink); }
label, .gr-input-label { font-size: .95rem !important; color: var(--pg-body); }

/* Page background (handled in CSS because this Gradio build crashes on a
   .set()-customized theme — see app.py launch() note).
   IMPORTANT: do NOT blanket-style .block / .form here.  In Gradio 6 those
   classes sit on nearly every wrapper (markdown, the header, each slider…),
   so adding borders/shadows there boxes the whole page into nested cards with
   uneven margins.  Surfaces come from the theme defaults plus the explicit
   .stat-box / .section-card classes below. */
body, gradio-app, .gradio-container { background: var(--pg-tint) !important; }

/* Real panels (inputs, dataframe, accordions, code) sit on white; plain
   text/markdown/button rows stay transparent on the page tint. */
.gr-input, .gr-box, .gr-panel,
.cm-editor, [data-testid="textbox"], [data-testid="dataframe"] {
  border-radius: 10px !important;
}
input, textarea, select { border-radius: 9px !important; border-color: #d8dee8 !important; }

/* Primary / secondary button colours */
button.primary {
  background: var(--pg-accent) !important; color: #fff !important;
  border: none !important;
}
button.primary:hover { background: var(--pg-accent-dk) !important; }
button.secondary {
  background: var(--pg-surface) !important; color: var(--pg-body) !important;
  border: 1px solid #d8dee8 !important;
}
button.secondary:hover { background: #f1f5f9 !important; }

/* Action buttons: medium weight, calm radius, subtle motion.  Scoped to
   primary/secondary so Gradio's internal buttons keep their layout. */
button.primary, button.secondary {
  flex-grow: 0 !important;
  font-size: .98rem !important;
  font-weight: 600 !important;
  min-height: 42px;
  border-radius: 9px !important;
  letter-spacing: .005em;
  transition: transform .08s ease, box-shadow .15s ease, background .15s ease;
}
button.primary { box-shadow: var(--pg-shadow); }
button.primary:hover  { box-shadow: var(--pg-shadow-md); transform: translateY(-1px); }
button.primary:active { transform: translateY(0); }
button.secondary:hover { transform: translateY(-1px); }

button.lg {
  font-size: 1.02rem !important;
  padding: 12px 26px !important;
  min-height: 50px;
  width: auto !important; min-width: 230px;
  flex: 0 0 auto !important;
}
button.sm {
  min-height: 34px; font-size: .9rem !important;
  width: fit-content !important; border-radius: 8px !important;
}

/* Evenly space the three workflow tabs across the full width.
   gradio 6 decides tab overflow (the "…" menu) by summing the button widths in a
   HIDDEN ".tab-container.visually-hidden" measuring copy and comparing to the
   wrapper width.  So we must spread ONLY the visible nav and leave the hidden
   measurer at its natural width — restyling that copy makes it over-measure and
   push the last tab into the overflow menu. */
.tab-container:not(.visually-hidden) { width: 100% !important; }
.tab-container:not(.visually-hidden) > button {
  flex: 1 1 0 !important; min-width: 0 !important;
  justify-content: center !important;
}
/* Older gradio builds use .tab-nav (no hidden measuring copy). */
.tab-nav { display: flex !important; }
.tab-nav button { flex: 1 1 0 !important; justify-content: center !important; }
.tab-nav button, .tab-container button {
  font-size: 1rem !important; font-weight: 550 !important;
}

/* Accordion headers (How it works, Generation parameters, Sections) */
.label-wrap span { font-size: 1.02rem !important; font-weight: 600; color: var(--pg-ink); }

/* Header */
#header { text-align:center; padding: 30px 0 18px; margin-bottom: 6px; }
#header h1 {
  margin: 0; font-size: 2.7rem; font-weight: 750;
  color: var(--pg-ink); letter-spacing: -.025em;
  cursor: pointer; user-select: none; -webkit-user-select: none;
  /* Fixed, centered hit area so the hover zone doesn't shrink under the cursor
     while the text backspaces (otherwise the effect reverses mid-purr).
     min-height keeps the box from collapsing to 0 when the text is fully
     deleted — without it the header (and page) jumps up mid-animation. */
  display: inline-block; min-width: 15em; text-align: center;
  line-height: 1.2; min-height: 1.2em;
}
#header .tagline {
  margin: 8px 0 0; font-size: 1.02rem; color: var(--pg-muted); font-weight: 400;
}
#header .device-badge {
  margin: 10px 0 0; font-size: .82rem; font-weight: 550; color: var(--pg-muted);
  display: inline-block; padding: 3px 12px; border: 1px solid var(--pg-line);
  border-radius: 999px; background: var(--pg-surface); letter-spacing: .01em;
}

.step-sub { color: var(--pg-muted); font-size: 1rem; margin: 0 0 16px; }

/* Stat cards: clean white surface, soft shadow, gentle hover lift. */
.stat-row { display: flex; gap: 16px; margin: 4px 0; }
.stat-row .stat-box { flex: 1 1 0; }
.stat-box {
  background: var(--pg-surface); border: 1px solid var(--pg-line);
  border-radius: 12px; padding: 20px 18px; text-align: center;
  box-shadow: var(--pg-shadow); transition: box-shadow .15s ease, transform .1s ease;
}
.stat-box:hover { box-shadow: var(--pg-shadow-md); transform: translateY(-2px); }
.stat-box .stat-num {
  font-size: 2.4rem; font-weight: 700; color: var(--pg-accent);
  line-height: 1; letter-spacing: -.02em;
}
.stat-box .stat-lbl {
  font-size: .82rem; color: var(--pg-muted); margin-top: 8px;
  text-transform: uppercase; letter-spacing: .06em; font-weight: 600;
}

/* Section cards */
.section-card {
  border: 1px solid var(--pg-line); border-radius: 12px;
  padding: 16px 18px; margin-bottom: 14px; background: var(--pg-surface);
  box-shadow: var(--pg-shadow);
}
.section-label { font-size: 1.05rem; font-weight: 650; color: var(--pg-ink); margin-bottom: 6px; }
.section-meta  { font-size: .92rem; color: var(--pg-muted); margin-bottom: 10px; }

/* Data table: lighter grid, readable rows, clear hover affordance. */
table th {
  font-size: .82rem !important; text-transform: uppercase; letter-spacing: .05em;
  color: var(--pg-muted) !important; font-weight: 600 !important;
  background: var(--pg-tint) !important; padding: 12px 14px !important;
}
table td { font-size: .98rem !important; padding: 11px 14px !important; color: var(--pg-body); }
table tbody tr { transition: background .1s ease; }
table tbody tr:hover td { background: #eff6ff !important; cursor: pointer; }

/* Accordions shouldn't show a horizontal scrollbar for their text/sliders.
   (accordion-content is a data-testid in Gradio 6, not a class) */
div[data-testid="accordion-content"],
div[data-testid="accordion-content"] *,
.block:has(> div[data-testid="accordion-content"]) {
  overflow-x: clip !important;
}
"""

HOW_IT_WORKS_MD = """
**ProGress** builds complete two-voice compositions in three steps:

1. **Generate** — *SchenkerDiff*, a discrete graph-diffusion model trained on
   Schenkerian analyses, samples short phrases as note graphs.  Each phrase is
   screened by music-theory filters (illegal harmonics on strong beats, bad
   mode mixture, bad counterpoint); only phrases that pass enter the pool.
   You can also load the pre-generated phrase set bundled with the paper's
   supplement instead of running the model.

2. **Browse & Select** — listen to phrases from the pool and pick one that
   starts on the tonic.  It becomes the locked *opening* of your piece.

3. **Compose** — a harmonic structure (e.g. I–IV–V–I) is chosen or randomly
   drawn.  The remaining sections are sampled from the pool, transposed to
   follow the progression, joined with voice-leading–aware stitching, and
   inner voices are filled in to complete the harmony.  The result is shown
   as sheet music and playable MIDI.

Paper: [ProGress: Structured Music Generation via Graph Diffusion and
Hierarchical Music Analysis](https://arxiv.org/abs/2510.10249)
"""

BIBTEX = """@article{nihahn2025progress,
  title={ProGress: Structured Music Generation via Graph Diffusion and Hierarchical Music Analysis},
  author={Ni-Hahn, Stephen and Yang, Chao P\\'eter and Ma, Mingchen and Rudin, Cynthia and Mak, Simon and Jiang, Yue},
  journal={arXiv preprint arXiv:2510.10249},
  year={2025}
}"""

# ── HTML helpers ──────────────────────────────────────────────────────────────

def _stat_html(num: str, lbl: str) -> str:
    return (
        f'<div class="stat-box">'
        f'<div class="stat-num">{num}</div>'
        f'<div class="stat-lbl">{lbl}</div>'
        f'</div>'
    )


def _section_card(label: str, pid: int | None, pool: dict | None) -> str:
    if pid is None or pool is None:
        return f"<div class='section-card'><div class='section-label'>{label}</div><em>—</em></div>"
    info = pool["info"][pid]
    try:
        midi_bytes = backend.score_to_midi_bytes(pool["scores"][pid])
        player = backend.midi_player_html(midi_bytes, height="60px", show_viz=False)
    except Exception as e:
        player = f"<em>Audio unavailable: {e}</em>"
    mode = info["mode"].capitalize()
    return (
        f"<div class='section-card'>"
        f"<div class='section-label'>{label}</div>"
        f"<div class='section-meta'>Phrase #{pid} &nbsp;·&nbsp; {mode}</div>"
        f"{player}</div>"
    )


_TABLE_COLS = ["ID", "Mode", "Harmonic variety"]


def _phrases_df(pool: dict | None, sort_by: str = "Harmonic variety ↓",
                tonic_only: bool = True) -> pd.DataFrame:
    if pool is None or not pool.get("info"):
        return pd.DataFrame(columns=_TABLE_COLS)
    rows = [
        {"ID": p["id"], "Mode": p["mode"].capitalize(),
         "Harmonic variety": p["quality"]}
        for p in pool["info"]
        if not tonic_only or p["start_rn"] in ("I", "i")
    ]
    if not rows:
        return pd.DataFrame(columns=_TABLE_COLS)
    df = pd.DataFrame(rows)
    if sort_by == "Harmonic variety ↓":
        df = df.sort_values("Harmonic variety", ascending=False)
    elif sort_by == "Harmonic variety ↑":
        df = df.sort_values("Harmonic variety", ascending=True)
    elif sort_by == "ID":
        df = df.sort_values("ID")
    elif sort_by == "Mode":
        df = df.sort_values(["Mode", "Harmonic variety"], ascending=[True, False])
    return df.reset_index(drop=True)


def _stat_row(pool: dict | None) -> str:
    """All three stat cards as one HTML block; empty string when no pool."""
    if not pool or not pool.get("scores"):
        return ""
    n   = len(pool["scores"])
    n_I = len(pool["starts"].get("I", []))
    n_i = len(pool["starts"].get("i", []))
    return (
        '<div class="stat-row">'
        + _stat_html(str(n),   "Total phrases")
        + _stat_html(str(n_I), "Major openings")
        + _stat_html(str(n_i), "Minor openings")
        + '</div>'
    )


def _selected_label(pid: int | None, info: dict | None) -> str:
    if pid is None or info is None:
        return "*No opening phrase selected yet.*"
    return f"### Opening phrase: #{pid} ({info['mode'].capitalize()})"


# ── Easter eggs 🐱 ──────────────────────────────────────────────────────────────
# Injected via the header gr.HTML's head= (the same mechanism that loads the MIDI
# / OSMD scripts), because gradio 6's .load(js=) didn't run reliably here.  The
# listeners are document-delegated, so they work the moment the page exists:
#   (1) console greeting, (2) hover the logo → it retypes as "🐱 PuuurrrrrGressssss 🐱",
#   (3) Konami code (↑↑↓↓←→←→ B A) → a shower of cats, (4) a cat strolls by.
CAT_EGGS_HEAD = r"""
<script>
(function () {
  if (window.__catEggs) return; window.__catEggs = true;

  console.log(
    "%c /\\_/\\   ProGress\n( o.o )  cat-powered graph diffusion =^.^=\n > ^ < ",
    "color:#2563eb;font-weight:bold;"
  );

  // Logo: hover → backspace "ProGress" then type "🐱 PuuurrrrrGressssss 🐱";
  // reverse on mouse-out.  Array.from keeps the cat emoji a single unit so the
  // backspace never slices it in half.
  var ORIG  = Array.from('ProGress');
  var FANCY = Array.from('🐱 PuuurrrrrGressssss 🐱');
  var logoTimers = [], logoHover = false;
  function logoClear() { logoTimers.forEach(clearTimeout); logoTimers = []; }
  function logoAnim(h1, target) {
    logoClear();
    var cur = Array.from(h1.textContent);
    (function back() {
      if (cur.length) { cur.pop(); h1.textContent = cur.join(''); logoTimers.push(setTimeout(back, 75)); }
      else { var i = 0; (function type() {
        if (i < target.length) { i++; h1.textContent = target.slice(0, i).join(''); logoTimers.push(setTimeout(type, 60)); }
      })(); }
    })();
  }
  document.addEventListener('mouseover', function (e) {
    var h1 = e.target.closest ? e.target.closest('#header h1') : null;
    if (!h1 || logoHover) return;
    logoHover = true; logoAnim(h1, FANCY);
  });
  document.addEventListener('mouseout', function (e) {
    var h1 = e.target.closest ? e.target.closest('#header h1') : null;
    if (!h1 || !logoHover) return;
    logoHover = false; logoAnim(h1, ORIG);
  });

  // Konami code -> cat rain (e.key based)
  var seq = ['arrowup','arrowup','arrowdown','arrowdown',
             'arrowleft','arrowright','arrowleft','arrowright','b','a'];
  var i = 0;
  document.addEventListener('keydown', function (e) {
    var key = (e.key || '').toLowerCase();
    i = (key === seq[i]) ? i + 1 : (key === seq[0] ? 1 : 0);
    if (i !== seq.length) return;
    i = 0;
    var GLYPHS = ['🐱','🐈','😺','😻','🐾','🎵','🎶','🎼','🎵','🎶'];
    function dropCat() {
      var c = document.createElement('div');
      c.textContent = GLYPHS[Math.floor(Math.random()*GLYPHS.length)];
      var dur = 2.6 + Math.random()*2.2;  // each glyph falls at its own speed
      c.style.cssText =
        'position:fixed;top:-48px;left:' + (Math.random()*100) + 'vw;' +
        'font-size:' + (20 + Math.random()*26) + 'px;z-index:99999;' +
        'pointer-events:none;transition:transform ' + dur + 's linear,opacity ' + dur + 's;';
      document.body.appendChild(c);
      requestAnimationFrame(function () {
        c.style.transform = 'translateY(' + (window.innerHeight + 96) +
          'px) rotate(' + ((Math.random()-0.5)*420) + 'deg)';
        c.style.opacity = '0';
      });
      setTimeout(function () { c.remove(); }, dur*1000 + 200);
    }
    // ~30 cats sprinkled over ~2.5s so it rains rather than dropping in one line
    for (var n = 0; n < 30; n++) setTimeout(dropCat, Math.random()*2500);
  });

  // Wandering cat: an animated 🐈 trots across the bottom every few minutes.
  // The bob/tilt walk-cycle is pure CSS (no sprite asset).
  var walkStyle = document.createElement('style');
  walkStyle.textContent =
    '@keyframes pgCatBob{0%,100%{transform:translateY(0) rotate(-5deg);}' +
    '50%{transform:translateY(-7px) rotate(5deg);}}';
  document.head.appendChild(walkStyle);
  function catStroll() {
    if (!document.body) return;
    var ltr = Math.random() < 0.5;                 // travel direction
    var w = window.innerWidth + 90;
    var from = ltr ? -90 : w, to = ltr ? w : -90;
    var wrap = document.createElement('div');
    wrap.style.cssText =
      'position:fixed;bottom:6px;left:0;z-index:99998;pointer-events:none;will-change:transform;';
    var facer = document.createElement('span');   // flip to face travel direction
    facer.style.cssText = 'display:inline-block;font-size:34px;transform:scaleX(' + (ltr ? -1 : 1) + ');';
    var bob = document.createElement('span');     // the walk-cycle bob/tilt
    bob.textContent = '🐈';
    bob.style.cssText = 'display:inline-block;animation:pgCatBob .45s ease-in-out infinite;';
    facer.appendChild(bob); wrap.appendChild(facer); document.body.appendChild(wrap);
    wrap.animate(
      [{ transform: 'translateX(' + from + 'px)' }, { transform: 'translateX(' + to + 'px)' }],
      { duration: 11000 + Math.random() * 5000, easing: 'linear' }
    ).onfinish = function () { wrap.remove(); };
  }
  window.catStroll = catStroll;                    // trigger on demand from the console
  (function loop() {
    setTimeout(function () { catStroll(); loop(); }, 180000 + Math.random() * 120000); // every 3–5 min
  })();
})();
</script>
"""


# ── App ───────────────────────────────────────────────────────────────────────

def create_app() -> gr.Blocks:

    with gr.Blocks(title="ProGress") as demo:

        # ── State ──────────────────────────────────────────────────────────────
        pool_state      = gr.State(None)   # phrases_data dict
        starting_id     = gr.State(None)   # locked beginning phrase id
        preview_pid     = gr.State(None)   # currently previewed phrase id
        sorted_df_state = gr.State(None)   # last Python-sorted df (for click lookup)

        # ── Header ─────────────────────────────────────────────────────────────
        gr.HTML(
            '<div id="header">'
            '<h1>ProGress</h1>'
            '<p class="tagline">Structured music generation via graph diffusion '
            'and hierarchical music analysis</p>'
            f'<p class="device-badge">Compute: {backend.device_info()}</p>'
            '</div>',
            head=backend.MIDI_PLAYER_HEAD + backend.OSMD_HEAD + CAT_EGGS_HEAD,
        )

        with gr.Tabs() as tabs:

            # ═════════════════════════════════════════════════════════════════
            # TAB 1  ·  Generate
            # ═════════════════════════════════════════════════════════════════
            with gr.Tab("1.  Generate", id="generate"):

                gr.Markdown(
                    "## Step 1 — Build your phrase pool\n"
                    "1. Click **Generate** to create phrases with the model, "
                    "or **Load pre-generated phrases** to start instantly.\n"
                    "2. Wait until the phrase counts appear below.\n"
                    "3. Press **Next** to go pick your opening phrase."
                )

                with gr.Accordion("How it works", open=False):
                    gr.Markdown(HOW_IT_WORKS_MD)

                ckpt_ok = backend.checkpoint_available()
                if not ckpt_ok:
                    gr.Markdown(
                        f"**Checkpoint not found** at `{backend.CHECKPOINT_PATH}`. "
                        "Generation is disabled — you can still load the "
                        "pre-generated phrase set below."
                    )

                with gr.Accordion("Generation parameters", open=False):
                    with gr.Row():
                        target_slider = gr.Slider(
                            minimum=8, maximum=500, value=100, step=4,
                            label="Target phrases",
                            info="Total valid phrases to generate. More = better variety but slower (~10–15 s each on CPU).",
                            scale=3,
                        )
                        batch_slider = gr.Slider(
                            minimum=4, maximum=32, value=8, step=4,
                            label="Batch size",
                            info="Phrases per model call. Larger batches use more memory.",
                            scale=2,
                        )

                with gr.Row():
                    gen_btn   = gr.Button("Generate", variant="primary",
                                          size="lg", interactive=ckpt_ok, scale=3)
                    load_btn  = gr.Button("Load pre-generated phrases",
                                          size="lg", scale=2)
                with gr.Row():
                    reset_btn = gr.Button("Reset pool", size="sm",
                                          variant="secondary", scale=0)

                gen_status = gr.Markdown("")

                # Single HTML block holding all three stat cards.  Keeping them
                # in one component means one loading state while phrases process
                # (three separate gr.HTML outputs each render their own spinner —
                # the "triplicated waiting icon").
                stat_row = gr.HTML("")

                next1_btn = gr.Button("Next: Browse & Select →",
                                      variant="primary", size="lg", visible=False)

                gr.Markdown("### Cite this work")
                gr.Code(BIBTEX, language=None, label=None, interactive=False)

            # ═════════════════════════════════════════════════════════════════
            # TAB 2  ·  Browse & Select
            # ═════════════════════════════════════════════════════════════════
            with gr.Tab("2.  Browse & Select", id="browse"):

                gr.Markdown(
                    "## Step 2 — Choose your opening phrase\n"
                    "1. Click a row in the table to hear the phrase and see its score.\n"
                    "2. When you find one you like, press **Use as opening phrase**."
                )

                with gr.Row():
                    sort_radio = gr.Radio(
                        ["Harmonic variety ↓", "Harmonic variety ↑", "ID", "Mode"],
                        value="Harmonic variety ↓",
                        label="Sort by",
                        scale=3,
                    )
                    tonic_check = gr.Checkbox(
                        value=True,
                        label="Opening phrases only",
                        info="Show only phrases that begin on the tonic. Untick to browse the whole pool.",
                        scale=2,
                    )

                phrase_table = gr.Dataframe(
                    headers=_TABLE_COLS,
                    datatype=["number", "str", "number"],
                    interactive=False,
                    label="Click a row to preview",
                    max_height=300,
                )

                preview_player   = gr.HTML("<em>Click a row above to preview a phrase.</em>",
                                           js_on_load=backend.MIDI_VIZ_JS_ON_LOAD)
                preview_score    = gr.HTML("", js_on_load=backend.OSMD_JS_ON_LOAD)
                preview_download = gr.DownloadButton("Download preview MIDI",
                                                     size="sm", visible=False)

                confirm_btn  = gr.Button("Use as opening phrase", variant="primary",
                                         size="lg", interactive=False)
                selection_md = gr.Markdown(_selected_label(None, None))

                next2_btn = gr.Button("Next: Compose →", variant="primary",
                                      size="lg", visible=False, elem_id="next2-btn")

            # ═════════════════════════════════════════════════════════════════
            # TAB 3  ·  Compose
            # ═════════════════════════════════════════════════════════════════
            with gr.Tab("3.  Compose", id="compose"):

                gr.Markdown(
                    "## Step 3 — Compose the full piece\n"
                    "1. Pick a harmonic structure, or leave it on **Random**.\n"
                    "2. Press **Compose**. Not happy? **Resample** keeps your "
                    "opening and redraws the rest.\n"
                    "3. Listen, read the score, and download the MIDI."
                )

                selection_summary = gr.Markdown(_selected_label(None, None))

                with gr.Row():
                    structure_dd = gr.Dropdown(
                        ["Random"] + backend.STRUCTURE_NAMES,
                        value="Random",
                        label="Harmonic structure",
                        scale=3,
                    )
                    stitch_btn   = gr.Button("Compose", variant="primary",
                                             size="lg", interactive=False, scale=1)
                    resample_btn = gr.Button("Resample", size="lg",
                                             interactive=False, scale=1)

                stitch_status = gr.Markdown("")

                with gr.Accordion("Sections", open=False):
                    opening_html = gr.HTML("<em>—</em>")
                    develop_html = gr.HTML("<em>—</em>")
                    expand_html  = gr.HTML("<em>—</em>")
                    closing_html = gr.HTML("<em>—</em>")

                gr.Markdown("### Full Composition")
                full_score   = gr.HTML("<em>—</em>", js_on_load=backend.OSMD_JS_ON_LOAD)
                full_player  = gr.HTML("<em>—</em>", js_on_load=backend.MIDI_VIZ_JS_ON_LOAD)
                download_btn = gr.DownloadButton("Download MIDI", visible=False)

        # ════════════════════════════════════════════════════════════════════
        # Handlers
        # ════════════════════════════════════════════════════════════════════

        # ── Generate ──────────────────────────────────────────────────────────
        def _generate(target, batch_size, sort_by, tonic_only, pool,
                      progress=gr.Progress()):
            if not backend.checkpoint_available():
                df = _phrases_df(None)
                return (
                    pool, "Checkpoint missing.",
                    _stat_row(pool),
                    df, df,
                    gr.update(visible=False),
                )
            try:
                pool = backend.generate_until_target(
                    int(target), int(batch_size), progress=progress, pool=pool,
                )
            except Exception as e:
                n  = len(pool["scores"]) if pool else 0
                df = _phrases_df(pool, sort_by, tonic_only)
                return (
                    pool, f"Error: {e}",
                    _stat_row(pool),
                    df, df,
                    gr.update(visible=n > 0),
                )
            n  = len(pool["scores"])
            df = _phrases_df(pool, sort_by, tonic_only)
            return (
                pool,
                f"{n} phrases ready.",
                _stat_row(pool),
                df, df,
                gr.update(visible=n > 0),
            )

        gen_outputs = [
            pool_state, gen_status, stat_row,
            phrase_table, sorted_df_state, next1_btn,
        ]
        gen_btn.click(
            fn=_generate,
            inputs=[target_slider, batch_slider, sort_radio, tonic_check, pool_state],
            outputs=gen_outputs,
        )

        # ── Load pre-generated phrases (ProGress_Supplement) ─────────────────
        def _load_pregen(sort_by, tonic_only, pool, progress=gr.Progress()):
            try:
                pool = backend.load_pregenerated(pool=pool, progress=progress)
            except Exception as e:
                n  = len(pool["scores"]) if pool else 0
                df = _phrases_df(pool, sort_by, tonic_only)
                return (
                    pool, f"Could not load pre-generated phrases: {e}",
                    _stat_row(pool),
                    df, df,
                    gr.update(visible=n > 0),
                )
            n  = len(pool["scores"])
            df = _phrases_df(pool, sort_by, tonic_only)
            return (
                pool,
                f"{n} phrases ready (pre-generated set loaded).",
                _stat_row(pool),
                df, df,
                gr.update(visible=n > 0),
            )

        load_btn.click(
            fn=_load_pregen,
            inputs=[sort_radio, tonic_check, pool_state],
            outputs=gen_outputs,
        )

        # ── Reset ─────────────────────────────────────────────────────────────
        def _reset():
            empty = _phrases_df(None)
            return (
                None, "",
                "",                            # stat row
                empty, empty,
                gr.update(visible=False),      # next1_btn
                None,                          # preview_pid
                "<em>Pool cleared. Generate to start.</em>",
                "", gr.update(visible=False),  # preview_score, preview_download
                None,                          # starting_id
                _selected_label(None, None),   # selection_md
                _selected_label(None, None),   # selection_summary
                gr.update(visible=False),      # next2_btn
                gr.update(interactive=False),  # confirm_btn
                gr.update(interactive=False),  # stitch_btn
                gr.update(interactive=False),  # resample_btn
            )

        reset_btn.click(
            fn=_reset,
            outputs=[
                pool_state, gen_status, stat_row,
                phrase_table, sorted_df_state, next1_btn,
                preview_pid, preview_player, preview_score, preview_download,
                starting_id, selection_md, selection_summary, next2_btn,
                confirm_btn, stitch_btn, resample_btn,
            ],
        )

        # ── Sort / filter ─────────────────────────────────────────────────────
        def _sort_table(pool, sort_by, tonic_only):
            df = _phrases_df(pool, sort_by, tonic_only)
            return df, df

        sort_radio.change(
            fn=_sort_table,
            inputs=[pool_state, sort_radio, tonic_check],
            outputs=[phrase_table, sorted_df_state],
        )
        tonic_check.change(
            fn=_sort_table,
            inputs=[pool_state, sort_radio, tonic_check],
            outputs=[phrase_table, sorted_df_state],
        )

        # ── Preview (shared logic) ────────────────────────────────────────────
        def _preview_fail(msg):
            return (None, f"<em>{msg}</em>", "", gr.update(visible=False),
                    gr.update(interactive=False))

        def _do_preview(pool, pid):
            if pool is None:
                return _preview_fail("Generate phrases first (Generate tab).")
            if pid is None or pid < 0 or pid >= len(pool["scores"]):
                return _preview_fail("Invalid phrase.")
            try:
                score      = pool["scores"][pid]
                midi_bytes = backend.score_to_midi_bytes(score)
                info       = pool["info"][pid]
                tf = tempfile.NamedTemporaryFile(
                    suffix=".mid", delete=False, prefix=f"phrase_{pid}_"
                )
                tf.write(midi_bytes); tf.close()
                header = (
                    f"<div style='font-size:1.3rem;font-weight:600;color:#1e293b;margin-bottom:8px;'>"
                    f"Phrase #{pid} &nbsp;·&nbsp; {info['mode'].capitalize()}</div>"
                )
                score_html = backend.sheet_music_html(score)
                return (
                    pid,
                    header + backend.midi_player_html(midi_bytes),
                    score_html,
                    gr.update(value=tf.name, visible=True),
                    gr.update(interactive=True),
                )
            except Exception as e:
                import traceback
                traceback.print_exc()   # full traceback to server / Space logs
                return _preview_fail(f"Preview error: {type(e).__name__}: {e}")

        # ── Row click → preview ───────────────────────────────────────────────
        def _on_row_click(evt: gr.SelectData, pool, df_state):
            if pool is None or df_state is None:
                return _preview_fail("Generate phrases first.")
            row_idx = evt.index[0]
            try:
                if isinstance(df_state, pd.DataFrame):
                    pid = int(df_state.iloc[row_idx]["ID"])
                else:
                    pid = int(df_state[row_idx][0])
            except (IndexError, KeyError, ValueError):
                return _preview_fail("Could not read selection.")
            return _do_preview(pool, pid)

        phrase_table.select(
            fn=_on_row_click,
            inputs=[pool_state, sorted_df_state],
            outputs=[preview_pid, preview_player, preview_score,
                     preview_download, confirm_btn],
        )

        # ── Lock opening phrase ───────────────────────────────────────────────
        def _confirm(pool, pid):
            if pool is None or pid is None:
                msg = _selected_label(None, None)
                return (None, msg, msg, gr.update(visible=False),
                        gr.update(interactive=False), gr.update(interactive=False))
            info = pool["info"][pid]
            if info["start_rn"] not in ("I", "i"):
                msg = (
                    f"Phrase #{pid} starts on **{info['start_rn']}** — "
                    "please select a phrase that begins on a major (I) or minor (i) tonic."
                )
                return (None, msg, msg, gr.update(visible=False),
                        gr.update(interactive=False), gr.update(interactive=False))
            msg = _selected_label(pid, info)
            return (pid, msg, msg, gr.update(visible=True),
                    gr.update(interactive=True), gr.update(interactive=True))

        confirm_btn.click(
            fn=_confirm,
            inputs=[pool_state, preview_pid],
            outputs=[starting_id, selection_md, selection_summary, next2_btn,
                     stitch_btn, resample_btn],
        ).then(
            # Locking succeeded iff the Next button is now visible — in that
            # case jump straight to the Compose tab (client-side, see below).
            fn=None,
            js="""() => {
              const b = document.getElementById('next2-btn');
              if (b && b.offsetParent !== null) {
                document.querySelectorAll('button[role="tab"]')[2]?.click();
              }
            }""",
        )

        # ── Next-step navigation ──────────────────────────────────────────────
        # Outputting an update to a gr.Tabs layout crashes the Gradio 6
        # frontend, so tab switching is done client-side instead: each
        # button just clicks the corresponding tab header in the DOM.
        next1_btn.click(
            fn=None,
            js="() => { document.querySelectorAll('button[role=\"tab\"]')[1]?.click(); }",
        )
        next2_btn.click(
            fn=None,
            js="() => { document.querySelectorAll('button[role=\"tab\"]')[2]?.click(); }",
        )

        # ── Stitch ────────────────────────────────────────────────────────────
        def _stitch(pool, sid, structure, progress=gr.Progress()):
            _blank = "<em>—</em>"
            if pool is None or not pool.get("scores"):
                msg = "Generate phrases first (Generate tab)."
                return (msg, _blank, _blank, _blank, _blank, _blank, _blank, gr.update(visible=False))
            if sid is None:
                msg = "Select an opening phrase first (Browse & Select tab)."
                return (msg, _blank, _blank, _blank, _blank, _blank, _blank, gr.update(visible=False))

            chosen_structure = structure
            if chosen_structure == "Random":
                chosen_structure = random.choice(backend.STRUCTURE_NAMES)

            try:
                progress(0.1, desc="Sampling sections…")
                final, ids = backend.run_stitch(
                    chosen_structure, pool, fixed_beg=int(sid),
                )
            except ValueError as e:
                return (f"Error: {e}", _blank, _blank, _blank, _blank, _blank, _blank, gr.update(visible=False))

            try:
                final = backend.format_satb(final)
            except Exception:
                pass  # fall back to the raw layout rather than failing the run

            beg_id, mid_id, mid2_id, end_id = ids
            progress(0.5, desc="Rendering sections…")
            open_h  = _section_card("Opening",  beg_id,  pool)
            dev_h   = _section_card("Phrase 2", mid_id,  pool)
            exp_h   = _section_card("Phrase 3", mid2_id, pool)
            close_h = _section_card("Ending",   end_id,  pool)

            progress(0.75, desc="Rendering score…")
            try:
                score_html = backend.sheet_music_html(final, height="480px")
            except Exception as e:
                score_html = f"<em>Score unavailable: {e}</em>"

            progress(0.9, desc="Rendering audio…")
            try:
                full_bytes = backend.score_to_midi_bytes(final)
                full_html  = backend.midi_player_html(full_bytes, height="220px", show_viz=True)
            except Exception as e:
                full_bytes = None
                full_html  = f"<em>Audio unavailable: {e}</em>"

            file_update = gr.update(visible=False)
            if full_bytes is not None:
                tf = tempfile.NamedTemporaryFile(
                    suffix=".mid", delete=False, prefix="progress_composition_"
                )
                tf.write(full_bytes); tf.close()
                file_update = gr.update(value=tf.name, visible=True)

            status = (
                f"Composed with *{chosen_structure}* &nbsp;·&nbsp; "
                f"#{beg_id} → #{mid_id} → #{mid2_id} → #{end_id}"
            )
            progress(1.0)
            return status, open_h, dev_h, exp_h, close_h, score_html, full_html, file_update

        stitch_outputs = [
            stitch_status,
            opening_html, develop_html, expand_html, closing_html,
            full_score, full_player, download_btn,
        ]
        stitch_btn.click(
            fn=_stitch,
            inputs=[pool_state, starting_id, structure_dd],
            outputs=stitch_outputs,
        )
        resample_btn.click(
            fn=_stitch,
            inputs=[pool_state, starting_id, structure_dd],
            outputs=stitch_outputs,
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = create_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        inbrowser=True,
        css=CSS,
        # NOTE: keep this exactly as the un-customized Soft(primary_hue="blue").
        # This Gradio build crashes (`'str' object has no attribute 'name'`) when
        # a .set()/neutral_hue-customized theme is compared against the built-in
        # themes' font lists, so all surface/colour styling lives in CSS instead.
        theme=gr.themes.Soft(primary_hue="blue"),
    )
