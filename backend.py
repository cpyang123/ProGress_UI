"""
backend.py – API layer for the ProGress UI.

Wraps:
  • ProGress_Supplement  –  phrase loading, rejection sampling, scoring, stitching
  • SchenkerDiff         –  (optional) diffusion-model inference for new phrase generation
"""

from __future__ import annotations

import base64
import copy
import json
import os
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

# ── Path setup ────────────────────────────────────────────────────────────────
# This app is self-contained: its cross-repo dependencies (phrase_stitching/ and
# SchenkerDiff/) are vendored under ./vendor so it can ship as a single package.
# When run in-place inside the original research tree (no ./vendor present), it
# falls back to the sibling ProGress_Supplement/ and SchenkerDiff/ folders.
# Either root can be overridden with the PROGRESS_SUPPLEMENT_DIR /
# PROGRESS_SCHENKER_DIR environment variables.

PKG_DIR = Path(__file__).resolve().parent

if (PKG_DIR / "vendor" / "SchenkerDiff").exists():     # packaged / deployed layout
    SUPPLEMENT_DIR = PKG_DIR / "vendor"
    SCHENKER_DIR   = PKG_DIR / "vendor" / "SchenkerDiff"
else:                                                  # original research-tree layout
    BASE_DIR       = PKG_DIR.parent
    SUPPLEMENT_DIR = BASE_DIR / "ProGress_Supplement"
    SCHENKER_DIR   = BASE_DIR / "SchenkerDiff"

SUPPLEMENT_DIR  = Path(os.environ.get("PROGRESS_SUPPLEMENT_DIR", SUPPLEMENT_DIR))
SCHENKER_DIR    = Path(os.environ.get("PROGRESS_SCHENKER_DIR", SCHENKER_DIR))
OUTPUT_VIS_DIR  = SCHENKER_DIR / "output_vis"
DIFFUSION_OUT   = SUPPLEMENT_DIR / "phrase_stitching" / "diffusion_output"
CACHE_FILE      = Path(__file__).parent / ".phrase_cache.json"
CHECKPOINT_PATH = SCHENKER_DIR / "last-v1.ckpt"

for _p in [str(SUPPLEMENT_DIR), str(SCHENKER_DIR), str(OUTPUT_VIS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── ProGress_Supplement imports ───────────────────────────────────────────────

from phrase_stitching.RN_analysis import (
    analyze_entire_phrase,
    check_bad_counterpoint,
    check_bad_mode_mixture,
    check_illegal_harmonics_on_integer_beats,
    InvalidAnalysisException,
    MODES_ROMAN_NUMERALS,
)
from phrase_stitching.stitch import (
    combine_two_scores,
    extend_last_note_to_fill_measure,
    POSSIBLE_STARTS_ENDING_FROM_TONIC,
    transpose_score,
)
from phrase_stitching.write_inner_voices import write_inner_voices
from phrase_stitching.config import (
    SCORE_INCLUDES_III,
    SCORE_INCLUDES_v,
    SCORE_MAJOR_AND_MINOR,
)

# ── Structure catalogue ───────────────────────────────────────────────────────

STRUCTURE_NAMES: list[str] = [
    "I – V – I  (Major)",
    "I – IV – V – I  (Major)",
    "i – III – iv – i  (Minor)",
    "i – III – V – i  (Minor)",
    "i – VI – iv – i  (Minor)",
]


# ─────────────────────────────────────────────────────────────────────────────
# Phrase loading & rejection sampling
# ─────────────────────────────────────────────────────────────────────────────

def _quality_score(analysis: list[str]) -> float:
    """Compute a simple quality score from config weights."""
    score = 0.0
    rn_set = set(analysis)
    if "III" in rn_set or "iii" in rn_set:
        score += SCORE_INCLUDES_III
    if "v" in rn_set:
        score += SCORE_INCLUDES_v
    major_core = MODES_ROMAN_NUMERALS["major"] - {"V", "viio"}
    minor_core = MODES_ROMAN_NUMERALS["minor"] - {"V", "viio"}
    if rn_set & major_core and rn_set & minor_core:
        score += SCORE_MAJOR_AND_MINOR
    return round(score, 3)


def _detect_mode(analysis: list[str]) -> str:
    rn_set = set(analysis)
    major_core = MODES_ROMAN_NUMERALS["major"] - {"V", "viio"}
    minor_core = MODES_ROMAN_NUMERALS["minor"] - {"V", "viio"}
    has_major = bool(rn_set & major_core)
    has_minor = bool(rn_set & minor_core)
    if has_major and not has_minor:
        return "major"
    if has_minor and not has_major:
        return "minor"
    return "mixed"


def load_phrases(use_cache: bool = True, progress=None) -> dict[str, Any]:
    """
    Load and rejection-sample all phrases from the diffusion_output folder.

    Returns a phrases_data dict:
      scores    – list[music21.Score]
      analyses  – list[list[str]]
      info      – list[dict]  (metadata per phrase)
      starts    – dict[str, list[int]]  harmony → [phrase_id]
      ends      – dict[str, list[int]]  harmony → [phrase_id]
      stats     – dict  (loaded / rejected / total)
    """
    cache_valid_sources: list[dict] | None = None

    if use_cache and CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                cache = json.load(f)
            cache_valid_sources = cache.get("valid", None)
        except Exception:
            cache_valid_sources = None

    scores:   list[Any]        = []
    analyses: list[list[str]]  = []
    info:     list[dict]       = []
    starts:   dict             = defaultdict(list)
    ends:     dict             = defaultdict(list)
    n_loaded  = 0
    n_rejected = 0

    # ── Cache hit: re-parse XMLs (fast) but skip re-analysis ─────────────────
    if cache_valid_sources is not None:
        total = len(cache_valid_sources)
        for idx, entry in enumerate(cache_valid_sources):
            if progress:
                progress(idx / max(total, 1), desc=f"Loading from cache ({idx}/{total})…")
            xml_path = Path(entry["source"])
            if not xml_path.exists():
                continue
            try:
                import music21.converter as _conv
                score = _conv.parse(str(xml_path))
            except Exception:
                continue

            analysis = entry["analysis"]
            phrase_id = len(scores)
            scores.append(score)
            analyses.append(analysis)

            meta = {
                "id":       phrase_id,
                "start_rn": entry["start_rn"],
                "end_rn":   entry["end_rn"],
                "mode":     entry["mode"],
                "quality":  entry["quality"],
                "source":   entry["source"],
            }
            info.append(meta)
            starts[entry["start_rn"]].append(phrase_id)
            ends[entry["end_rn"]].append(phrase_id)
            n_loaded += 1

        stats = {"loaded": n_loaded, "rejected": 0, "total": n_loaded, "from_cache": True}
        return dict(scores=scores, analyses=analyses, info=info,
                    starts=starts, ends=ends, stats=stats)

    # ── Full load: parse + analyse + reject ───────────────────────────────────
    folders    = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]
    total_files = len(folders) * 100
    cache_entries: list[dict] = []

    for j in folders:
        for i in range(1, 101):
            xml_path = DIFFUSION_OUT / f"output_graphs_{j}" / f"output_graph_{i}.xml"
            done = (j - 1) * 100 + i
            if progress:
                progress(done / total_files,
                         desc=f"Analysing phrase {j}/{i} ({n_loaded} valid, {n_rejected} rejected)…")

            if not xml_path.exists():
                n_rejected += 1
                continue
            try:
                score, analysis = analyze_entire_phrase(str(xml_path))
                check_illegal_harmonics_on_integer_beats(score)
                check_bad_mode_mixture(score)
                check_bad_counterpoint(score)
            except (InvalidAnalysisException, FileNotFoundError, Exception):
                n_rejected += 1
                continue

            start_rn = analysis[0]
            end_rn   = analysis[-1]
            quality  = _quality_score(analysis)
            mode     = _detect_mode(analysis)
            phrase_id = len(scores)

            scores.append(score)
            analyses.append(analysis)
            meta = dict(id=phrase_id, start_rn=start_rn, end_rn=end_rn,
                        mode=mode, quality=quality, source=str(xml_path))
            info.append(meta)
            starts[start_rn].append(phrase_id)
            ends[end_rn].append(phrase_id)

            cache_entries.append(dict(
                source=str(xml_path), analysis=analysis,
                start_rn=start_rn, end_rn=end_rn, mode=mode, quality=quality,
            ))
            n_loaded += 1

    # Persist cache
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump({"valid": cache_entries}, f)
    except Exception:
        pass

    stats = {"loaded": n_loaded, "rejected": n_rejected,
             "total": n_loaded + n_rejected, "from_cache": False}
    return dict(scores=scores, analyses=analyses, info=info,
                starts=starts, ends=ends, stats=stats)


# ─────────────────────────────────────────────────────────────────────────────
# Phrase table helpers
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd

ALL_HARMONIES = sorted({
    "I", "i", "ii", "iio", "iii", "III", "IV", "iv",
    "V", "v", "vi", "VI", "viio", "VII",
})


def build_phrase_df(info: list[dict],
                    mode_filter: str = "all",
                    start_filter: str = "any",
                    end_filter: str = "any",
                    selected_ids: set[int] | None = None) -> pd.DataFrame:
    rows = []
    for p in info:
        if mode_filter != "all" and p["mode"] != mode_filter:
            continue
        if start_filter != "any" and p["start_rn"] != start_filter:
            continue
        if end_filter != "any" and p["end_rn"] != end_filter:
            continue
        rows.append({
            "ID":      p["id"],
            "Start":   p["start_rn"],
            "End":     p["end_rn"],
            "Mode":    p["mode"],
            "Quality": p["quality"],
            "Fav":     "♥" if (selected_ids and p["id"] in selected_ids) else "",
        })
    return pd.DataFrame(rows, columns=["ID", "Start", "End", "Mode", "Quality", "Fav"])


# ─────────────────────────────────────────────────────────────────────────────
# MIDI / audio helpers
# ─────────────────────────────────────────────────────────────────────────────

# HTML to inject into the document <head> via gr.HTML(head=...).
# Loads html-midi-player + its dependencies once for the whole page.
MIDI_PLAYER_HEAD = (
    '<script src="https://cdn.jsdelivr.net/combine/'
    'npm/tone@14.7.58,'
    'npm/@magenta/music@1.23.1/es6/core.js,'
    'npm/focus-visible@5,'
    'npm/html-midi-player@1.5.0"></script>'
)

OSMD_HEAD = (
    '<script src="https://cdn.jsdelivr.net/npm/'
    'opensheetmusicdisplay@1.8.8/build/opensheetmusicdisplay.min.js"></script>'
)


def _rebuild_sites(score):
    """Deep-copy a score to rebuild music21's element-site bookkeeping.

    gradio's gr.State pickles the pooled scores on HF Spaces; unpickling leaves
    each element's `activeSite` missing from its `siteDict`, so music21 export
    raises KeyError(<id>) deep in expandRepeats()/sortTuple().  A fresh deep copy
    reconstructs consistent sites.  Used as an on-failure retry, so clean scores
    (e.g. local, in-memory) pay no extra cost.
    """
    import copy
    return copy.deepcopy(score)


def score_to_midi_bytes(score) -> bytes:
    """Convert a music21 Score to raw MIDI bytes."""
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        tmp = f.name
    try:
        try:
            score.write("midi", fp=tmp)
        except Exception:
            _rebuild_sites(score).write("midi", fp=tmp)
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def midi_player_html(midi_bytes: bytes, height: str = "140px", show_viz: bool = True) -> str:
    """Return an HTML snippet with an embedded MIDI player (html-midi-player).

    Layout notes:
      * Gradio 6 does NOT execute <script> tags inside gr.HTML values, so the
        library must come from gr.HTML(head=…) on the page header; the custom
        elements then upgrade automatically when this HTML is inserted.
      * The visualizer gets its own src so it draws the piano roll immediately,
        rather than waiting for the player to push notes to it.
      * Each element gets explicit dimensions + display:block so they reserve
        their box up-front (avoids the "everything piles on top" pre-upgrade
        flash).
    """
    b64 = base64.b64encode(midi_bytes).decode()
    data_uri = f"data:audio/midi;base64,{b64}"
    pid = random.randint(100_000, 999_999)

    viz_block = ""
    viz_attr = ""
    if show_viz:
        # height:auto + min-height lets the roll grow to fit both voices;
        # the flex wrapper centers it because short phrases render narrow.
        viz_block = (
            f'<div style="display:flex;justify-content:center;width:100%;'
            f'overflow-x:auto;">'
            f'<midi-visualizer type="piano-roll" id="viz{pid}" src="{data_uri}" '
            f'style="display:block;height:auto;min-height:{height};'
            f'max-width:100%;background:#fff;border-radius:6px;'
            f'border:1px solid #e2e8f0;'
            f'box-sizing:border-box;overflow:auto;"></midi-visualizer>'
            f'</div>'
        )
        viz_attr = f'visualizer="#viz{pid}"'

    player_block = (
        f'<midi-player src="{data_uri}" sound-font {viz_attr} '
        f'style="display:block;width:100%;min-height:80px;'
        f'box-sizing:border-box;'
        f'--player-background-color:#f8fafc;'
        f'--player-button-color:#1d4ed8;"></midi-player>'
    )

    container = (
        '<div style="display:flex;flex-direction:column;gap:8px;width:100%;'
        'margin:4px 0;">'
        f"{viz_block}{player_block}"
        "</div>"
    )

    return container


def score_to_xml_b64(score) -> str:
    """Convert a music21 Score to a base64-encoded MusicXML string."""
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        tmp = f.name
    try:
        try:
            score.write("musicxml", fp=tmp)
        except Exception:
            _rebuild_sites(score).write("musicxml", fp=tmp)
        with open(tmp, "rb") as f:
            return base64.b64encode(f.read()).decode()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def sheet_music_html(score, height: str = "360px") -> str:
    """Return an HTML snippet that renders a music21 Score as sheet music via OSMD.

    Gradio 6 does NOT execute <script> tags inside gr.HTML values, so this
    emits only a marker <div> carrying the base64 MusicXML in a data attribute.
    The actual rendering is done by OSMD_JS_ON_LOAD, which must be passed as
    js_on_load= to every gr.HTML component that displays scores.
    """
    b64 = score_to_xml_b64(score)
    return (
        f'<div class="osmd-target" data-osmd="{b64}" '
        f'style="width:100%;max-width:900px;margin:0 auto;'
        f'min-height:{height};background:#fff;'
        f'border-radius:6px;padding:8px;overflow-x:auto;'
        f'border:1px solid #e2e8f0;box-sizing:border-box;">'
        f'<em style="color:#94a3b8">Rendering score…</em></div>'
    )


# js_on_load handler for gr.HTML components that display OSMD scores.
# `element` is provided by Gradio; it is the component's root DOM element.
# Re-renders whenever the component's HTML value changes (MutationObserver).
OSMD_JS_ON_LOAD = """
function ensureOsmd(cb) {
  if (window.opensheetmusicdisplay) { cb(); return; }
  if (!window.__osmdLoading) {
    window.__osmdLoading = true;
    var s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@1.8.8/build/opensheetmusicdisplay.min.js';
    document.head.appendChild(s);
  }
  var t = setInterval(function() {
    if (window.opensheetmusicdisplay) { clearInterval(t); cb(); }
  }, 150);
}
function renderAll() {
  element.querySelectorAll('[data-osmd]').forEach(function(el) {
    var b64 = el.getAttribute('data-osmd');
    if (!b64 || el.getAttribute('data-osmd-done') === b64) return;
    el.setAttribute('data-osmd-done', b64);
    ensureOsmd(function() {
      try {
        var bin = atob(b64);
        var bytes = new Uint8Array(bin.length);
        for (var j = 0; j < bin.length; j++) bytes[j] = bin.charCodeAt(j);
        var xml = new TextDecoder('utf-8').decode(bytes);
        el.innerHTML = '';
        var osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(el, {
          autoResize: true, backend: 'svg', drawTitle: false,
          drawComposer: false, drawLyricist: false,
          drawPartNames: false, drawPartAbbreviations: false,
        });
        osmd.load(xml).then(function() { osmd.render(); });
      } catch (e) {
        el.innerHTML = "<em style='color:#e11d48'>Score render error: " + e + "</em>";
      }
    });
  });
}
renderAll();
new MutationObserver(renderAll).observe(element, { childList: true, subtree: true });
"""


# js_on_load handler for gr.HTML components containing a <midi-visualizer>.
# Two jobs:
#   1. Bump the piano-roll zoom (default rendering is far too small to read).
#   2. Whenever the src changes (user previews a different phrase), swap the
#      visualizer for a freshly created element.  html-midi-player's
#      VisualizerElement calls attachShadow unguarded in connectedCallback, so
#      a reused/reattached node breaks and keeps showing the old piano roll;
#      a brand-new node always initializes cleanly.
MIDI_VIZ_JS_ON_LOAD = """
function refreshViz() {
  element.querySelectorAll('midi-visualizer').forEach(function(v) {
    var src = v.getAttribute('src') || '';
    if (!src || v.__vizSrc === src) return;
    var fresh = document.createElement('midi-visualizer');
    for (var i = 0; i < v.attributes.length; i++) {
      fresh.setAttribute(v.attributes[i].name, v.attributes[i].value);
    }
    fresh.__vizSrc = src;
    v.replaceWith(fresh);
    customElements.whenDefined('midi-visualizer').then(function() {
      fresh.config = { noteHeight: 8, pixelsPerTimeStep: 60 };
      var id = fresh.getAttribute('id');
      if (!id) return;
      var p = element.querySelector('midi-player[visualizer="#' + id + '"]');
      if (p) customElements.whenDefined('midi-player').then(function() {
        p.setAttribute('visualizer', '#' + id);
      });
    });
  });
}
refreshViz();
new MutationObserver(refreshViz).observe(element,
  { childList: true, subtree: true, attributes: true, attributeFilter: ['src'] });
"""


# ─────────────────────────────────────────────────────────────────────────────
# Phrase stitching
# ─────────────────────────────────────────────────────────────────────────────

def _candidates(starts: dict, ends: dict, via_key: str, end_key: str) -> list[int]:
    """
    Collect phrase IDs whose start harmony is compatible with the given
    key-change pivot and whose end harmony matches end_key.
    """
    possible_starts = POSSIBLE_STARTS_ENDING_FROM_TONIC.get(via_key, [])
    end_set = set(ends.get(end_key, []))
    return [pid for k in possible_starts for pid in starts.get(k, []) if pid in end_set]


def _sample_three_distinct(
    pools: list[list[int]],
    preferred: set[int] | None,
    exclude: set[int] | None = None,
    max_tries: int = 3000,
) -> list[int]:
    """Draw one ID from each of three pools; all distinct and not in `exclude`."""
    exclude = exclude or set()
    def draw(pool: list[int]) -> int:
        if preferred:
            pref_pool = [x for x in pool if x in preferred and x not in exclude]
            if pref_pool:
                return random.choice(pref_pool)
        any_pool = [x for x in pool if x not in exclude]
        if not any_pool:
            return random.choice(pool)
        return random.choice(any_pool)

    for _ in range(max_tries):
        picks = [draw(p) for p in pools]
        if len(set(picks)) == 3 and not (set(picks) & exclude):
            return picks
    raise ValueError(
        "Cannot find 3 distinct phrases for middle/middle-2/end (distinct from "
        "the locked beginning).  Generate more phrases or pick a different starting one."
    )


def _sample_four_distinct(
    pools: list[list[int]],
    preferred: set[int] | None,
    max_tries: int = 3000,
) -> list[int]:
    """
    Draw one ID from each of the four pools so all four are distinct,
    preferring IDs that appear in `preferred` where possible.
    """
    def draw(pool: list[int]) -> int:
        if preferred:
            pref_pool = [x for x in pool if x in preferred]
            if pref_pool:
                return random.choice(pref_pool)
        return random.choice(pool)

    for _ in range(max_tries):
        picks = [draw(p) for p in pools]
        if len(set(picks)) == 4:
            return picks

    raise ValueError(
        "Cannot find 4 distinct phrases for all sections.  "
        "Try loading more phrases or picking a different structure."
    )


def _realize(scores: list, analyses: list, pid: int, semitones: int = 0):
    """Deep-copy phrase `pid`, extend, fill inner voices, optionally transpose."""
    s = copy.deepcopy(scores[pid])
    a = list(analyses[pid])
    s, a = extend_last_note_to_fill_measure(s, a)
    write_inner_voices(s, a)
    if semitones:
        s = transpose_score(s, semitones)
    return s


def _require(pool: list[int], label: str) -> None:
    if not pool:
        raise ValueError(
            f"No valid phrases found for section '{label}'.  "
            "Make sure all phrases have been loaded."
        )


def format_satb(score):
    """
    Re-lay-out a stitched 4-part score in SATB order with register-appropriate
    clefs.

    The stitched score's parts arrive as (melody, bass, inner1, inner2), with
    the inner voices notated in treble clef regardless of register, and each
    section reprinting its own clef/meter/tempo.  This sorts the inner voices
    into alto/tenor by average pitch, orders parts S-A-T-B, gives each voice a
    clef that fits its register, and keeps only the opening meter/tempo marks.
    """
    from music21 import clef as m21clef
    from music21 import meter as m21meter
    from music21 import tempo as m21tempo
    from music21.stream import Score as M21Score

    parts = list(score.parts)
    if len(parts) < 4:
        return score

    def avg_midi(p):
        pitches = [n.pitch.midi for n in p.recurse().notes if hasattr(n, "pitch")]
        return sum(pitches) / len(pitches) if pitches else 60.0

    soprano, bass = parts[0], parts[1]
    inners = sorted(parts[2:4], key=avg_midi, reverse=True)  # higher = alto
    ordered = [soprano, inners[0], inners[1], bass]

    for p in ordered:
        # One clef per part, chosen by register; drop all section reprints.
        for old in list(p.recurse().getElementsByClass(m21clef.Clef)):
            old.activeSite.remove(old)
        a = avg_midi(p)
        if a >= 60:
            new_clef = m21clef.TrebleClef()
        elif a >= 50:
            new_clef = m21clef.Treble8vbClef()
        else:
            new_clef = m21clef.BassClef()
        target = p.measure(1) if p.hasMeasures() else p
        (target if target is not None else p).insert(0, new_clef)

        # Keep only the first time signature / tempo mark.
        for cls in (m21meter.TimeSignature, m21tempo.MetronomeMark):
            seen = False
            for el in list(p.recurse().getElementsByClass(cls)):
                if seen:
                    el.activeSite.remove(el)
                seen = True

    out = M21Score()
    for p in ordered:
        out.append(p)
    return out


# Each stitch function returns (final_score, [beg_id, mid_id, mid2_id, end_id]).

def _stitch_generic(
    starts: dict, ends: dict, scores: list, analyses: list,
    beg_key: str,
    mid_via: str, mid_end: str, mid_semitones: int,
    mid2_via: str, mid2_end: str, mid2_semitones: int,
    end_via: str, end_end: str,
    preferred: set[int] | None,
    fixed_beg: int | None = None,
) -> tuple:
    mid_pool  = _candidates(starts, ends, mid_via,  mid_end)
    mid2_pool = _candidates(starts, ends, mid2_via, mid2_end)
    end_pool  = _candidates(starts, ends, end_via,  end_end)

    _require(mid_pool,  f"middle ({mid_via}→{mid_end})")
    _require(mid2_pool, f"middle 2 ({mid2_via}→{mid2_end})")
    _require(end_pool,  f"end ({end_via}→{end_end})")

    if fixed_beg is not None:
        b = fixed_beg
        # Sample the other three so all four IDs are distinct.
        m, m2, e = _sample_three_distinct(
            [mid_pool, mid2_pool, end_pool], preferred, exclude={b},
        )
    else:
        beg_pool = ends.get(beg_key, [])
        _require(beg_pool, f"beginning ({beg_key})")
        b, m, m2, e = _sample_four_distinct(
            [beg_pool, mid_pool, mid2_pool, end_pool], preferred,
        )

    parts = [
        _realize(scores, analyses, b),
        _realize(scores, analyses, m,  mid_semitones),
        _realize(scores, analyses, m2, mid2_semitones),
        _realize(scores, analyses, e),
    ]
    final = parts[0]
    for p in parts[1:]:
        final = combine_two_scores(final, p)
    return final, [b, m, m2, e]


def stitch_I_V_I(starts, ends, scores, analyses, preferred=None, fixed_beg=None):
    return _stitch_generic(
        starts, ends, scores, analyses,
        beg_key="I",
        mid_via="V",   mid_end="I",  mid_semitones=-5,
        mid2_via="I",  mid2_end="I", mid2_semitones=-5,
        end_via="IV",  end_end="I",
        preferred=preferred, fixed_beg=fixed_beg,
    )


def stitch_I_IV_V_I(starts, ends, scores, analyses, preferred=None, fixed_beg=None):
    return _stitch_generic(
        starts, ends, scores, analyses,
        beg_key="I",
        mid_via="IV",  mid_end="I",  mid_semitones=5,
        mid2_via="ii", mid2_end="i", mid2_semitones=7,
        end_via="IV",  end_end="I",
        preferred=preferred, fixed_beg=fixed_beg,
    )


def stitch_i_III_iv_i(starts, ends, scores, analyses, preferred=None, fixed_beg=None):
    return _stitch_generic(
        starts, ends, scores, analyses,
        beg_key="i",
        mid_via="III", mid_end="I",  mid_semitones=3,
        mid2_via="ii", mid2_end="i", mid2_semitones=5,
        end_via="V",   end_end="i",
        preferred=preferred, fixed_beg=fixed_beg,
    )


def stitch_i_III_V_i(starts, ends, scores, analyses, preferred=None, fixed_beg=None):
    return _stitch_generic(
        starts, ends, scores, analyses,
        beg_key="i",
        mid_via="III",  mid_end="I",  mid_semitones=3,
        mid2_via="iii", mid2_end="i", mid2_semitones=7,
        end_via="IV",   end_end="i",
        preferred=preferred, fixed_beg=fixed_beg,
    )


def stitch_i_VI_iv_i(starts, ends, scores, analyses, preferred=None, fixed_beg=None):
    return _stitch_generic(
        starts, ends, scores, analyses,
        beg_key="i",
        mid_via="VI",  mid_end="I",  mid_semitones=-4,
        mid2_via="vi", mid2_end="i", mid2_semitones=-7,
        end_via="V",   end_end="i",
        preferred=preferred, fixed_beg=fixed_beg,
    )


_STITCH_FNS = {
    "I – V – I  (Major)":          stitch_I_V_I,
    "I – IV – V – I  (Major)":     stitch_I_IV_V_I,
    "i – III – iv – i  (Minor)":   stitch_i_III_iv_i,
    "i – III – V – i  (Minor)":    stitch_i_III_V_i,
    "i – VI – iv – i  (Minor)":    stitch_i_VI_iv_i,
}


def run_stitch(
    structure: str,
    phrases_data: dict,
    selected_ids: set[int] | None = None,
    fixed_beg: int | None = None,
) -> tuple:
    """
    Run phrase stitching for the chosen structure.

    fixed_beg  – if set, lock the beginning section to this phrase ID instead of
                 sampling.  Used when the user has picked a specific I-starting
                 phrase that should anchor the piece.
    Returns (final_score, [beg_id, mid_id, mid2_id, end_id]).
    """
    fn = _STITCH_FNS[structure]
    # Prefer high-quality phrases (non-zero quality = have III/v/mode mixture).
    high_quality = {p["id"] for p in phrases_data["info"] if p["quality"] > 0}
    preferred = (set(selected_ids) if selected_ids else set()) | high_quality
    return fn(
        phrases_data["starts"],
        phrases_data["ends"],
        phrases_data["scores"],
        phrases_data["analyses"],
        preferred=preferred or None,
        fixed_beg=fixed_beg,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SchenkerDiff generation  (optional – requires checkpoint + GPU deps)
# ─────────────────────────────────────────────────────────────────────────────

def checkpoint_available() -> bool:
    return CHECKPOINT_PATH.exists()


def device_info() -> str:
    """Human-readable compute device for display in the UI.

    Reports 'GPU · <name>' when CUDA is available, else 'CPU'.  Querying
    cuda.is_available()/get_device_name does not initialise a CUDA context,
    so this is cheap to call at startup.
    """
    # On a ZeroGPU Space the GPU only exists inside @spaces.GPU calls; touching
    # torch.cuda in the main process triggers a forbidden low-level CUDA init.
    # Detect ZeroGPU via the `spaces` package FIRST and avoid torch.cuda there.
    try:
        import spaces  # noqa: F401
        return "ZeroGPU (attached on demand)"
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return f"GPU · {torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    return "CPU"


# ─── Module-level cache for SchenkerDiff inference ───────────────────────────
# Loading the checkpoint is slow (~5 s).  We keep the model + helpers in memory
# so successive batches in generate_until_target reuse them.
_GEN_CACHE: dict = {}

# When True, generation is pinned to CPU regardless of what torch.cuda reports.
# Needed for the ZeroGPU CPU-fallback path: ZeroGPU's emulated torch.cuda says
# a GPU is available even in the main process, so acting on it there would
# trigger a forbidden low-level CUDA init.  The fallback sets this before
# retrying on CPU.
_FORCE_CPU: bool = False


def _available_processed_idxs() -> list[int]:
    """Indices of processed .pt files actually on disk (for varied conditioning)."""
    pdir = SCHENKER_DIR / "data/schenker/processed/heterdatacleaned/processed"
    idxs = []
    if not pdir.exists():
        return idxs
    for f in pdir.glob("*_processed.pt"):
        try:
            idxs.append(int(f.stem.split("_")[0]))
        except ValueError:
            continue
    return sorted(idxs)


def _ensure_generation_setup(progress=None) -> dict:
    """Load (or return cached) SchenkerDiff model + helpers.  Idempotent."""
    if _GEN_CACHE.get("ready"):
        return _GEN_CACHE

    if not checkpoint_available():
        raise RuntimeError(
            f"SchenkerDiff checkpoint not found at {CHECKPOINT_PATH}.\n"
            "Place last-v1.ckpt in the SchenkerDiff/ folder."
        )

    if progress:
        progress(0.0, desc="Initialising SchenkerDiff (first-time setup)…")

    # ── Workaround stubs for SchenkerDiff inference ─────────────────────────
    # 1. graph_tool is only used by training-time evaluation metrics that we
    #    never invoke at inference, but its C++ binary may conflict with the
    #    env's libgomp.  Stub it.
    import types as _types
    if "graph_tool" not in sys.modules:
        _gt = _types.ModuleType("graph_tool")
        _gt.all = _types.ModuleType("graph_tool.all")
        sys.modules["graph_tool"] = _gt
        sys.modules["graph_tool.all"] = _gt.all

    # 2. PlanarSamplingMetrics is pickled into the checkpoint's module tree,
    #    so the stub must subclass nn.Module for unpickling to walk it.
    if "src.analysis.spectre_utils" not in sys.modules:
        import torch.nn as _nn
        _spec_stub = _types.ModuleType("src.analysis.spectre_utils")
        class _NoOpMetrics(_nn.Module):
            def __init__(self, *a, **kw):
                super().__init__()
            def reset(self): pass
            def forward(self, *a, **kw): return {}
        _spec_stub.PlanarSamplingMetrics = _NoOpMetrics
        sys.modules["src.analysis.spectre_utils"] = _spec_stub

    old_dir = os.getcwd()
    os.chdir(SCHENKER_DIR)
    import torch
    import torch.nn.functional as F

    # 3. torch.load fixes, applied on both CPU and GPU:
    #    (a) torch>=2.6 flipped the weights_only default to True, which rejects
    #        the Lightning checkpoint's pickled config objects (omegaconf
    #        DictConfig, etc.).  We trust our own checkpoint, so force False.
    #    (b) When no GPU is present, PL passes map_location=None, so default the
    #        load to CPU.
    _orig_torch_load = torch.load
    # _FORCE_CPU wins over the (possibly emulated) cuda.is_available() so the
    # CPU-fallback path never initialises CUDA in the ZeroGPU main process.
    use_cuda = (not _FORCE_CPU) and torch.cuda.is_available()
    _no_cuda = not use_cuda
    _cpu_dev = torch.device("cpu")
    def _patched_load(f, *a, **kw):
        kw["weights_only"] = False
        if _no_cuda and kw.get("map_location") is None:
            kw["map_location"] = _cpu_dev
        return _orig_torch_load(f, *a, **kw)
    torch.load = _patched_load

    try:
        from inference import initialize_model
        from src.diffusion import diffusion_utils
        from src.datasets.schenker_dataset import SchenkerDiffHeteroGraphData
        from realization import realization  # output_vis/realization.py

        if progress:
            progress(0.5, desc="Loading checkpoint…")

        model = initialize_model()
        # Pin the model + all generation tensors to one device chosen *now*.
        # Under ZeroGPU, CUDA only becomes available inside the @spaces.GPU call
        # (after import time), so we can't rely on the import-time config.DEVICE.
        dev = torch.device("cuda") if use_cuda else _cpu_dev
        model = model.to(dev)
        edim  = int(model.limit_dist.E.shape[0])

        _GEN_CACHE.update(dict(
            ready=True,
            model=model,
            edim=edim,
            DEVICE=dev,
            torch=torch,
            F=F,
            diffusion_utils=diffusion_utils,
            HeteroData=SchenkerDiffHeteroGraphData,
            realization=realization,
            schenker_dir=str(SCHENKER_DIR),
            available_idxs=_available_processed_idxs(),
        ))
        if progress:
            progress(1.0, desc=f"Model ready (Edim={edim}, device={dev}).")
        return _GEN_CACHE
    finally:
        # Intentionally leave torch.load patched (weights_only=False) for the rest
        # of the session: later loads of our own trusted .pt conditioning files
        # (see _local_sample_r_E) also need it under torch>=2.6.  All files loaded
        # by this app are produced by us, so disabling the weights-only guard is safe.
        os.chdir(old_dir)


def _local_sample_r_E(batch_size: int, edim: int, idxs: list[int]):
    """
    Replacement for inference.sample_r_E that

      • reads the actual edge_attr width from the model (Edim, e.g. 30),
      • samples random idx values from the available .pt files so each
        batch element gets *different* rhythm / edge conditioning.
    """
    cache = _GEN_CACHE
    torch_, F, HD = cache["torch"], cache["F"], cache["HeteroData"]

    E_list, r_list, name_list, node_sizes = [], [], [], []
    pool = idxs or [1]
    proc_dir = SCHENKER_DIR / "data" / "schenker" / "processed" / "heterdatacleaned" / "processed"
    for _ in range(batch_size):
        idx = random.choice(pool)
        fp = str(proc_dir / f"{idx}_processed.pt")
        data_dict = torch_.load(fp)
        data = HD.hetero_to_data(data_dict)

        m = data.x.shape[0]
        E_sample = torch_.zeros((m, m, edim))
        for i in range(data.edge_index.shape[1]):
            u = data.edge_index[0, i].item()
            v = data.edge_index[1, i].item()
            if u < m and v < m:
                E_sample[u, v, :] = data.edge_attr[i, :edim]

        dr = data.r.shape[1]
        r_sample = torch_.zeros((m, dr))
        r_sample[:m, :] = data.r[:m, :]

        E_list.append(E_sample)
        r_list.append(r_sample)
        name_list.append(data_dict["name"])
        node_sizes.append(m)

    max_nodes = max(t.shape[0] for t in r_list)
    E_pad = [F.pad(e, (0, 0, 0, max_nodes - e.shape[0], 0, max_nodes - e.shape[0])) for e in E_list]
    r_pad = [F.pad(r, (0, 0, 0, max_nodes - r.shape[0])) for r in r_list]
    return torch_.stack(E_pad, dim=0), torch_.stack(r_pad, dim=0), name_list, node_sizes


def _generate_one_batch(batch_size: int, progress=None, prog_lo=0.0, prog_hi=1.0) -> list[dict]:
    """Run one diffusion batch.  Returns list of phrase dicts that passed rejection."""
    cache = _ensure_generation_setup(progress=progress)
    torch_ = cache["torch"]
    model  = cache["model"]
    edim   = cache["edim"]
    DEVICE = cache["DEVICE"]
    diff_u = cache["diffusion_utils"]
    realization = cache["realization"]

    span = prog_hi - prog_lo
    def _p(frac, desc):
        if progress:
            progress(prog_lo + span * frac, desc=desc)

    _p(0.02, "Sampling conditioning data…")
    E, r, names, n_nodes_list = _local_sample_r_E(batch_size, edim, cache["available_idxs"])
    num_nodes = torch_.tensor([int(x) for x in n_nodes_list]).to(model.device)
    n_max = torch_.max(num_nodes).item()

    arange = torch_.arange(n_max, device=model.device).unsqueeze(0).expand(batch_size, -1)
    node_mask = arange < num_nodes.unsqueeze(1)
    z_T = diff_u.sample_discrete_feature_noise(limit_dist=model.limit_dist, node_mask=node_mask)
    X, _, y = z_T.X, z_T.E, z_T.y

    E_t = E.permute(0, 2, 1, 3)
    E   = torch_.maximum(E, E_t).to(DEVICE)
    r   = r.to(DEVICE)

    _p(0.05, "Running diffusion (100 steps)…")
    for s_int in reversed(range(0, model.T)):
        s_arr = s_int * torch_.ones((batch_size, 1)).type_as(y)
        t_arr = s_arr + 1
        sampled_s, _ = model.sample_p_zs_given_zt(s_arr / model.T, t_arr / model.T,
                                                  X, E, r, y, node_mask)
        X, _, y = sampled_s.X, sampled_s.E, sampled_s.y
        if s_int % 20 == 0:
            _p(0.05 + 0.85 * (1 - s_int / model.T), f"Diffusion step {model.T - s_int}/{model.T}")

    sampled_s = sampled_s.mask(node_mask, collapse=True)
    X, _, y = sampled_s.X, sampled_s.E, sampled_s.y
    E, _ = model.apply_node_mask_E_r(E, r, node_mask)

    _p(0.92, "Realising + rejection sampling…")
    tmp_dir = Path(tempfile.mkdtemp())
    new_phrases = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        X_i = X[i, :n].cpu().numpy()
        r_i = r[i, :n, :].cpu().numpy()
        out_xml = str(tmp_dir / f"gen_{i}.xml")
        try:
            realization(X_i, r_i, output_file=out_xml, num_voices=2)
            score, analysis = analyze_entire_phrase(out_xml)
            check_illegal_harmonics_on_integer_beats(score)
            check_bad_mode_mixture(score)
            check_bad_counterpoint(score)
            new_phrases.append({
                "score":    score,
                "analysis": analysis,
                "start_rn": analysis[0],
                "end_rn":   analysis[-1],
                "mode":     _detect_mode(analysis),
                "quality":  _quality_score(analysis),
                "source":   out_xml,
            })
        except (InvalidAnalysisException, Exception):
            continue

    _p(1.0, f"Batch done: {len(new_phrases)}/{batch_size} passed filters")
    return new_phrases


def empty_phrases_data() -> dict:
    """Fresh, empty phrases_data dict (same shape as load_phrases output)."""
    return dict(
        scores=[], analyses=[], info=[],
        starts=defaultdict(list), ends=defaultdict(list),
        stats=dict(loaded=0, rejected=0, total=0, from_cache=False),
    )


def _append_phrase(pool: dict, p: dict) -> None:
    """Append one generated phrase dict to a pool (mutates pool)."""
    pid = len(pool["scores"])
    pool["scores"].append(p["score"])
    pool["analyses"].append(p["analysis"])
    pool["info"].append({
        "id":       pid,
        "start_rn": p["start_rn"],
        "end_rn":   p["end_rn"],
        "mode":     p["mode"],
        "quality":  p["quality"],
        "source":   p["source"],
    })
    pool["starts"][p["start_rn"]].append(pid)
    pool["ends"][p["end_rn"]].append(pid)


def load_pregenerated(pool: dict | None = None, progress=None) -> dict:
    """
    Load the pre-generated phrases shipped in ProGress_Supplement
    (phrase_stitching/diffusion_output) and merge them into `pool`.

    Works without the SchenkerDiff checkpoint, so the demo can run on the
    bundled phrase set alone.  Returns the (new or mutated) phrases_data dict.
    """
    data = load_phrases(use_cache=True, progress=progress)
    if pool is None or not pool.get("scores"):
        return data
    existing_sources = {p.get("source") for p in pool["info"]}
    for i, score in enumerate(data["scores"]):
        info = data["info"][i]
        if info["source"] in existing_sources:
            continue
        _append_phrase(pool, {
            "score":    score,
            "analysis": data["analyses"][i],
            "start_rn": info["start_rn"],
            "end_rn":   info["end_rn"],
            "mode":     info["mode"],
            "quality":  info["quality"],
            "source":   info["source"],
        })
    pool["stats"]["loaded"] = len(pool["scores"])
    return pool


def _gpu_decorator(duration: int = 120):
    """@spaces.GPU on a ZeroGPU Space; a no-op decorator everywhere else.

    On ZeroGPU a GPU is attached only for the duration of the decorated call,
    which is why the model is loaded lazily *inside* generation.  Locally and on
    CPU/standard-GPU Spaces the `spaces` package is absent and this is a no-op.
    """
    try:
        import spaces
        return spaces.GPU(duration=duration)
    except Exception:
        return lambda fn: fn


def _run_generation(
    target: int,
    batch_size: int,
    max_attempts_factor: int,
    progress,
    pool: dict | None,
) -> dict:
    """Core generation loop.  Device is chosen in _ensure_generation_setup."""
    if pool is None:
        pool = empty_phrases_data()

    attempts = 0
    max_attempts = target * max_attempts_factor
    batch_idx = 0
    while len(pool["scores"]) < target and attempts < max_attempts:
        n_have = len(pool["scores"])
        if progress:
            progress(
                n_have / max(target, 1),
                desc=f"Batch {batch_idx + 1} – {n_have}/{target} valid so far (attempted {attempts})",
            )
        try:
            batch = _generate_one_batch(batch_size)
        except Exception as exc:
            # Let device/GPU failures propagate so the caller can fall back to
            # CPU; swallow only benign per-batch errors (e.g. a bad realisation).
            if any(s in str(exc).lower()
                   for s in ("cuda", "gpu", "device-side", "out of memory", "nvml", "zerogpu")):
                raise
            if progress:
                progress(n_have / max(target, 1),
                         desc=f"Batch {batch_idx + 1} failed: {exc}")
            break
        for p in batch:
            _append_phrase(pool, p)
        attempts += batch_size
        batch_idx += 1

    pool["stats"] = dict(
        loaded=len(pool["scores"]),
        rejected=max(attempts - len(pool["scores"]), 0),
        total=attempts,
        from_cache=False,
    )
    if progress:
        progress(
            1.0,
            desc=f"Done – {pool['stats']['loaded']} valid phrases from {attempts} attempts",
        )
    return pool


@_gpu_decorator(duration=120)
def _run_generation_gpu(target, batch_size, max_attempts_factor, progress, pool):
    return _run_generation(target, batch_size, max_attempts_factor, progress, pool)


def generate_until_target(
    target: int = 100,
    batch_size: int = 16,
    max_attempts_factor: int = 5,
    progress=None,
    pool: dict | None = None,
) -> dict:
    """
    Generate SchenkerDiff phrases in batches until `target` valid (post-rejection)
    phrases are in the pool, or `target * max_attempts_factor` samples have been
    attempted.

    Runs on GPU via ZeroGPU's @spaces.GPU when available; if the GPU can't be
    acquired or fails mid-run, transparently falls back to CPU.

    Returns the (mutated or new) phrases_data dict.
    """
    global _FORCE_CPU
    try:
        return _run_generation_gpu(target, batch_size, max_attempts_factor, progress, pool)
    except Exception as gpu_err:
        # GPU unavailable / failed → retry on CPU.  Pin to CPU first: under
        # ZeroGPU the main process must never touch CUDA (emulated is_available()
        # reports True), so _FORCE_CPU stops setup from initialising the GPU here.
        import traceback
        print("GPU generation failed; falling back to CPU:\n" + traceback.format_exc(),
              file=sys.stderr, flush=True)
        _GEN_CACHE.clear()
        _FORCE_CPU = True
        if progress:
            try:
                progress(0.0, desc=f"GPU unavailable ({type(gpu_err).__name__}); running on CPU…")
            except Exception:
                pass
        try:
            return _run_generation(target, batch_size, max_attempts_factor, progress, pool)
        finally:
            _FORCE_CPU = False


# Keep the old function name as a thin wrapper for back-compat.
def generate_new_phrases(batch_size: int = 4, progress=None) -> list[dict]:
    return _generate_one_batch(batch_size, progress=progress)


def append_generated_phrases(phrases_data: dict, new_phrases: list[dict]) -> dict:
    """Merge newly generated phrases into an existing phrases_data dict."""
    for entry in new_phrases:
        pid = len(phrases_data["scores"])
        phrases_data["scores"].append(entry["score"])
        phrases_data["analyses"].append(entry["analysis"])
        meta = {
            "id":       pid,
            "start_rn": entry["start_rn"],
            "end_rn":   entry["end_rn"],
            "mode":     entry["mode"],
            "quality":  entry["quality"],
            "source":   entry["source"],
        }
        phrases_data["info"].append(meta)
        phrases_data["starts"][entry["start_rn"]].append(pid)
        phrases_data["ends"][entry["end_rn"]].append(pid)
    return phrases_data
