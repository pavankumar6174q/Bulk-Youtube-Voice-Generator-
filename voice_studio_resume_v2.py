import os
import re
import threading
import json
import datetime
import sys
import time
import shutil
import hashlib
from pathlib import Path

import customtkinter as ctk
import torch
import torchaudio
import numpy as np
from tkinter import filedialog, messagebox

# ─── Audio playback ────────────────────────────────────────────
try:
    import pygame
    pygame.mixer.init()
    AUDIO_BACKEND = "pygame"
except ImportError:
    AUDIO_BACKEND = "os"
    print("WARNING: pygame not installed. Run: pip install pygame")
    print("Without pygame, playback will open Windows Media Player.")

# ─── Settings ──────────────────────────────────────────────────
DEFAULT_OUTPUT_FOLDER = r"E:\Yt Assests\chatterbox\new tts\outputs"
MAX_CHARS     = 110
PAUSE_MS      = 180
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "studio_settings.json")
# Every running job keeps its progress here, one sub-folder per job, so a
# pause / crash / power cut never loses more than the chunk being generated.
RESUME_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resume_sessions")
MAX_TAKES     = 6
BATCH_PAGE_SIZE = 40     # rows drawn at once (Tk cannot draw >~32767px of scroll area)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

BG_MAIN    = "#0D0D0F"
BG_PANEL   = "#141418"
BG_CARD    = "#1A1A22"
BG_INPUT   = "#111115"
ACCENT     = "#7C5CFC"
ACCENT2    = "#A855F7"
ACCENT_DIM = "#3D2E7C"
GREEN      = "#22C55E"
RED        = "#EF4444"
AMBER      = "#F59E0B"
TEAL       = "#14B8A6"
TEXT_1     = "#F0EEF8"
TEXT_2     = "#9490B0"
TEXT_3     = "#5A5570"
BORDER     = "#252535"

TURBO_TAGS = [
    "[laugh]", "[chuckle]", "[sigh]", "[gasp]",
    "[cough]", "[groan]", "[sniff]", "[shush]", "[clear throat]"
]

# ─── Helpers ───────────────────────────────────────────────────
def load_settings():
    try:
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    except:
        return {}

def save_settings(data):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except:
        pass

def chunk_script(text, max_chars=MAX_CHARS):
    sentences = re.split(r'(?<=[.!?…])\s+', text.strip())
    chunks, current = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(current) + len(s) < max_chars:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]

# ─── Generation core (shared by Single + Batch) ────────────────
class JobPaused(Exception):
    """Raised inside the worker thread when the user pressed Pause."""


def synth_chunk(model, model_key, text, voice, lang, exag, cfg, temp, rep):
    """One model call. Identical calls to the original app, just in one place."""
    if model_key == "turbo":
        return model.generate(
            text,
            audio_prompt_path=voice,
            exaggeration=exag,
            temperature=temp,
        )
    elif model_key == "multilingual":
        try:
            return model.generate(
                text,
                audio_prompt_path=voice,
                language_id=lang,
                exaggeration=exag,
                cfg_weight=cfg,
                temperature=temp,
                repetition_penalty=rep,
            )
        except TypeError:
            try:
                return model.generate(text,
                    audio_prompt_path=voice,
                    language=lang,
                    exaggeration=exag,
                    cfg_weight=cfg)
            except TypeError:
                return model.generate(text,
                    audio_prompt_path=voice,
                    language=lang)
    else:
        return model.generate(
            text,
            audio_prompt_path=voice,
            exaggeration=exag,
            cfg_weight=cfg,
        )


def chunk_audio_quality(wav, sr, text):
    """
    Cheap sanity check for one generated chunk. Returns (ok, deviation).
    Catches the classic TTS failures: silence / NaN, cut-off one-word output,
    and runaway babbling or looping that is far longer than the text needs.
    """
    try:
        if wav is None or wav.numel() == 0:
            return False, 99.0
        if not bool(torch.isfinite(wav).all()):
            return False, 99.0
        if float(wav.abs().max()) < 1e-4:
            return False, 99.0
        dur = wav.shape[-1] / float(sr)
        n   = max(1, len(text.strip()))
        exp = n / 14.0                       # ~14 characters per second of speech
        lo  = exp * 0.2 if n >= 25 else 0.0
        hi  = exp * 2.8 + 1.5
        dev = abs(dur - exp) / max(exp, 0.5)
        return (lo <= dur <= hi), dev
    except Exception:
        return True, 0.0


def smooth_edges(wav, sr, fade_in_ms=4, fade_out_ms=12):
    """Tiny fades on both ends of a chunk so joins never click or pop."""
    w = wav.clone()
    n = w.shape[-1]
    fi = min(int(sr * fade_in_ms / 1000), n // 4)
    fo = min(int(sr * fade_out_ms / 1000), n // 4)
    if fi > 1:
        w[..., :fi] *= torch.linspace(0.0, 1.0, fi)
    if fo > 1:
        w[..., -fo:] *= torch.linspace(1.0, 0.0, fo)
    return w


def assemble_take(parts, sr, pause_ms, smooth):
    silence = torch.zeros(1, int(sr * (pause_ms / 1000)))
    seq = []
    for w in parts:
        seq.extend([smooth_edges(w, sr) if smooth else w, silence])
    return torch.cat(seq, dim=1)


def fmt_eta(seconds):
    seconds = int(max(0, seconds))
    h, r = divmod(seconds, 3600)
    m, _ = divmod(r, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"

# ─── Checkpoint / resume system ────────────────────────────────
def session_dir(job_id):
    return os.path.join(RESUME_ROOT, job_id)

def _atomic_json_write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def save_manifest(job):
    _atomic_json_write(os.path.join(session_dir(job["id"]), "job.json"), job)

def load_manifest(job_id):
    try:
        with open(os.path.join(session_dir(job_id), "job.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def ckpt_path(job_id, item_i, take, sub_i):
    return os.path.join(session_dir(job_id), f"i{item_i:04d}_t{take}_c{sub_i:05d}.pt")

def save_ckpt(path, wav):
    """Write a finished chunk to disk safely (temp file, fsync, atomic rename)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(wav.detach().cpu().contiguous().clone(), f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def load_ckpt(path):
    if not os.path.exists(path):
        return None
    try:
        t = torch.load(path, map_location="cpu")
        if isinstance(t, torch.Tensor) and t.ndim == 2 and t.numel() > 0:
            return t
    except Exception:
        pass
    try:                      # unreadable (e.g. power cut mid-write) -> regenerate it
        os.remove(path)
    except OSError:
        pass
    return None

def save_wav_atomic(path, audio, sr):
    tmp = os.path.splitext(path)[0] + ".partial.wav"
    torchaudio.save(tmp, audio, sr)
    os.replace(tmp, path)

def out_name_for(item, take, takes):
    if takes == 1:
        return item["stem"] + item["ext"]
    return f"{item['stem']}_t{take}{item['ext']}"

def _voice_sig(path):
    try:
        st = os.stat(path)
        return [st.st_size, int(st.st_mtime)]
    except OSError:
        return None

def build_job(kind, items, p):
    key = {
        "kind": kind,
        "items": [[i["stem"], i["ext"], i["text"]] for i in items],
        "model": p["model_key"], "voice": p["voice"], "voice_sig": _voice_sig(p["voice"]),
        "lang": p["lang"], "takes": p["takes"], "folder": p["folder"],
        "exag": round(p["exag"], 3), "cfg": round(p["cfg"], 3),
        "temp": round(p["temp"], 3), "rep": round(p["rep"], 3),
        "guard": p["guard"], "max_chars": MAX_CHARS, "pause_ms": PAUSE_MS,
    }
    blob = json.dumps(key, sort_keys=True, ensure_ascii=False)
    job_id = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
    job = {
        "version": 1, "id": job_id, "kind": kind,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "items": items, "done_outputs": [],
        "max_chars": MAX_CHARS, "pause_ms": PAUSE_MS,
    }
    job.update(p)
    return job

def count_session_progress(job):
    """(chunks already saved, total chunks) for a job, counting every take."""
    mc    = job.get("max_chars", MAX_CHARS)
    takes = job["takes"]
    done_set = set(job.get("done_outputs", []))
    done = total = 0
    for ii, it in enumerate(job["items"]):
        n = len(chunk_script(it["text"], mc))
        for t in range(1, takes + 1):
            total += n
            if f"{ii}:{t}" in done_set:
                done += n
            else:
                done += sum(1 for ci in range(n)
                            if os.path.exists(ckpt_path(job["id"], ii, t, ci)))
    return done, total

def find_unfinished_jobs():
    out = []
    if not os.path.isdir(RESUME_ROOT):
        return []
    for name in os.listdir(RESUME_ROOT):
        d = os.path.join(RESUME_ROOT, name)
        try:
            with open(os.path.join(d, "job.json"), encoding="utf-8") as f:
                job = json.load(f)
            out.append((os.path.getmtime(d), job))
        except Exception:
            continue
    out.sort(key=lambda x: x[0], reverse=True)
    return [j for _, j in out]

# ─── Audio playback ────────────────────────────────────────────
_playback_lock   = threading.Lock()
_stop_flag       = threading.Event()
_playback_thread = None

def play_audio(path):
    global _playback_thread
    _stop_flag.set()
    if _playback_thread and _playback_thread.is_alive():
        _playback_thread.join(timeout=1.0)
    _stop_flag.clear()

    def _play():
        try:
            if AUDIO_BACKEND == "pygame":
                with _playback_lock:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                    pygame.mixer.music.load(path)
                    pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    if _stop_flag.is_set():
                        with _playback_lock:
                            pygame.mixer.music.stop()
                        break
                    pygame.time.Clock().tick(10)
            else:
                # Fallback: open with system player
                if sys.platform == "win32":
                    os.startfile(path)
        except Exception as e:
            print(f"Playback error: {e}")

    _playback_thread = threading.Thread(target=_play, daemon=True)
    _playback_thread.start()

def stop_audio():
    _stop_flag.set()
    if AUDIO_BACKEND == "pygame":
        try:
            with _playback_lock:
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
        except:
            pass

# ─── Batch queue notification sounds ───────────────────────────
# Generated tones (no external sound files needed) so a distinct
# "chunk done" ding and a different "batch finished" tune can play
# automatically while the queue runs unattended.
def _get_mixer_rate_and_channels():
    try:
        info = pygame.mixer.get_init()
        if info:
            return info[0], info[2]
    except Exception:
        pass
    return 44100, 2

def _make_tone_array(freq=880, duration_ms=150, volume=0.4):
    rate, channels = _get_mixer_rate_and_channels()
    n_samples = max(1, int(rate * duration_ms / 1000))
    t = np.linspace(0, duration_ms / 1000, n_samples, False)
    wave = np.sin(freq * t * 2 * np.pi)
    fade = min(200, n_samples // 4)
    if fade > 0:
        env = np.ones(n_samples)
        env[:fade]  = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        wave = wave * env
    audio = (wave * volume * 32767).astype(np.int16)
    if channels == 1:
        return np.ascontiguousarray(audio)
    return np.ascontiguousarray(np.column_stack([audio, audio]))

def play_chunk_done_sound():
    """Short two-note 'ding' played after each queued chunk finishes."""
    try:
        if AUDIO_BACKEND == "pygame":
            for freq, dur in [(880, 110), (1175, 160)]:
                snd = pygame.sndarray.make_sound(_make_tone_array(freq, dur, 0.35))
                snd.play()
                pygame.time.wait(dur)
        elif sys.platform == "win32":
            import winsound
            winsound.Beep(880, 110)
            winsound.Beep(1175, 160)
    except Exception as e:
        print(f"Notification sound error: {e}")

def play_batch_complete_sound():
    """Distinct little melody played once the whole queue is finished."""
    try:
        if AUDIO_BACKEND == "pygame":
            for freq, dur in [(523, 150), (659, 150), (784, 150), (1046, 260)]:
                snd = pygame.sndarray.make_sound(_make_tone_array(freq, dur, 0.4))
                snd.play()
                pygame.time.wait(dur)
        elif sys.platform == "win32":
            import winsound
            for freq, dur in [(523, 150), (659, 150), (784, 150), (1046, 260)]:
                winsound.Beep(freq, dur)
    except Exception as e:
        print(f"Completion sound error: {e}")

# ─── Model loading ─────────────────────────────────────────────
print("Loading Standard + Multilingual models...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

from chatterbox.tts       import ChatterboxTTS
from chatterbox.mtl_tts   import ChatterboxMultilingualTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS

english_model      = ChatterboxTTS.from_pretrained(device=DEVICE)
multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
turbo_model        = None

def get_turbo():
    global turbo_model
    if turbo_model is None:
        print("Loading Turbo model...")
        turbo_model = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
        print("Turbo loaded ✓")
    return turbo_model

print("Models loaded ✓")

# ─── Widgets ───────────────────────────────────────────────────
class StyledCard(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color=BG_CARD, corner_radius=12,
                         border_width=1, border_color=BORDER, **kwargs)

class SectionLabel(ctk.CTkLabel):
    def __init__(self, master, text, **kwargs):
        super().__init__(master, text=text, font=("Helvetica", 11, "bold"),
                         text_color=TEXT_3, **kwargs)

class BrowseRow(ctk.CTkFrame):
    def __init__(self, master, label, placeholder, browse_fn, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.columnconfigure(1, weight=1)
        SectionLabel(self, text=label.upper()).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 4))
        self.entry = ctk.CTkEntry(
            self, placeholder_text=placeholder, height=38,
            fg_color=BG_INPUT, border_color=BORDER, border_width=1,
            text_color=TEXT_1, placeholder_text_color=TEXT_3,
            font=("Helvetica", 13), corner_radius=8)
        self.entry.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 8))
        ctk.CTkButton(self, text="Browse", width=80, height=38,
            fg_color=ACCENT_DIM, hover_color=ACCENT, text_color=TEXT_1,
            font=("Helvetica", 12, "bold"), corner_radius=8,
            command=browse_fn).grid(row=1, column=2, sticky="e")

    def get(self): return self.entry.get().strip()
    def set(self, val):
        self.entry.delete(0, "end")
        self.entry.insert(0, val)

# ─── History Panel ─────────────────────────────────────────────
class HistoryPanel(ctk.CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color=BG_PANEL,
                         scrollbar_button_color=BORDER, **kwargs)
        self._rows = {}

    def add(self, filename, path, duration_s, model_tag="STD"):
        if path in self._rows:
            self._rows[path].destroy()

        row = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8,
                           border_width=1, border_color=BORDER)
        row.pack(fill="x", padx=6, pady=4)
        self._rows[path] = row

        left = ctk.CTkFrame(row, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=10, pady=8)

        tag_color = TEAL if model_tag == "TURBO" else (ACCENT2 if model_tag == "MULTI" else ACCENT_DIM)
        ctk.CTkLabel(left,
            text=f"  {model_tag}  ",
            font=("Helvetica", 10, "bold"),
            text_color=TEXT_1,
            fg_color=tag_color,
            corner_radius=4).pack(anchor="w", pady=(0, 3))

        ctk.CTkLabel(left, text=filename, font=("Helvetica", 13, "bold"),
                     text_color=TEXT_1, anchor="w").pack(anchor="w")
        ctk.CTkLabel(left, text=f"{duration_s:.1f}s",
                     font=("Helvetica", 11), text_color=TEXT_3,
                     anchor="w").pack(anchor="w")

        btn_frame = ctk.CTkFrame(row, fg_color="transparent")
        btn_frame.pack(side="right", padx=8, pady=8)

        play_btn = ctk.CTkButton(btn_frame, text="▶", width=38, height=38,
            fg_color=ACCENT_DIM, hover_color=ACCENT, text_color=TEXT_1,
            font=("Helvetica", 15, "bold"), corner_radius=8)

        def on_play(p=path, pb=play_btn):
            stop_audio()
            play_audio(p)
            pb.configure(fg_color=GREEN)
            pb.after(600, lambda: pb.configure(fg_color=ACCENT_DIM))

        play_btn.configure(command=on_play)
        play_btn.pack(side="left", padx=(0, 4))

        ctk.CTkButton(btn_frame, text="■", width=38, height=38,
            fg_color="#1e1e2e", hover_color="#2a1a1a", text_color=AMBER,
            font=("Helvetica", 14, "bold"), corner_radius=8,
            command=stop_audio).pack(side="left", padx=(0, 4))

        ctk.CTkButton(btn_frame, text="📂", width=38, height=38,
            fg_color="transparent", hover_color=BG_INPUT, font=("Helvetica", 16),
            command=lambda p=path: os.startfile(os.path.dirname(p))
            ).pack(side="left", padx=(0, 4))

        def on_delete(r=row, p=path):
            if messagebox.askyesno("Delete", f"Delete '{Path(p).name}' from disk?"):
                stop_audio()
                import time; time.sleep(0.5)
                try:
                    if os.path.exists(p):
                        os.remove(p)
                    r.pack_forget()
                    r.destroy()
                    self._rows.pop(p, None)
                except PermissionError:
                    messagebox.showerror("File In Use",
                        "Stop playback first, wait a moment, then delete.")
                except Exception as e:
                    messagebox.showerror("Error", str(e))

        ctk.CTkButton(btn_frame, text="🗑", width=38, height=38,
            fg_color="transparent", hover_color="#2a1010", text_color=RED,
            font=("Helvetica", 16), command=on_delete).pack(side="left")

# ─── Main App ──────────────────────────────────────────────────
class VoiceStudio(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Voice Studio")
        self.geometry("1540x960")
        self.minsize(1200, 800)
        self.configure(fg_color=BG_MAIN)
        self.settings = load_settings()
        self._selected_model = "standard"
        self.ui = {}
        self._job_state  = "idle"      # idle | running | pausing | paused
        self._active_key = None
        self._active_job = None
        self._banner_job = None
        self._pause_req  = threading.Event()
        self._build_ui()
        self._restore_settings()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_banners()

    def _slider(self, parent, label, from_, to_, default, attr,
                row=0, col=0, pad_right=0):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=col, sticky="ew",
                   padx=(0, pad_right), pady=(4, 0))
        SectionLabel(frame, text=label).pack(anchor="w", padx=4)
        val_lbl = ctk.CTkLabel(frame, text=f"{default:.2f}",
            font=("Helvetica", 12, "bold"), text_color=ACCENT)
        val_lbl.pack(anchor="e", padx=4)
        sl = ctk.CTkSlider(frame, from_=from_, to=to_,
            fg_color=BG_INPUT, progress_color=ACCENT,
            button_color=ACCENT2, button_hover_color=ACCENT, height=16)
        sl.set(default)
        sl.pack(fill="x", padx=4)
        sl.configure(command=lambda v, lbl=val_lbl: lbl.configure(text=f"{v:.2f}"))
        setattr(self, attr, sl)
        setattr(self, attr + "_lbl", val_lbl)
        return frame

    def _build_ui(self):
        # ── Top bar ──
        topbar = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0,
                              height=58, border_width=0)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)
        ctk.CTkLabel(topbar, text="⬡  Voice Studio",
            font=("Helvetica", 20, "bold"), text_color=TEXT_1
            ).pack(side="left", padx=24, pady=14)

        # Warn if pygame missing
        if AUDIO_BACKEND != "pygame":
            ctk.CTkLabel(topbar,
                text="  ⚠ pip install pygame  for in-app playback  ",
                font=("Helvetica", 11, "bold"), text_color=AMBER,
                fg_color="#2a2010", corner_radius=6
            ).pack(side="right", padx=8)

        ctk.CTkLabel(topbar,
            text=f"  {'GPU ✓' if DEVICE=='cuda' else 'CPU'}  ",
            font=("Helvetica", 12, "bold"),
            text_color=GREEN if DEVICE=="cuda" else AMBER,
            fg_color="#1a3326" if DEVICE=="cuda" else "#2a2010",
            corner_radius=6).pack(side="right", padx=24)
        ctk.CTkLabel(topbar, text="Standard · Turbo · Multilingual 23 langs",
            font=("Helvetica", 12), text_color=TEXT_3
            ).pack(side="right", padx=8)

        # ── Tabs: Single Generation (original UI, untouched) + Batch Queue (new) ──
        self.tabview = ctk.CTkTabview(self, fg_color=BG_MAIN,
            segmented_button_fg_color=BG_PANEL,
            segmented_button_selected_color=ACCENT,
            segmented_button_selected_hover_color=ACCENT2,
            segmented_button_unselected_color=BG_PANEL,
            text_color=TEXT_1, corner_radius=0)
        self.tabview.pack(fill="both", expand=True)
        self.tabview.add("🎙  Single Generation")
        self.tabview.add("📋  Batch Queue")
        tab_single = self.tabview.tab("🎙  Single Generation")
        tab_batch  = self.tabview.tab("📋  Batch Queue")
        self._tab_names = {"single": "🎙  Single Generation", "batch": "📋  Batch Queue"}

        # ── Body ──
        body = ctk.CTkFrame(tab_single, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=20, pady=(16, 20))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=3)
        body.columnconfigure(2, weight=2)
        body.rowconfigure(0, weight=1)

        # ── LEFT — Script ──
        left = ctk.CTkFrame(body, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        ctk.CTkLabel(left, text="Script", font=("Helvetica", 22, "bold"),
            text_color=TEXT_1).grid(row=0, column=0, sticky="w", pady=(0, 10))

        sc = StyledCard(left)
        sc.grid(row=1, column=0, sticky="nsew")
        sc.rowconfigure(0, weight=1)
        sc.columnconfigure(0, weight=1)

        self.script_box = ctk.CTkTextbox(sc, font=("Helvetica", 15),
            fg_color="transparent", text_color=TEXT_1,
            border_width=0, wrap="word")
        self.script_box.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)
        self.script_box.bind("<KeyRelease>", self._update_counts)

        # Turbo tag bar
        self.tag_bar = ctk.CTkFrame(sc, fg_color=BG_INPUT, corner_radius=0)
        self.tag_bar.grid(row=1, column=0, sticky="ew")
        self.tag_bar.grid_remove()

        ctk.CTkLabel(self.tag_bar, text="Insert tag:",
            font=("Helvetica", 11), text_color=TEXT_3
            ).pack(side="left", padx=(10, 6), pady=6)

        for tag in TURBO_TAGS:
            ctk.CTkButton(self.tag_bar, text=tag,
                width=0, height=28,
                fg_color="#1a1a2e", hover_color=ACCENT_DIM,
                text_color=TEAL, font=("Helvetica", 11, "bold"),
                corner_radius=6,
                command=lambda t=tag: self._insert_tag(t)
            ).pack(side="left", padx=3, pady=6)

        cbar = ctk.CTkFrame(sc, fg_color="#0D0D0F", corner_radius=0, height=32)
        cbar.grid(row=2, column=0, sticky="ew")
        self.word_count  = ctk.CTkLabel(cbar, text="0 words",
            font=("Helvetica", 11), text_color=TEXT_3)
        self.word_count.pack(side="left", padx=12)
        self.chunk_count = ctk.CTkLabel(cbar, text="0 chunks",
            font=("Helvetica", 11), text_color=TEXT_3)
        self.chunk_count.pack(side="right", padx=12)

        # ── MIDDLE — Settings (scrollable so Generate is always visible) ──
        mid_outer = ctk.CTkFrame(body, fg_color="transparent")
        mid_outer.grid(row=0, column=1, sticky="nsew", padx=(0, 10))
        mid_outer.columnconfigure(0, weight=1)
        mid_outer.rowconfigure(1, weight=1)

        ctk.CTkLabel(mid_outer, text="Settings", font=("Helvetica", 22, "bold"),
            text_color=TEXT_1).grid(row=0, column=0, sticky="w", pady=(0, 10))

        # Scrollable container for settings + generate button
        mid_scroll = ctk.CTkScrollableFrame(mid_outer, fg_color="transparent",
            scrollbar_button_color=BORDER, corner_radius=0)
        mid_scroll.grid(row=1, column=0, sticky="nsew")
        mid_scroll.columnconfigure(0, weight=1)

        sc2 = StyledCard(mid_scroll)
        sc2.pack(fill="x", pady=(0, 10))
        sc2.columnconfigure(0, weight=1)

        inn = ctk.CTkFrame(sc2, fg_color="transparent")
        inn.pack(fill="x", padx=16, pady=14)
        inn.columnconfigure(0, weight=1)

        # ── Model selector ──
        SectionLabel(inn, text="MODEL").grid(
            row=0, column=0, sticky="w", padx=4, pady=(0, 6))

        model_row = ctk.CTkFrame(inn, fg_color="transparent")
        model_row.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        model_row.columnconfigure(0, weight=1)
        model_row.columnconfigure(1, weight=1)
        model_row.columnconfigure(2, weight=1)

        self.model_btns = {}
        models = [
            ("Standard", "standard", ACCENT_DIM,
             "High quality EN\nEmotion sliders"),
            ("Turbo", "turbo", "#1a3a38",
             "Fastest · EN only\n[laugh][sigh] tags"),
            ("Multilingual", "multilingual", "#2a1a3a",
             "23 languages\nVoice cloning"),
        ]
        for i, (label, key, bg, tip) in enumerate(models):
            btn = ctk.CTkButton(model_row,
                text=f"{label}\n{tip.split(chr(10))[1]}",
                height=52, fg_color=bg,
                hover_color=ACCENT,
                text_color=TEXT_1,
                font=("Helvetica", 12, "bold"),
                corner_radius=10,
                command=lambda k=key: self._select_model(k))
            btn.grid(row=0, column=i, sticky="ew",
                     padx=(0 if i == 0 else 4, 0))
            self.model_btns[key] = btn

        # Reference voice + output
        self.ref_row = BrowseRow(inn, "Reference Voice",
            "Select .wav voice sample...", self._browse_reference)
        self.ref_row.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.out_row = BrowseRow(inn, "Output Folder",
            "Select output folder...", self._browse_output)
        self.out_row.grid(row=3, column=0, sticky="ew", pady=(0, 8))

        # Filename
        SectionLabel(inn, text="OUTPUT FILENAME").grid(
            row=4, column=0, sticky="w", padx=4, pady=(0, 4))
        fn_row = ctk.CTkFrame(inn, fg_color="transparent")
        fn_row.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        fn_row.columnconfigure(0, weight=1)
        self.filename_entry = ctk.CTkEntry(fn_row,
            placeholder_text="voiceover.wav", height=38,
            fg_color=BG_INPUT, border_color=BORDER, border_width=1,
            text_color=TEXT_1, placeholder_text_color=TEXT_3,
            font=("Helvetica", 13), corner_radius=8)
        self.filename_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.filename_entry.insert(0, "voiceover.wav")
        ctk.CTkButton(fn_row, text="Auto", width=60, height=38,
            fg_color=ACCENT_DIM, hover_color=ACCENT, text_color=TEXT_1,
            font=("Helvetica", 12, "bold"), corner_radius=8,
            command=self._auto_name).grid(row=0, column=1)

        # Lang + Takes
        lt = ctk.CTkFrame(inn, fg_color="transparent")
        lt.grid(row=6, column=0, sticky="ew", pady=(0, 8))
        lt.columnconfigure(0, weight=1)
        lt.columnconfigure(1, weight=1)

        # Language (shown only for multilingual)
        self.lang_col = ctk.CTkFrame(lt, fg_color="transparent")
        self.lang_col.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.lang_col.columnconfigure(0, weight=1)
        SectionLabel(self.lang_col, text="LANGUAGE").grid(
            row=0, column=0, sticky="w", padx=4, pady=(0, 4))
        self.lang_menu = ctk.CTkOptionMenu(self.lang_col,
            values=["en — English", "hi — Hindi", "te — Telugu",
                    "de — German", "fr — French", "es — Spanish",
                    "ja — Japanese", "ko — Korean", "zh — Chinese",
                    "ar — Arabic", "ru — Russian", "pt — Portuguese",
                    "it — Italian", "nl — Dutch", "pl — Polish",
                    "sv — Swedish", "tr — Turkish"],
            fg_color=BG_INPUT, button_color=ACCENT_DIM,
            button_hover_color=ACCENT, dropdown_fg_color=BG_CARD,
            dropdown_hover_color=ACCENT_DIM,
            text_color=TEXT_1, dropdown_text_color=TEXT_1,
            font=("Helvetica", 13), height=38, corner_radius=8,
            dynamic_resizing=False)
        self.lang_menu.grid(row=1, column=0, sticky="ew")
        self.lang_menu.set("en — English")

        tc = ctk.CTkFrame(lt, fg_color="transparent")
        tc.grid(row=0, column=1, sticky="ew")
        tc.columnconfigure(0, weight=1)
        SectionLabel(tc, text="TAKES").grid(row=0, column=0, sticky="w",
            padx=4, pady=(0, 4))
        self.takes_seg = ctk.CTkSegmentedButton(tc, values=[str(n) for n in range(1, MAX_TAKES + 1)],
            font=("Helvetica", 14, "bold"),
            selected_color=ACCENT, selected_hover_color=ACCENT2,
            unselected_color=BG_INPUT, unselected_hover_color=ACCENT_DIM,
            text_color=TEXT_1, height=38, corner_radius=8)
        self.takes_seg.grid(row=1, column=0, sticky="ew")
        self.takes_seg.set("1")

        # ── Voice Parameters ──
        sep = ctk.CTkFrame(inn, fg_color=BORDER, height=1)
        sep.grid(row=7, column=0, sticky="ew", pady=(6, 8))

        SectionLabel(inn, text="VOICE PARAMETERS").grid(
            row=8, column=0, sticky="w", padx=4, pady=(0, 4))

        params = ctk.CTkFrame(inn, fg_color="transparent")
        params.grid(row=9, column=0, sticky="ew")
        params.columnconfigure(0, weight=1)
        params.columnconfigure(1, weight=1)

        # Row 0: Exaggeration (always visible) + CFG (hidden for turbo)
        self._slider(params, "EXAGGERATION  (0=flat · 1=expressive · 2=dramatic)",
                     0.0, 2.0, 0.5, "exag_val", row=0, col=0, pad_right=8)
        self.cfg_frame = self._slider(params,
                     "CFG WEIGHT  (clone strength · 0=ignore · 1=strict)",
                     0.0, 1.0, 0.5, "cfg_val", row=0, col=1)

        # Row 1: Temperature (always visible) + Repetition Penalty (hidden for turbo)
        self._slider(params, "TEMPERATURE  (lower=steady · higher=varied)",
                     0.05, 2.0, 0.8, "temp_val", row=1, col=0, pad_right=8)
        self.rep_frame = self._slider(params, "REPETITION PENALTY",
                     1.0, 3.0, 1.2, "rep_val",  row=1, col=1)

        # Turbo note
        self.turbo_note = ctk.CTkLabel(inn,
            text="⚡ Turbo: use [laugh] [sigh] [chuckle] etc. tags for emotion.",
            font=("Helvetica", 11), text_color=TEAL,
            wraplength=440, justify="left")
        self.turbo_note.grid(row=10, column=0, sticky="w", padx=4, pady=(6, 0))
        self.turbo_note.grid_remove()

        # Quality guard: auto-retry glitchy chunks + click-free joins
        self.guard_var = ctk.IntVar(value=0)
        self.guard_chk = ctk.CTkCheckBox(inn,
            text="Quality guard  (OPTIONAL · auto-retry glitchy chunks + click-free joins)",
            variable=self.guard_var, onvalue=1, offvalue=0,
            fg_color=ACCENT, hover_color=ACCENT2, text_color=TEXT_2,
            font=("Helvetica", 12), checkbox_width=20, checkbox_height=20)
        self.guard_chk.grid(row=11, column=0, sticky="w", padx=4, pady=(12, 0))

        # ── Generate button (inside scroll area so always reachable) ──
        self.gen_btn = ctk.CTkButton(mid_scroll,
            text="▶   Generate Voiceover", height=54,
            fg_color=ACCENT, hover_color=ACCENT2, text_color="#FFFFFF",
            font=("Helvetica", 17, "bold"), corner_radius=12,
            command=self._generate)
        self.gen_btn.pack(fill="x", pady=(0, 8))
        self._make_controls("single", mid_scroll, self.gen_btn)

        # Progress card
        pc = StyledCard(mid_scroll)
        pc.pack(fill="x", pady=(0, 8))
        self.progress_bar = ctk.CTkProgressBar(pc,
            fg_color=BG_INPUT, progress_color=ACCENT, height=6, corner_radius=3)
        self.progress_bar.pack(fill="x", padx=14, pady=(12, 6))
        self.progress_bar.set(0)
        self.status_label = ctk.CTkLabel(pc, text="Ready to generate",
            font=("Helvetica", 13), text_color=TEXT_2)
        self.status_label.pack(padx=14, pady=(0, 10))
        self.status_label.configure(wraplength=480)
        self.ui["single"]["progress"] = self.progress_bar
        self.ui["single"]["status"]   = self.status_label

        # ── RIGHT — History ──
        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=0, column=2, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(right, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        hdr.columnconfigure(0, weight=1)
        ctk.CTkLabel(hdr, text="History",
            font=("Helvetica", 22, "bold"), text_color=TEXT_1
            ).grid(row=0, column=0, sticky="w")
        brow = ctk.CTkFrame(hdr, fg_color="transparent")
        brow.grid(row=0, column=1, sticky="e")
        ctk.CTkButton(brow, text="■ Stop", width=74, height=34,
            fg_color="#2a1a1a", hover_color="#3a2020", text_color=AMBER,
            font=("Helvetica", 12, "bold"), corner_radius=8,
            command=stop_audio).pack(side="left", padx=(0, 6))
        ctk.CTkButton(brow, text="📂 Folder", width=96, height=34,
            fg_color=ACCENT_DIM, hover_color=ACCENT, text_color=TEXT_1,
            font=("Helvetica", 12, "bold"), corner_radius=8,
            command=self._open_output_folder).pack(side="left")

        ho = StyledCard(right)
        ho.grid(row=1, column=0, sticky="nsew")
        ho.rowconfigure(0, weight=1)
        ho.columnconfigure(0, weight=1)
        self.history = HistoryPanel(ho)
        self.history.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.history_hint = ctk.CTkLabel(ho,
            text="Generated files appear here.\n▶ Play   ■ Stop   📂 Folder   🗑 Delete",
            font=("Helvetica", 12), text_color=TEXT_3, justify="center")
        self.history_hint.grid(row=1, column=0, pady=(0, 12))

        # ── Batch Queue tab (new — uses Reference Voice / Output Folder /
        #     Model / Voice Parameters from the Single tab above) ──
        self._build_batch_tab(tab_batch)

        # Apply model selection after all widgets are built
        self._select_model("standard", init=True)

    # ── Model selector ──
    def _select_model(self, key, init=False):
        self._selected_model = key
        colors = {
            "standard":    (ACCENT,  "#1a1a3a"),
            "turbo":       (TEAL,    "#1a3a38"),
            "multilingual":(ACCENT2, "#2a1a3a"),
        }
        for k, btn in self.model_btns.items():
            active, idle = colors[k]
            btn.configure(fg_color=active if k == key else idle,
                          border_width=2 if k == key else 0,
                          border_color=active)

        is_turbo = (key == "turbo")
        is_multi = (key == "multilingual")

        # Turbo tag bar in script panel
        if hasattr(self, "tag_bar"):
            if is_turbo:
                self.tag_bar.grid()
            else:
                self.tag_bar.grid_remove()

        # Turbo note
        if hasattr(self, "turbo_note"):
            if is_turbo:
                self.turbo_note.grid()
            else:
                self.turbo_note.grid_remove()

        # Hide CFG Weight + Repetition Penalty for Turbo (they're ignored)
        if hasattr(self, "cfg_frame"):
            if is_turbo:
                self.cfg_frame.grid_remove()
            else:
                self.cfg_frame.grid()

        if hasattr(self, "rep_frame"):
            if is_turbo:
                self.rep_frame.grid_remove()
            else:
                self.rep_frame.grid()

        # Language selector only for multilingual
        if hasattr(self, "lang_col"):
            if is_multi:
                self.lang_col.grid()
            else:
                self.lang_col.grid_remove()

    def _insert_tag(self, tag):
        self.script_box.insert("insert", f" {tag} ")
        self.script_box.focus()

    # ── Batch Queue tab ──────────────────────────────────────────
    def _build_batch_tab(self, tab):
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

        hdr = ctk.CTkFrame(tab, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 10))
        hdr.columnconfigure(0, weight=1)
        ctk.CTkLabel(hdr, text="Batch Queue", font=("Helvetica", 22, "bold"),
            text_color=TEXT_1).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(hdr,
            text=("Add one row per chunk: a chunk number and its script line. "
                  "Click Start and every row generates automatically, one after "
                  "another — using the Reference Voice, Output Folder, Model and "
                  "Voice Parameters set in the Single Generation tab. A row's "
                  "chunk number is also its output filename (e.g. chunk 33 → 33.wav). "
                  "A ding plays after each chunk finishes, and a different tune plays "
                  "when the whole queue is done. Use Pause any time — every finished "
                  "chunk is saved to disk, so you can resume even after a power cut."),
            font=("Helvetica", 12), text_color=TEXT_3,
            wraplength=760, justify="left"
            ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.batch_rows = []
        self.batch_rows_container = ctk.CTkScrollableFrame(tab,
            fg_color="transparent", scrollbar_button_color=BORDER)
        self.batch_rows_container.grid(row=1, column=0, sticky="nsew",
            padx=20, pady=(0, 0))
        self.batch_rows_container.columnconfigure(0, weight=1)

        add_bar = ctk.CTkFrame(tab, fg_color="transparent")
        add_bar.grid(row=2, column=0, sticky="ew", padx=20, pady=(8, 8))
        ctk.CTkButton(add_bar, text="+  Add Row", height=40, width=140,
            fg_color=ACCENT_DIM, hover_color=ACCENT, text_color=TEXT_1,
            font=("Helvetica", 13, "bold"), corner_radius=8,
            command=lambda: self._add_batch_row()).pack(side="left")
        self._batch_page = 0
        nav = ctk.CTkFrame(add_bar, fg_color="transparent")
        nav.pack(side="right")
        self.batch_prev_btn = ctk.CTkButton(nav, text="◀", width=44, height=40,
            fg_color=ACCENT_DIM, hover_color=ACCENT, text_color=TEXT_1,
            font=("Helvetica", 14, "bold"), corner_radius=8,
            command=lambda: self._batch_goto(self._batch_page - 1))
        self.batch_prev_btn.pack(side="left", padx=(0, 8))
        self.batch_page_lbl = ctk.CTkLabel(nav, text="", font=("Helvetica", 12, "bold"),
            text_color=TEXT_2)
        self.batch_page_lbl.pack(side="left", padx=(0, 8))
        self.batch_next_btn = ctk.CTkButton(nav, text="▶", width=44, height=40,
            fg_color=ACCENT_DIM, hover_color=ACCENT, text_color=TEXT_1,
            font=("Helvetica", 14, "bold"), corner_radius=8,
            command=lambda: self._batch_goto(self._batch_page + 1))
        self.batch_next_btn.pack(side="left")

        bottom = ctk.CTkFrame(tab, fg_color="transparent")
        bottom.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 16))
        bottom.columnconfigure(0, weight=1)

        self.batch_start_btn = ctk.CTkButton(bottom,
            text="▶  Start Batch Generation", height=54,
            fg_color=ACCENT, hover_color=ACCENT2, text_color="#FFFFFF",
            font=("Helvetica", 17, "bold"), corner_radius=12,
            command=self._start_batch_generation)
        self.batch_start_btn.pack(fill="x", pady=(0, 8))
        self._make_controls("batch", bottom, self.batch_start_btn)

        bpc = StyledCard(bottom)
        bpc.pack(fill="x")
        self.batch_progress = ctk.CTkProgressBar(bpc,
            fg_color=BG_INPUT, progress_color=ACCENT, height=6, corner_radius=3)
        self.batch_progress.pack(fill="x", padx=14, pady=(12, 6))
        self.batch_progress.set(0)
        self.batch_status = ctk.CTkLabel(bpc, text="Ready — add rows and start",
            font=("Helvetica", 13), text_color=TEXT_2)
        self.batch_status.pack(padx=14, pady=(0, 10))
        self.batch_status.configure(wraplength=900)
        self.ui["batch"]["progress"] = self.batch_progress
        self.ui["batch"]["status"]   = self.batch_status

        # One default row to start with
        self._add_batch_row()

    def _batch_sync(self):
        """Copy whatever is typed in the visible rows back into the data list."""
        for rd in self.batch_rows:
            if rd.get("frame") is not None:
                rd["chunk"] = rd["chunk_entry"].get().strip()
                rd["text"]  = rd["text_box"].get("1.0", "end").strip()

    def _batch_pages(self):
        return max(1, -(-len(self.batch_rows) // BATCH_PAGE_SIZE))

    def _render_batch_page(self):
        for rd in self.batch_rows:
            if rd.get("frame") is not None:
                try:
                    rd["frame"].destroy()
                except Exception:
                    pass
                rd["frame"] = None
        pages = self._batch_pages()
        self._batch_page = max(0, min(self._batch_page, pages - 1))
        start = self._batch_page * BATCH_PAGE_SIZE
        end   = min(start + BATCH_PAGE_SIZE, len(self.batch_rows))
        for i in range(start, end):
            self._build_batch_row_widgets(self.batch_rows[i], i + 1)
        self.batch_page_lbl.configure(
            text=f"Rows {start + 1}-{end} of {len(self.batch_rows)}  ·  page {self._batch_page + 1}/{pages}")
        self.batch_prev_btn.configure(state="normal" if self._batch_page > 0 else "disabled")
        self.batch_next_btn.configure(state="normal" if self._batch_page < pages - 1 else "disabled")

    def _batch_goto(self, page):
        self._batch_sync()
        self._batch_page = page
        self._render_batch_page()
        try:
            self.batch_rows_container._parent_canvas.yview_moveto(0)
        except Exception:
            pass

    def _build_batch_row_widgets(self, rd, idx):
        row_frame = StyledCard(self.batch_rows_container)
        row_frame.pack(fill="x", pady=6, padx=4)
        row_frame.columnconfigure(1, weight=1)

        row_label = ctk.CTkLabel(row_frame, text=f"Row {idx}",
            font=("Helvetica", 11, "bold"), text_color=TEXT_3)
        row_label.grid(row=0, column=0, columnspan=3, sticky="w",
            padx=12, pady=(8, 0))

        SectionLabel(row_frame, text="CHUNK #").grid(
            row=1, column=0, sticky="w", padx=(12, 6), pady=(4, 0))
        chunk_entry = ctk.CTkEntry(row_frame, width=70, height=36,
            fg_color=BG_INPUT, border_color=BORDER, border_width=1,
            text_color=TEXT_1, font=("Helvetica", 13), corner_radius=8,
            justify="center")
        chunk_entry.grid(row=2, column=0, sticky="nw", padx=(12, 6), pady=(0, 10))
        chunk_entry.insert(0, rd["chunk"])

        text_box = ctk.CTkTextbox(row_frame, height=64, font=("Helvetica", 13),
            fg_color=BG_INPUT, text_color=TEXT_1, border_width=1,
            border_color=BORDER, corner_radius=8, wrap="word")
        text_box.grid(row=2, column=1, sticky="ew", padx=(0, 6), pady=(0, 10))
        if rd["text"]:
            text_box.insert("1.0", rd["text"])

        remove_btn = ctk.CTkButton(row_frame, text="✕", width=36, height=36,
            fg_color="transparent", hover_color="#2a1010", text_color=RED,
            font=("Helvetica", 14, "bold"), corner_radius=8,
            command=lambda r=rd: self._remove_batch_row(r))
        remove_btn.grid(row=2, column=2, sticky="ne", padx=(0, 12), pady=(0, 10))

        rd["frame"]       = row_frame
        rd["label"]       = row_label
        rd["chunk_entry"] = chunk_entry
        rd["text_box"]    = text_box

    def _add_batch_row(self, chunk=None, text="", render=True):
        self._batch_sync()
        n  = len(self.batch_rows) + 1
        rd = {"chunk": str(chunk) if chunk is not None else str(n),
              "text": text, "frame": None}
        self.batch_rows.append(rd)
        if render:
            self._batch_page = self._batch_pages() - 1
            self._render_batch_page()

    def _remove_batch_row(self, row_data):
        if len(self.batch_rows) <= 1:
            messagebox.showinfo("Batch Queue", "Keep at least one row.")
            return
        self._batch_sync()
        for i, r in enumerate(self.batch_rows):
            if r is row_data:
                del self.batch_rows[i]
                break
        else:
            return   # already removed (e.g. double-click on the same X)

        # The removed row is no longer in self.batch_rows, so _render_batch_page()
        # would never destroy its frame and it would stay on screen. Destroy it
        # explicitly, and do it (plus the page rebuild) AFTER this button callback
        # has returned, not from inside the clicked button's own command.
        dead_frame = row_data.get("frame")
        row_data["frame"] = None

        def _finish_remove():
            try:
                if dead_frame is not None:
                    dead_frame.destroy()
                self._batch_sync()
                self._render_batch_page()
            except Exception:
                import traceback
                traceback.print_exc()
        self.after(30, _finish_remove)

    def _start_batch_generation(self):
        if not self.batch_rows:
            messagebox.showerror("Empty Queue", "Add at least one row to the batch queue.")
            return

        p = self._collect_params()
        if not p["voice"] or not os.path.exists(p["voice"]):
            messagebox.showerror("Missing Voice",
                "Select a valid reference voice file in the Single Generation tab.")
            return
        if not p["folder"]:
            messagebox.showerror("Missing Folder",
                "Select an output folder in the Single Generation tab.")
            return

        self._batch_sync()
        items = []
        for rd in self.batch_rows:
            chunk_no = rd["chunk"]
            text     = rd["text"]
            if not chunk_no or not text:
                messagebox.showerror("Incomplete Row",
                    "Every row needs a chunk number and a script line.")
                return
            items.append({"stem": chunk_no, "ext": ".wav", "text": text})

        self._begin_job("batch", items, "batch")

    # ── Callbacks ──
    def _restore_settings(self):
        s = self.settings
        if s.get("voice_sample"):   self.ref_row.set(s["voice_sample"])
        if s.get("output_folder"):  self.out_row.set(s["output_folder"])
        elif DEFAULT_OUTPUT_FOLDER: self.out_row.set(DEFAULT_OUTPUT_FOLDER)

    def _save_current_settings(self):
        save_settings({"voice_sample": self.ref_row.get(),
                       "output_folder": self.out_row.get()})

    def _browse_reference(self):
        p = filedialog.askopenfilename(title="Select Reference Voice",
            filetypes=[("Audio", "*.wav *.mp3")])
        if p: self.ref_row.set(p); self._save_current_settings()

    def _browse_output(self):
        f = filedialog.askdirectory(title="Select Output Folder")
        if f: self.out_row.set(f); self._save_current_settings()

    def _open_output_folder(self):
        folder = self.out_row.get()
        if folder and os.path.exists(folder): os.startfile(folder)
        else: messagebox.showinfo("Info", "Output folder not found.")

    def _auto_name(self):
        ts   = datetime.datetime.now().strftime("take_%H%M%S")
        mdl  = self._selected_model[:3].upper()
        lang = self.lang_menu.get().split(" ")[0] if self._selected_model == "multilingual" else "en"
        self.filename_entry.delete(0, "end")
        self.filename_entry.insert(0, f"{ts}_{mdl}_{lang}.wav")

    def _update_counts(self, _=None):
        text   = self.script_box.get("1.0", "end").strip()
        words  = len(text.split()) if text else 0
        chunks = len(chunk_script(text)) if text else 0
        self.word_count.configure(text=f"{words} words")
        self.chunk_count.configure(text=f"{chunks} chunks")

    def _generate(self):
        script = self.script_box.get("1.0", "end").strip()
        voice  = self.ref_row.get()
        folder = self.out_row.get()
        fname  = self.filename_entry.get().strip()

        if not script:
            messagebox.showerror("Missing Script", "Please enter your script.")
            return
        if not voice or not os.path.exists(voice):
            messagebox.showerror("Missing Voice", "Select a valid reference voice file.")
            return
        if not folder:
            messagebox.showerror("Missing Folder", "Select an output folder.")
            return
        if not fname:
            self._auto_name()
            fname = self.filename_entry.get().strip()
        if not fname.endswith(".wav"):
            fname += ".wav"

        items = [{"stem": Path(fname).stem, "ext": Path(fname).suffix, "text": script}]
        self._begin_job("single", items, "single")

    def _hide_hint(self):
        try: self.history_hint.grid_remove()
        except: pass

    # ══════════════════════════════════════════════════════════════
    #  Pause / Resume engine
    # ══════════════════════════════════════════════════════════════
    def _collect_params(self):
        return {
            "model_key":  self._selected_model,
            "voice":      self.ref_row.get(),
            "folder":     self.out_row.get(),
            "lang":       self.lang_menu.get().split(" ")[0],
            "lang_label": self.lang_menu.get(),
            "takes":      int(self.takes_seg.get()),
            "exag":       float(self.exag_val.get()),
            "cfg":        float(self.cfg_val.get()),
            "temp":       float(self.temp_val.get()),
            "rep":        float(self.rep_val.get()),
            "guard":      bool(self.guard_var.get()),
        }

    def _make_controls(self, key, parent, start_btn):
        """Resume banner (above the start button) + Pause/Resume/Discard row (below it)."""
        banner = ctk.CTkFrame(parent, fg_color="#1c1733", corner_radius=10,
                              border_width=1, border_color=ACCENT)
        banner_lbl = ctk.CTkLabel(banner, text="", font=("Helvetica", 12),
            text_color=TEXT_1, wraplength=470, justify="left", anchor="w")
        banner_lbl.pack(fill="x", padx=12, pady=(10, 6))
        brow = ctk.CTkFrame(banner, fg_color="transparent")
        brow.pack(fill="x", padx=12, pady=(0, 10))
        ctk.CTkButton(brow, text="▶  Resume", height=36, width=120,
            fg_color=GREEN, hover_color="#16a34a", text_color="#0D0D0F",
            font=("Helvetica", 13, "bold"), corner_radius=8,
            command=self._resume_from_banner).pack(side="left", padx=(0, 8))
        ctk.CTkButton(brow, text="✕  Discard", height=36, width=100,
            fg_color="#2a1a1a", hover_color="#3a2020", text_color=RED,
            font=("Helvetica", 12, "bold"), corner_radius=8,
            command=self._discard_banner_job).pack(side="left")

        ctrl = ctk.CTkFrame(parent, fg_color="transparent")
        pause_btn = ctk.CTkButton(ctrl, text="⏸  Pause", height=44,
            fg_color=AMBER, hover_color="#d97706", text_color="#0D0D0F",
            font=("Helvetica", 15, "bold"), corner_radius=10,
            command=self._on_pause_click)
        pause_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))
        discard_btn = ctk.CTkButton(ctrl, text="✕  Discard", height=44, width=110,
            fg_color="#2a1a1a", hover_color="#3a2020", text_color=RED,
            font=("Helvetica", 13, "bold"), corner_radius=10,
            command=self._on_discard_click)
        discard_btn.pack(side="left")

        self.ui[key] = {
            "start_btn": start_btn, "start_text": start_btn.cget("text"),
            "banner": banner, "banner_lbl": banner_lbl,
            "ctrl": ctrl, "pause_btn": pause_btn, "discard_btn": discard_btn,
        }

    # ── state handling ──
    def _set_job_state(self, state):
        self._job_state = state
        for key, ui in self.ui.items():
            if state == "idle":
                ui["start_btn"].configure(state="normal", text=ui["start_text"])
                ui["ctrl"].pack_forget()
            elif key == self._active_key:
                ui["start_btn"].configure(state="disabled",
                    text="⏸  Job paused" if state == "paused" else "⏳  Generating...")
                ui["ctrl"].pack(fill="x", pady=(0, 8), after=ui["start_btn"])
                if state == "running":
                    ui["pause_btn"].configure(state="normal", text="⏸  Pause",
                        fg_color=AMBER, hover_color="#d97706")
                    ui["discard_btn"].configure(state="disabled")
                elif state == "pausing":
                    ui["pause_btn"].configure(state="disabled",
                        text="⏸  Pausing... (finishing current chunk)")
                    ui["discard_btn"].configure(state="disabled")
                elif state == "paused":
                    ui["pause_btn"].configure(state="normal", text="▶  Resume",
                        fg_color=GREEN, hover_color="#16a34a")
                    ui["discard_btn"].configure(state="normal")
            else:
                ui["start_btn"].configure(state="disabled")
        self._refresh_banners()

    def _refresh_banners(self):
        for ui in self.ui.values():
            ui["banner"].pack_forget()
        self._banner_job = None
        if self._job_state != "idle":
            return
        for job in find_unfinished_jobs():
            try:
                done, total = count_session_progress(job)
            except Exception:
                continue
            if done <= 0 or done >= total or job.get("kind") not in self.ui:
                continue
            self._banner_job = job
            ui  = self.ui[job["kind"]]
            pct = int(100 * done / max(1, total))
            what = (Path(job["items"][0]["stem"]).name if job["kind"] == "single"
                    else f"{len(job['items'])} batch rows")
            ui["banner_lbl"].configure(text=(
                f"⏸  Unfinished job found: {what}\n"
                f"{done}/{total} chunks already saved ({pct}%). "
                f"Resume continues exactly where it stopped."))
            ui["banner"].pack(fill="x", pady=(0, 8), before=ui["start_btn"])
            break

    def _set_slider(self, attr, value):
        sl = getattr(self, attr)
        sl.set(value)
        getattr(self, attr + "_lbl").configure(text=f"{value:.2f}")

    def _apply_job_to_ui(self, job):
        """Put a saved job's script + settings back into the UI."""
        self._select_model(job["model_key"])
        self.ref_row.set(job["voice"])
        self.out_row.set(job["folder"])
        self.lang_menu.set(job.get("lang_label", "en — English"))
        self.takes_seg.set(str(job["takes"]))
        self._set_slider("exag_val", job["exag"])
        self._set_slider("cfg_val",  job["cfg"])
        self._set_slider("temp_val", job["temp"])
        self._set_slider("rep_val",  job["rep"])
        self.guard_var.set(1 if job.get("guard", False) else 0)
        if job["kind"] == "single":
            it = job["items"][0]
            self.script_box.delete("1.0", "end")
            self.script_box.insert("1.0", it["text"])
            self.filename_entry.delete(0, "end")
            self.filename_entry.insert(0, it["stem"] + it["ext"])
            self._update_counts()
        else:
            for rd in list(self.batch_rows):
                if rd.get("frame") is not None:
                    rd["frame"].destroy()
                    rd["frame"] = None
            self.batch_rows.clear()
            for it in job["items"]:
                self._add_batch_row(it["stem"], it["text"], render=False)
            self._batch_page = 0
            self._render_batch_page()
        try:
            self.tabview.set(self._tab_names[job["kind"]])
        except Exception:
            pass

    # ── starting ──
    def _begin_job(self, kind, items, key):
        if self._job_state != "idle":
            messagebox.showinfo("Busy", "A job is already running or paused. "
                                        "Finish, resume or discard it first.")
            return
        p = self._collect_params()
        os.makedirs(p["folder"], exist_ok=True)
        self._save_current_settings()
        job = build_job(kind, items, p)

        existing = load_manifest(job["id"])
        if existing:
            try:
                done, total = count_session_progress(existing)
            except Exception:
                done, total = 0, 0
            if done > 0:
                ans = messagebox.askyesnocancel("Saved progress found",
                    f"This exact script + settings already has {done}/{total} chunks "
                    f"saved.\n\nYes  =  resume from where it stopped\n"
                    f"No  =  start over (deletes the saved progress)\n"
                    f"Cancel  =  do nothing")
                if ans is None:
                    return
                if ans:
                    job = existing
                else:
                    shutil.rmtree(session_dir(job["id"]), ignore_errors=True)
        save_manifest(job)
        self._start_job(job, key)

    def _start_job(self, job, key):
        self._active_job = job
        self._active_key = key
        self._pause_req.clear()
        self._set_job_state("running")
        self.ui[key]["progress"].set(0)
        threading.Thread(target=self._job_worker, args=(job, key), daemon=True).start()

    def _resume_from_banner(self):
        job = self._banner_job
        if not job or self._job_state != "idle":
            return
        if not os.path.exists(job["voice"]):
            messagebox.showerror("Missing Voice",
                f"The reference voice used by this job was not found:\n{job['voice']}")
            return
        self._apply_job_to_ui(job)
        self._start_job(job, job["kind"])

    def _discard_banner_job(self):
        job = self._banner_job
        if not job:
            return
        if messagebox.askyesno("Discard saved progress",
                "Delete the saved progress for this job? This cannot be undone."):
            shutil.rmtree(session_dir(job["id"]), ignore_errors=True)
            self._refresh_banners()

    # ── buttons ──
    def _on_pause_click(self):
        if self._job_state == "running":
            self._pause_req.set()
            self._set_job_state("pausing")
            self.ui[self._active_key]["status"].configure(
                text="Pausing — finishing the current chunk, everything is saved...")
        elif self._job_state == "paused":
            self._start_job(self._active_job, self._active_key)

    def _on_discard_click(self):
        if self._job_state != "paused" or not self._active_job:
            return
        if not messagebox.askyesno("Discard job",
                "Throw away this paused job and its saved progress?"):
            return
        key = self._active_key
        shutil.rmtree(session_dir(self._active_job["id"]), ignore_errors=True)
        self._active_job = None
        self._set_job_state("idle")
        self._active_key = None
        self.ui[key]["progress"].set(0)
        self.ui[key]["status"].configure(text="Saved progress discarded")

    def _on_close(self):
        if self._job_state in ("running", "pausing"):
            if not messagebox.askyesno("Generation running",
                    "A job is still running.\n\nEvery finished chunk is already saved to "
                    "disk, so you can resume next time you open the app (only the chunk "
                    "being generated right now is lost).\n\nClose anyway?"):
                return
            stop_audio()
            try:
                self.destroy()
            finally:
                os._exit(0)
        stop_audio()
        self.destroy()

    # ── the worker ──
    def _generate_once(self, job, model, text):
        """One model call; retries on transient errors (e.g. CUDA out-of-memory)."""
        last = None
        for _ in range(3):
            try:
                return synth_chunk(model, job["model_key"], text, job["voice"],
                                   job["lang"], job["exag"], job["cfg"],
                                   job["temp"], job["rep"])
            except Exception as e:
                last = e
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                time.sleep(2)
        raise last

    def _synth_guarded(self, job, model, text, sr):
        guard = job.get("guard", False)
        tries = 4 if guard else 1
        best, best_dev = None, None
        for attempt in range(tries):
            if attempt > 0 and self._pause_req.is_set():
                raise JobPaused()
            wav = self._generate_once(job, model, text).detach().cpu()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            if not guard:
                return wav
            ok, dev = chunk_audio_quality(wav, sr, text)
            if best is None or dev < best_dev:
                best, best_dev = wav, dev
            if ok:
                return wav
        return best

    def _job_worker(self, job, key):
        ui = self.ui[key]

        def status(t):
            self.after(0, lambda t=t: ui["status"].configure(text=t))

        def progress(v):
            self.after(0, ui["progress"].set, v)

        mk        = job["model_key"]
        model_tag = {"standard": "STD", "turbo": "TURBO", "multilingual": "MULTI"}[mk]
        try:
            if mk == "turbo":
                status("Loading Turbo model (first time)...")
                model = get_turbo()
            elif mk == "multilingual":
                model = multilingual_model
            else:
                model = english_model

            sr        = model.sr
            items     = job["items"]
            takes     = job["takes"]
            n_items   = len(items)
            is_batch  = job["kind"] == "batch"
            max_chars = job.get("max_chars", MAX_CHARS)
            pause_ms  = job.get("pause_ms", PAUSE_MS)
            plan      = [chunk_script(it["text"], max_chars) for it in items]
            done_ops, total_ops = count_session_progress(job)
            total_ops = max(1, total_ops)
            done_set  = set(job.get("done_outputs", []))
            gen_times = []
            progress(min(1.0, done_ops / total_ops))

            for ii, item in enumerate(items):
                subs        = plan[ii]
                item_worked = False
                for take in range(1, takes + 1):
                    if f"{ii}:{take}" in done_set:
                        continue
                    out_name = out_name_for(item, take, takes)
                    parts = []
                    for ci, sub in enumerate(subs):
                        if self._pause_req.is_set():
                            raise JobPaused()
                        cp      = ckpt_path(job["id"], ii, take, ci)
                        existed = os.path.exists(cp)
                        wav     = load_ckpt(cp)
                        if wav is None:
                            if existed:
                                done_ops -= 1          # damaged file, redo it
                            row_lbl = (f"Row {ii+1}/{n_items} (chunk {item['stem']}) · "
                                       if is_batch else "")
                            eta_txt = ""
                            if gen_times:
                                avg = sum(gen_times) / len(gen_times)
                                eta_txt = f" · ETA {fmt_eta(avg * (total_ops - done_ops))}"
                            status(f"[{model_tag}] {row_lbl}Take {take}/{takes} · "
                                   f"Chunk {ci+1}/{len(subs)} · "
                                   f"{done_ops}/{total_ops} saved{eta_txt}")
                            t0  = time.time()
                            wav = self._synth_guarded(job, model, sub, sr)
                            save_ckpt(cp, wav)
                            gen_times.append(time.time() - t0)
                            gen_times = gen_times[-40:]
                            done_ops += 1
                            progress(min(1.0, done_ops / total_ops))
                            if DEVICE == "cuda" and done_ops % 20 == 0:
                                torch.cuda.empty_cache()
                        parts.append(wav)

                    audio    = assemble_take(parts, sr, pause_ms, job.get("guard", False))
                    duration = audio.shape[1] / sr
                    out_path = os.path.join(job["folder"], out_name)
                    save_wav_atomic(out_path, audio, sr)

                    done_set.add(f"{ii}:{take}")
                    job.setdefault("done_outputs", []).append(f"{ii}:{take}")
                    save_manifest(job)
                    item_worked = True
                    del parts, audio

                    self.after(0, self.history.add,
                               out_name, out_path, duration, model_tag)
                    self.after(0, self._hide_hint)

                # this row's take(s) are fully done — notify (batch only, like before)
                if is_batch and item_worked:
                    play_chunk_done_sound()

            if is_batch:
                play_batch_complete_sound()
            shutil.rmtree(session_dir(job["id"]), ignore_errors=True)
            self.after(0, self._on_job_finished, key, model_tag)

        except JobPaused:
            self.after(0, self._on_job_paused, key, None)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.after(0, self._on_job_paused, key, str(e))

    def _on_job_finished(self, key, model_tag):
        job = self._active_job
        ui  = self.ui[key]
        ui["progress"].set(1)
        if job and job["kind"] == "batch":
            ui["status"].configure(
                text=f"✓  [{model_tag}] Batch complete — {len(job['items'])} row(s) generated")
        elif job:
            t = job["takes"]
            ui["status"].configure(
                text=f"✓  [{model_tag}] {t} take{'s' if t > 1 else ''} saved")
        self._active_job = None
        self._set_job_state("idle")
        self._active_key = None

    def _on_job_paused(self, key, err):
        job = self._active_job
        ui  = self.ui[key]
        self._set_job_state("paused")
        done, total = count_session_progress(job)
        ui["progress"].set(min(1.0, done / max(1, total)))
        if err:
            ui["status"].configure(text=(
                f"⚠  Stopped by an error — {done}/{total} chunks saved. "
                f"Press Resume to retry from here."))
            messagebox.showerror("Generation Error",
                f"{err}\n\nNothing was lost — {done}/{total} chunks are saved. "
                f"Press Resume to continue.")
        else:
            ui["status"].configure(text=(
                f"⏸  Paused — {done}/{total} chunks saved. Press Resume to continue "
                f"(safe to close the app)."))


if __name__ == "__main__":
    app = VoiceStudio()
    app.mainloop()