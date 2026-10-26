#!/usr/bin/env python3
"""Compare models on accuracy (WER) and speed with `mynah transcribe`.

  scripts/bench.py fetch [N]          # N clips per set (default 50) from the
                                      # Open ASR Leaderboard test sets
  scripts/bench.py run MODEL.gguf...  # WER + speed per model

Your own clips: drop 16 kHz mono wavs in $BENCH_DIR/wav and add
`name.wav<TAB>reference text` lines to $BENCH_DIR/refs.tsv. A set is the
filename prefix before the first `_`.

Needs the duckdb and ffmpeg CLIs, plus a release build of mynah.
"""
import csv
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

BENCH = Path(os.environ.get("BENCH_DIR", Path.home() / ".cache/mynah-bench"))
MYNAH = Path(__file__).resolve().parent.parent / "target/release/mynah"
HF = "hf://datasets/hf-audio/open-asr-leaderboard"
# (set, parquet shard, min/max clip length): dictation-length clips only.
SETS = [
    ("voxpopuli", "voxpopuli/test-00002-of-00004", 5, 9),  # accented EU parliament
    ("ami", "ami/test-00002-of-00015", 3, 15),  # meetings, spontaneous
    ("earnings22", "earnings22/test-00002-of-00005", 4, 8),  # accented, jargon
    ("libri-other", "librispeech/test.other-00000-of-00001", 4, 15),  # read, harder half
]


def fetch(n):
    csv.field_size_limit(sys.maxsize)
    (BENCH / "wav").mkdir(parents=True, exist_ok=True)
    refs = []
    for name, shard, lo, hi in SETS:
        print(f"fetching {name}...", file=sys.stderr)
        sql = (
            f"SELECT text, hex(audio.bytes) FROM (SELECT * FROM '{HF}/{shard}.parquet' "
            f"WHERE audio_length_s BETWEEN {lo} AND {hi}) "
            f"USING SAMPLE reservoir({n} ROWS) REPEATABLE (42)"
        )
        out = subprocess.run(
            ["duckdb", "-csv", "-noheader", "-c", sql],
            capture_output=True, text=True, check=True,
        ).stdout
        for i, (text, blob) in enumerate(csv.reader(out.splitlines())):
            wav = f"{name}_{i:03}.wav"
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-y", "-i", "pipe:",
                 "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", BENCH / "wav" / wav],
                input=bytes.fromhex(blob), check=True,
            )
            refs.append(f"{wav}\t{text}")
    (BENCH / "refs.tsv").write_text("\n".join(refs) + "\n")
    print(f"{len(refs)} clips in {BENCH}", file=sys.stderr)


# Fillers are dropped like the leaderboard normalizer does: transcribing
# them or not is a style choice, not an error (and dictation wants them gone).
FILLERS = {"um", "uh", "uhm", "umm", "hmm", "mm", "ah", "er", "erm"}


# ponytail: lowercase + strip punctuation + fillers only. No number/spelling
# normalization ("20" vs "twenty" counts as an error), so absolute WER runs
# higher than the leaderboard's; the ranking between models still holds.
def words(s):
    ws = re.sub(r"[^a-z0-9' ]+", " ", s.lower().replace("-", " ")).split()
    return [w for w in ws if w not in FILLERS]


def edits(ref, hyp):
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1]


def run(models):
    refs = [line.split("\t", 1) for line in (BENCH / "refs.tsv").read_text().splitlines() if line]
    wavs = [str(BENCH / "wav" / w) for w, _ in refs]
    print(f"{'model':<44} " + " ".join(f"{s:>11}" for s in sets(refs)) + f" {'ALL':>6} {'RTFx':>6} {'load':>6}")
    for model in models:
        p = subprocess.run(
            [MYNAH, "transcribe", *wavs], capture_output=True, text=True,
            # Empty config dir: the user's vocabulary.txt would skew results.
            env={**os.environ, "MYNAH_MODEL": model, "RUST_LOG": "warn",
                 "XDG_CONFIG_HOME": str(BENCH / "config")},
        )
        hyps = p.stdout.splitlines()
        if p.returncode or len(hyps) != len(refs):
            print(f"{Path(model).name:<44} FAILED: {p.stderr.strip().splitlines()[-1:]}")
            continue
        (BENCH / f"{Path(model).stem}.txt").write_text(p.stdout)
        err, tot = defaultdict(int), defaultdict(int)
        for (wav, ref), hyp in zip(refs, hyps):
            s = wav.split("_")[0]
            r = words(ref)
            err[s] += edits(r, words(hyp))
            tot[s] += len(r)
        times = re.findall(r"inference: ([\d.]+)s for ([\d.]+)s", p.stderr)
        # Skip the first clip: it pays one-time GPU warmup.
        infer = sum(float(t) for t, _ in times[1:])
        audio = sum(float(a) for _, a in times[1:])
        load = re.search(r"load: (\S+)", p.stderr).group(1)
        cols = " ".join(f"{100 * err[s] / tot[s]:>10.2f}%" for s in sets(refs))
        all_wer = 100 * sum(err.values()) / sum(tot.values())
        print(f"{Path(model).name:<44} {cols} {all_wer:>5.2f}% {audio / infer:>6.1f} {load:>6}")


def sets(refs):
    return list(dict.fromkeys(w.split("_")[0] for w, _ in refs))


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["help"]
    if cmd == "fetch":
        fetch(int(rest[0]) if rest else 50)
    elif cmd == "run" and rest:
        run(rest)
    else:
        sys.exit(__doc__)
