#!/usr/bin/env python
"""Re-read an eval_s2t_full.py result with ELF-S2T's own scoring call.

elf_s2t/infer.py::_compute_wer, verbatim:
    safe = [h if h.strip() else "<empty>" for h in hypos]
    return float(jiwer.wer(refs, safe))
refs are the LibriSpeech transcripts lower-cased (their data.py), hyps the T5
decode as is. Corpus-level, no other normalisation.

  .venv_f5eval/bin/python score_elf_s2t.py out/paper/mix48_100000_s2t_elfsde.json
"""
import json
import sys

import jiwer


def main():
    for f in sys.argv[1:]:
        d = json.load(open(f))
        for w, r in d["runs"].items():
            pairs = r["pairs"]
            refs = [p[0].lower() for p in pairs.values()]
            hyps = [p[1] for p in pairs.values()]
            safe = [h if h.strip() else "<empty>" for h in hyps]
            print(f"{f}  sampler {d.get('sampler')}  w={w}  n={len(refs)}  "
                  f"ELF-S2T jiwer.wer {100 * jiwer.wer(refs, safe):.2f}  (ours {r['wer_full'][0]:.2f})")


if __name__ == "__main__":
    main()
