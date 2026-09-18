# Version history

## 2026-08-12

- Added a Tailscale-only DNS gateway configuration for `labtest`, combining ad/tracker, phishing, and adult-content blocking for tailnet devices.
- Added a LoRA and early-stopping MADLAD training path plus repetition-aware evaluation after the first full fine-tuning baseline collapsed on the book-disjoint test.
- Switched the tailnet DNS upstream from a Family filter to adult-only filtering after Family restrictions blocked YouTube videos.

## 2026-08-11

- Corrected NLLB target-language routing, added a restricted-data training acknowledgement, and introduced a reproducible book-disjoint Bible evaluation split.
- Added an Apache-2.0 MADLAD-400 3B training path as the 8×A6000 deployable candidate; NLLB 3.3B is now an explicit CC-BY-NC research comparator.
- Added MADLAD-specific generation and evaluation scripts, including auditable prediction outputs and BLEU/chrF++ reporting.
- Added a CC BY 4.0 ASJP Wobé wordlist and two newly located Egner PDF leads, with explicit license-verification gates before text extraction or training.
- Logged the 107-page *Syllabaire Wobe* literacy book as a non-Bible digitization and rights-acquisition target.
- Logged the 2001 *Wè: (parler wobé)* literacy-book lead and named co-author Paul Guei Bozon for rights and community contact research.
- Updated server9 deployment for SSH-key authentication, explicit GPU selection, and data-volume caches; launched the corrected MADLAD-400 3B book-disjoint baseline on GPUs 2 and 7.
