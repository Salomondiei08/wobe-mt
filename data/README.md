# Wobé language data collection

**Collected:** 2026-08-11  
**Language:** Wè Northern / Wobé (`wob`)  
**Purpose:** Local research corpus for Wobé ↔ French machine translation

> **Rights:** The Wobé New Testament is © 2010 Wycliffe Bible Translators, **All rights reserved**.  
> See [`LICENSE_AND_RIGHTS.md`](LICENSE_AND_RIGHTS.md) before any training, sharing, or publication.

---

## Quick stats (measured)

| Resource | Quantity |
|----------|----------|
| **FR ↔ WOB verse pairs** | **7,927** |
| FR ↔ WOB ↔ EN triples | 7,919 |
| Wobé words (NT) | ~275,341 |
| French words (aligned NT) | ~182,884 |
| Wobé unique whitespace tokens | ~7,870 |
| Avg Wobé words / verse | ~34.7 |
| Train / dev / test (deterministic 90/5/5) | 7,186 / 384 / 357 |
| GRN audio tracks | 18 files, ~**122 minutes** est. |
| Open ASJP Wobé wordlist | **30** concept-word entries |
| Literacy-book lead | *Syllabaire Wobe* (1982), 107 pp.; catalogue-only |
| Literacy-book lead | *Wè (parler wobé)* (EDILIS, 2001); catalogue-only |
| Related CI language Bible packages | 9 |
| Total disk | **~206 MB** |

---

## Directory layout

```
data/
├── README.md                          ← this file
├── LICENSE_AND_RIGHTS.md              ← mandatory rights summary
├── aligned/                           ← READY-TO-USE parallel corpora
│   ├── wobe_french_nt_parallel.jsonl  ← main FR–WOB pairs
│   ├── wobe_french_nt_parallel.tsv
│   ├── wobe_french_english_nt_parallel.jsonl  ← FR–WOB–EN triples
│   ├── wobe_french_nt_train.jsonl
│   ├── wobe_french_nt_dev.jsonl
│   ├── wobe_french_nt_test.jsonl
│   ├── wobe_french_nt_verse_plus_sentsplit.jsonl
│   ├── wobe_nt_monolingual.txt
│   ├── french_nt_aligned_monolingual.txt
│   ├── corpus_stats.json
│   └── SAMPLE_PREVIEW.md
├── wobe_bible/                        ← raw Wobé NT (all formats)
│   ├── wob_html.zip / html/
│   ├── wob_usfm.zip / usfm/           ← best for parsing
│   ├── wob_usfx.zip / usfx/
│   ├── wob_readaloud.zip / readaloud/
│   ├── wob.epub
│   └── wob2010eb.zip                  ← Crosswire Sword
├── french_bible/                      ← Louis Segond 1910 (public domain)
│   ├── fraLSG_*.zip
│   ├── html_lsg/ usfm_lsg/
│   └── frasbl_html.zip                ← Free Holy Bible for the World
├── english_bible/                     ← WEB for multi-way align
│   └── engwebp_usfm.zip / usfm/
├── audio_catalog/
│   ├── AUDIO_SOURCES.md
│   ├── grn_wobe_mp3.zip
│   ├── grn_wobe/                      ← 18 Words of Life MP3s
│   └── sample_wobe_noah.mp3
├── related/                           ← other CI / Kru Bibles (transfer)
│   ├── RELATED_LANGUAGES.md
│   ├── nwb_*  (Nyabwa)
│   ├── ted_*  (Tepo Krumen)
│   ├── ktj_*  (Plapo Krumen)
│   ├── kyf_*  (Kouya)
│   ├── dnj_* daf_* (Dan)
│   ├── adj_*  (Adioukrou)
│   ├── any_*  (Anyin)
│   └── yreNT_* (Yaouré)
├── sil/
│   ├── egner1989_page.html
│   ├── Inge_Egner_Precis_de_Grammaire_Wobe_1989.pdf
│   └── Egner_Analyse_conversationnelle_wobe_1988.pdf
├── lexicons/
│   ├── asjp_wobe_wordlist.json      ← CC BY 4.0, ASJP transcription
│   └── README.md
└── metadata/                          ← offline reference pages
    ├── glottolog_weno1238.html
    ├── joshuaproject_wobe.html
    ├── wikipedia_fr_wobe.html
    ├── wikipedia_en_wobe.html
    └── grn_wobe.html
```

---

## Primary parallel corpus (how to load)

### JSONL fields

```json
{
  "book": "JHN",
  "chapter": 3,
  "verse": 16,
  "ref": "JHN 3:16",
  "french": "Car Dieu a tant aimé le monde…",
  "wobe": "Dɛ‑nʋɛ' Kea 'a nyiɔ 'ɔ ‑tɩ' kmaa'…",
  "source_wobe": "eBible WOBWBT © 2010 Wycliffe…",
  "source_french": "eBible fraLSG Louis Segond 1910 (Public Domain)",
  "license_note": "Wobé text is copyrighted…"
}
```

### Python

```python
import json
from pathlib import Path

pairs = [
    json.loads(line)
    for line in Path("data/aligned/wobe_french_nt_parallel.jsonl").read_text().splitlines()
    if line.strip()
]
print(len(pairs), pairs[0]["ref"])
```

### Hugging Face datasets (local)

```python
from datasets import load_dataset
ds = load_dataset("json", data_files={
    "train": "data/aligned/wobe_french_nt_train.jsonl",
    "validation": "data/aligned/wobe_french_nt_dev.jsonl",
    "test": "data/aligned/wobe_french_nt_test.jsonl",
})
```

**Do not push Wobé text to the public Hub without a license.**

---

## What was successfully obtained

### 1. Full digital New Testament in Wobé (2010)

| Format | Status | Path |
|--------|--------|------|
| USFM | ✅ | `wobe_bible/usfm/` (30 books, MAT–REV) |
| USFX XML | ✅ | `wobe_bible/usfx/` |
| HTML (chapter) | ✅ | `wobe_bible/html/` (317 files) |
| Read-aloud plain text | ✅ | `wobe_bible/readaloud/` |
| EPUB | ✅ | `wobe_bible/wob.epub` |
| Sword module | ✅ | `wobe_bible/wob2010eb.zip` |

Source: https://ebible.org/find/details.php?id=wobwbt

### 2. Verse-aligned French (Louis Segond 1910)

Public domain. Full Bible USFM available; NT verses aligned to Wobé.

### 3. Verse-aligned English (World English Bible)

Public-domain / open package for FR–WOB–EN triples.

### 4. GRN oral materials (~2 hours)

18 MP3 tracks, “Words of Life” program 01151:

1. From Creation to Christ  
2. The Story of Jesus  
3. The Heart of Man  
… through …  
18. Spread the Good News  

No automatic transcripts included (need human transcription for ASR/MT).

### 5. Related-language Bibles (transfer candidates)

Nyabwa, Tepo Krumen, Plapo Krumen, Kouya, Dan (2), Adioukrou, Anyin, Yaouré — HTML + USFM zips.

### 6. Newly verified linguistic resources (not yet MT training data)

- **ASJP Wobé wordlist:** 30 open CC BY 4.0 concept-word entries, in ASJP
  phonological transcription. See `lexicons/README.md`.
- **Egner 1988 conversation study:** 267 scanned pages, a French + Wobé study
  of authentic conversation. The host states Creative Commons use, but the
  exact sublicense still needs verification before extraction/training.
- **Egner 1989 grammar:** 254 scanned pages. Same license-verification gate.

### 7. Reference metadata

Glottolog, Joshua Project, Wikipedia FR/EN, GRN page (partial), SIL archive page for Egner 1989.

---

## What could NOT be fully obtained automatically

| Resource | Status | Action |
|----------|--------|--------|
| **Exact reuse terms for Egner PDFs** | Host says Creative Commons but exact license/sublicense not captured | Verify rights before any example extraction or model training |
| **Syllabaire Wobe** (Gfeller & Hofer, EDICEF, 1982) | 107-page literacy book located in CERDOTOLA catalogue, not digitized/downloaded | Locate a copy; request publisher/rightsholder permission before scanning or OCR |
| **Wè (parler wobé)** (Bozon, Hofer & Link, EDILIS, 2001) | Wobé-language printed book located in WorldCat, co-authored with Paul Guei Bozon and SIL/ILA attribution | Locate a copy; request EDILIS/rightsholder permission before digitization |
| **Guéré / Wè Southern Bible** (`gxx`) | Not on free eBible redistribution | Apps / Scripture Earth; license separately |
| **Old Testament Wobé** | Incomplete / not in eBible package | Contact Wycliffe / local team |
| **Faith Comes By Hearing full drama NT audio** | Streaming only in practice | bible.is / FCBH apps; license for offline dump |
| **Jesus Film We Northern** | Stream online | jesusfilm.org |
| **SIL literacy primers** (print archives) | Not bulk-downloadable | Request via SIL CI |
| **Public news/chat parallel data** | **Does not exist** | Must collect with speakers |
| **Hugging Face Wobé datasets** | None found | N/A |

Direct PDF URL (manual browser download):  
`https://www.sil.org/system/files/reapdata/13/40/45/134045664734789791923891915214905627350/Inge_Egner_Precis_de_Grammaire_Wobe_1989.pdf`

---

## Alignment method (reproducible)

1. Download Wobé USFM + French LSG USFM from eBible.  
2. Parse `\c` / `\v` markers into `(book, chapter, verse) → text`.  
3. Strip USFM markup / Strong’s tags from French.  
4. Inner-join on verse keys for NT books only.  
5. Result: **7,927** aligned pairs (1 Wobé verse missing French counterpart).

Rebuild script logic is documented in the research session; re-run from USFM if needed.

---

## Domain coverage warning

| Domain | Status |
|--------|--------|
| Biblical / religious | ✅ Strong (~8k verses) |
| News | ❌ None |
| Conversational / SMS | ❌ None |
| Health / agriculture / admin | ❌ None |
| Literary / folklore | ❌ None in digital form |

Bible-only models will **not** generalize to modern general-domain French (see MAFAND-MT domain-shift results). Plan to collect 10k+ non-Bible pairs next.

---

## Recommended immediate use

1. **Baseline model:** Fine-tune NLLB-200-distilled on `wobe_french_nt_train.jsonl` **only after** obtaining Wycliffe research permission.  
2. **Evaluation:** Hold out `wobe_french_nt_test.jsonl` forever.  
3. **Grammar:** Manually download Egner 1989; extract example sentences for lexicon.  
4. **Speech pilot:** Transcribe 1–2 GRN tracks as ASR seed.  
5. **Do not** publish the Wobé side of this corpus publicly yet.

---

## External URLs (live)

| Resource | URL |
|----------|-----|
| eBible Wobé | https://ebible.org/find/details.php?id=wobwbt |
| eBible mobile HTML | https://ebible.org/wob/ |
| YouVersion | https://www.bible.com/languages/wob |
| Bible.is | https://live.bible.is/bible/WOBWBT |
| Scripture Earth | https://www.scriptureearth.org/00eng.php?iso=wob |
| GRN | https://globalrecordings.net/en/language/wob |
| Jesus Film | https://www.jesusfilm.org/watch/jesus.html/we-northern.html |
| Glottolog | https://glottolog.org/resource/languoid/id/weno1238 |
| SIL Egner 1989 | https://www.sil.org/resources/archives/3367 |
| Play Store app | https://play.google.com/store/apps/details?id=org.ipsapps.cotedivoire.wob.we.nord.wobe.bible |

---

## File count summary

| Area | Approx. size |
|------|--------------|
| Aligned corpora | 24 MB |
| Audio (GRN) | 96 MB |
| French Bibles | 50 MB |
| Related languages | 18 MB |
| Wobé raw Bible | 14 MB |
| English Bible | 2.8 MB |
| Metadata + SIL page | <1 MB |
| **Total** | **~206 MB** |
