# License and rights notice for collected data

**Gathered:** 2026-08-11  
**Project purpose:** Research feasibility and corpus scaffolding for Wobé ↔ French MT.

This folder contains materials with **different licenses**. Do not assume free reuse for machine-learning training or redistribution.

---

## Wobé New Testament (primary parallel source)

| Field | Value |
|-------|--------|
| Title | Wè Northern: DBƐƐDƐ DEE‑ New Testament+ |
| ID | WOBWBT / wob / WOBNT |
| Copyright | © **2010 Wycliffe Bible Translators, Inc.** |
| Contributor | WBT Central Zone |
| License | **All rights reserved** |
| Source | https://ebible.org/find/details.php?id=wobwbt |
| Paths | `wobe_bible/`, `aligned/wobe_*` |

### What this means

1. **Reading / personal study:** Generally allowed via eBible/YouVersion apps.  
2. **Redistributing the Wobé text** (e.g. publishing a new website or dataset with full text): **requires permission**.  
3. **Training ML models** (research or commercial): **requires written permission** from the rights holder.  
4. **Sharing derived models** trained on this text: may create derivative-work issues; get legal clearance.

**Do not** publish the aligned Wobé text on Hugging Face or GitHub without Wycliffe approval.

---

## French Louis Segond 1910

| Field | Value |
|-------|--------|
| Title | Louis Segond 1910 |
| ID | fraLSG |
| License | **Public Domain** |
| Source | https://ebible.org/find/details.php?id=fraLSG |
| Paths | `french_bible/` |

Safe to use freely for alignment and training (French side only).

---

## English World English Bible (if present)

| Field | Value |
|-------|--------|
| ID | engwebp (or similar) |
| License | Public domain / very permissive (check package) |
| Paths | `english_bible/` |

---

## Global Recordings Network audio

| Field | Value |
|-------|--------|
| Paths | `audio_catalog/grn_wobe/` |
| Rights | GRN copyright; terms at globalrecordings.net |
| Use | Evangelism/education focus; confirm before ASR training or redistribution |

---

## SIL Egner grammar (1989)

| Field | Value |
|-------|--------|
| Title | Précis de grammaire wobé |
| Author | Inge Egner |
| Paths | `sil/Inge_Egner_Precis_de_Grammaire_Wobe_1989.pdf` |
| Source | https://www.anstoessefuerherzundkopf.net/fr/linguistique/pr%C3%A9cis-de-grammaire-wob%C3%A9 |
| Rights | Host states that it uses the text under a Creative Commons license; **exact license and authority to sublicense the original work are not yet verified** |

## Egner conversation study (1988)

| Field | Value |
|-------|-------|
| Title | *Analyse conversationnelle de l'échange réparateur en wobé: parler wɛɛ de Côte d'Ivoire* |
| Author | Inge Egner |
| Paths | `sil/Egner_Analyse_conversationnelle_wobe_1988.pdf` |
| Source | https://inspirationforheartandmind.net/index.php/fr/linguistique/analyse-conversationnelle-de-l%C3%A9change-r%C3%A9parateur-en-wob%C3%A9 |
| Content status | 267-page scanned PDF; French + Wobé catalogue record; needs human/OCR inspection |
| Rights | Host states that it uses the text under a Creative Commons license; **exact license and authority to sublicense the Peter Lang original are not yet verified** |

Do not extract or train on either Egner work until the exact license is captured
and reviewed. They are strong leads for manually curated example pairs.

## ASJP Wobé wordlist

| Field | Value |
|-------|-------|
| Path | `lexicons/asjp_wobe_wordlist.json` |
| Source | https://asjp.clld.org/languages/WOBE |
| License | **CC BY 4.0** (ASJP database) |
| Scope | 30-item wordlist in ASJP transcription |
| ML use | Open lexicon / terminology-evaluation seed, not sentence-level MT data |

## Literacy-book digitization target

| Field | Value |
|-------|-------|
| Title | *Syllabaire Wobe* |
| Authors | Elisabeth Gfeller; Verena Hofer |
| Publisher / year | EDICEF, 1982 |
| Size | 107 pages |
| Evidence | CERDOTOLA catalogue: https://cerdotola.center/pmb/opac_css/index.php?dcote=&id=27&lcote=0&location=1&lvl=section_see&main=&nb_per_page_custom=100&nbr_lignes=525&nc=0&page=4&plettreaut=&ssub=0 |
| Status | Catalogue record only; no local copy or license |

This is a strong candidate for a small, non-Bible educational corpus. It must
be obtained, scanned, and OCRed only after permission from the relevant rights
holder.

## Additional literacy-book digitization target

| Field | Value |
|-------|-------|
| Title | *Wè: (parler wobé)* |
| Authors | Paul Guei Bozon; Verena Hofer; Christa Link; SIL Côte d'Ivoire / Institut de Linguistique Appliquée attribution |
| Publisher / year | EDILIS, Abidjan, 2001 |
| Evidence | WorldCat: https://search.worldcat.org/fr/title/we-parler-wobe/oclc/68706898 |
| Status | Catalogue record only; no local copy, page count, or license |

This is likely a second literacy resource, and the named Wobé co-author makes
it a useful relationship and rights lead. It is not training data until a copy
and rights permission are obtained.

---

## Related-language Bibles (`related/`)

Each package has its own copyright (often Wycliffe or local Bible societies). Many are marked **restricted** on eBible (gray listings). Check each `details.php?id=…` page before any ML use.

---

## Metadata HTML scrapes (`metadata/`)

Downloaded public web pages for offline reference. Copyright remains with original sites. Not training data.

---

## Recommended next legal step

Email **Wycliffe Bible Translators** / Digital Bible Library requesting:

1. Research-use license for the Wobé 2010 NT text  
2. Permission to create and (optionally) release a verse-aligned FR–WOB research corpus  
3. Terms for non-commercial academic model training and publication of evaluation scores  

Until then, treat all Wobé text files as **restricted local working copies**.
