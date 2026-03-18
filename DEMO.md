# Neural Dashboard — POC XAI (Demo Runbook)

Objectif: présenter une démo “enterprise‑ready” d’explicabilité **mécanistique** basée sur les papiers *Transformer Circuits* (attribution graphs + interventions + validation).

Cette démo est volontairement cadrée sur:
- un **crux token** (logit cible, next‑token),
- une **hypothèse** (les features/edges les plus influents),
- une **validation causale** (interventions → changement de distribution + changement d’activations).

## Pré‑requis
- Python 3.10+ (recommandé: 3.12)
- GPU conseillé (sinon c’est possible mais lent)
- Accès HuggingFace (téléchargement modèle + transcoders au premier run)

## Lancer l’app

Windows (PowerShell):
```powershell
.\run.ps1
```

Unix / Git Bash:
```bash
./run.sh
```

L’app Streamlit s’ouvre ensuite sur `http://localhost:8501`.

## Structure d’une démo (script)

1) **Choisir un prompt**
- Sidebar → `Démos` (ou prompt custom)

2) **Analyze circuits**
- Sidebar → `Analyze circuits`
- Résultat attendu:
  - top logits (next token) visibles
  - features classées par **influence sur le logit cible**
  - edges prunées (poids élevés) pour raconter une “histoire causale”

3) **Choisir le crux token**
- Colonne gauche → `Target logit (crux token)`
- Choisir un token stable et “sémantique” (ex: `"Paris"`, `"grand"`, `"95"`)

4) **Intervenir**
- Colonne droite → `Intervention mode` (recommandé: *Add delta* ou *Ablate*)
- Cocher:
  - `Freeze attention` (recommandé)
  - (optionnel) `Constrain layers` si vous voulez un mode “direct effects style”

4bis) **Donner du sens à une feature (quick win)**
- Colonne droite → `Feature inspection (semantic hints)`
- `Compute direct logit effects` montre les tokens que la feature “pousse” / “supprime” quand on la booste (`Δ`).
  - Note: sur des features très “early layer”, l’effet peut être faible (influence indirecte). Dans ce cas, préférez `Activation validation`.

5) **Valider**
- `Run validation`: compare `Before` vs `After` (top‑k, entropie, margin)
  - regarder aussi **Top surface groups** (fusionne les variantes `"Austin"`, `" Austin"`, `"austin"`)
- (optionnel) `Enable sweep`: montre `p(token)` en fonction de la force `alpha`
- `Run activation validation`: montre les deltas d’activations sur des features choisies (intervened / top influence / kept)

6) **Exporter**
- `Download demo report (Markdown)` pour une pièce jointe prête à mettre dans un deck/notion/email
- `Download … JSON` si vous voulez intégrer dans un pipeline interne

## Démos recommandées (3–5 minutes)

### 1) Capitale (France)
- Prompt: `La capitale de la France est`
- Crux token: `" Paris"` (ou le token top‑1)
- Story:
  - features de “capitale / pays” → token de capitale
  - intervention: ablate top feature(s) → la proba de `"Paris"` baisse, alternatives montent
  - activation validation: les features downstream “say Paris” (ou équivalent) bougent

### 1bis) Swap “state → capital” (Dallas vs Oakland) — démo “wow”
Objectif: montrer une **édition** du mécanisme (swap d’un état) plutôt qu’une simple baisse de confiance.

1. Choisir la démo: `Multi-step (Oakland→California→Sacramento)`
2. `Analyze circuits`, puis dans la table **Top features**, repérer quelques features fortement liées à `California` / `Sacramento` (souvent sur le token `Oakland` et/ou dans les layers tardifs).
3. Copier leurs références et activer **Manual interventions**:
   - format: `LAYER POS FEATURE_IDX = VALUE`
4. Revenir sur `Multi-step (Dallas→Texas→Austin)`
5. Faire une intervention combinée:
   - **ablate** des features Texas (ou top-influence features sur `Dallas`)
   - **inject** (set value) des features California (prises du prompt Oakland)
6. `Run validation`:
   - attendu: baisse du groupe `"austin"` et montée de `"sacramento"` (ou au moins d’alternatives “California-capital”)
7. `Enable sweep` si besoin pour trouver un alpha où le flip se produit proprement.

### 2) Antonymes (FR)
- Prompt: `Le contraire de "petit" est "`
- Crux token: `"grand"`
- Story:
  - features “antonym / opposite” + features “petit” → output “grand”
  - intervention: ablate feature(s) d’opération → distribution se déplace

### 3) Addition
- Prompt: `calc: 36+59=`
- Crux token: `"95"`
- Story:
  - features d’arithmétique → output `"95"`
  - intervention: ablate une feature top → distribution se dégrade (montrer top‑k)

## Points à dire (cadre entreprise)
- On ne vend pas “la vérité absolue” du modèle: on vend une **méthode de preuve** (hypothèse → intervention → effet).
- On évite les claims “introspection” (au sens Lindsey 2025) sauf si protocole dédié.
- On fournit des artefacts: `report.md` + JSON pour audit interne / reproductibilité.

## Environnement (Graphviz)
- Le graphe dans l’UI nécessite le package Python `graphviz` + les binaires Graphviz installés sur la machine.
- Sans Graphviz, l’app fonctionne quand même (la table des edges reste visible).

## Limitations (à assumer explicitement)
- Attribution par **token** (next‑token): pour des raisonnements longs, il faut itérer sur plusieurs tokens / étapes.
- Les résultats dépendent du modèle + transcoders disponibles (ici `google/gemma-2-2b-it` + `gemma` transcoders).
- Les interventions sont sensibles au choix de features et au contexte; la démo inclut un sweep `alpha` pour le montrer proprement.
