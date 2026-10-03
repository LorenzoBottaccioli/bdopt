# Piano e stato di avanzamento – `bdopt`, alternativa a BESOS

> **Stato (implementato e verificato dove indicato).** Il piano sotto è quello originale; questa sezione dice
> cosa è stato fatto, cosa è stato fatto in modo diverso e cosa NON è stato verificato.

| Fase | Stato | Note |
|---|---|---|
| 0 Setup | fatto | `pyproject.toml`, ruff, pytest, CI GitHub Actions (YAML validato, **non eseguito** su GitHub), `tools/install_eplus.py` (testato su 8.7 `.sh` e 8.8 `.tar.gz`) |
| 1 Layer EnergyPlus multi-versione | fatto | testato su **8.7, 9.0, 9.6, 23.1, 24.2, 25.2** (Linux). Matrice completa in `tests/test_eplus_versions.py` |
| 2 Sampling e adaptive | fatto | algoritmi **nativi** (numpy/scipy); ponte harlow opzionale e testato |
| 3 Surrogati e ottimizzazione | fatto | GP, MLP sklearn, TorchMLP; NSGA-II/III (pymoo) con variabili miste |
| 4 Piattaforma | fatto, **non pubblicato** | notebook migrati ed eseguiti con EnergyPlus 9.6; Dockerfile/compose **non costruiti** (nessun daemon Docker qui). Il push su `andresorehida1/Smart_Builing_Educational_Platform_Dockers` è stato rifiutato (403: app GitHub non installata): le modifiche sono in `platform-migration/` come patch (`git am --keep-cr`) |
| 5 Rilascio | parziale | README e docs scritti; **nessuna pubblicazione su PyPI**; il pacchetto si installa da git |

## Differenze rispetto al piano

* **harlow non è il motore di default.** `harlow 0.1.0` importa TensorFlow/TensorFlow-Probability a livello di
  modulo, in contrasto con la scelta "solo PyTorch", e oggi un `pip install harlow` produce un import rotto
  (serve `tf_keras`, non dichiarato). Gli algoritmi sono quindi riscritti in `bdopt.adaptive` (FLOLA/LOLA-Voronoi,
  Voronoi, varianza GP); `HarlowAdaptiveSampler` è un ponte opzionale che esegue harlow vero.
* **Lo schema SQL è uguale in tutte le versioni 8.7–25.2** (verificato): non servono rami per versione, a differenza
  di quanto ipotizzato nel piano. Cambiano invece nomi di campi/classi/meter (tabella `aliases.py`, ricavata dal
  confronto reale degli IDD).
* **Parser IDF/IDD propri invece di eppy**: niente IDD globale per processo, quindi nessun conflitto tra versioni.
* **`PyEnergyPlusRunner` non implementato** (opzionale nel piano, nessun vantaggio rispetto al subprocess).
  `CommandRunner` (prefisso di comando, es. `docker exec`) sì.
* **pymoo con variabili codificate nel cubo unitario e "snap"** dei discreti, invece di `MixedVariableGA`.
* **`study.json` iniettato dal backend: non fatto.** `inject_data.py` continua a inserire una cella di codice (ora con
  `load_idf`), modifica minima; `DesignStudy.from_config` è disponibile se il backend vorrà passare a una configurazione.
* **Immagine Docker dedicata al surrogate** (`conda-environment/surrogate`, Python 3.11) invece di modificare l'ambiente
  condiviso a Python 3.9, perché `bdopt` richiede Python >= 3.10 e non potevo costruire/testare l'immagine condivisa.
* Il GP ha richiesto un fix di robustezza (vedi `docs/ADAPTIVE.md`): senza, alcuni disegni LHS davano
  errori ~1.0 per un ottimo locale della verosimiglianza.

## Risultati misurati

* Uguale a BESOS 2.2.3 su 4 punti reali (Office, EnergyPlus 9.6): differenza relativa massima 4e-13.
  Tempo in quella misura: bdopt 33 s, BESOS 15 min 25 s (singola misura, causa non indagata).
* Adaptive vs LHS a parità di punti di training: −37% di errore su una risposta con caratteristica **localizzata**,
  nessun vantaggio su risposta liscia o su soglia obliqua (tabella in `README.md`).
* Aggiornamento automatico 9.6 → 25.2 del modello Office: ~230 s una tantum; risultati diversi di −2.3% / +4.9% / −5.8%
  rispetto a 9.6 (causa non isolata).

## Non verificato

CI su GitHub, esecuzione su Windows/macOS, build delle immagini Docker, pubblicazione su PyPI, modifica del frontend
(il bundle `frontend/dist` non cita BESOS, ma non è stato rigenerato né provato).

---

# Piano originale

Nome: **`bdopt`** (Building Design Optimization), libero su PyPI. Licenza: **MIT**.

## 0. Requisiti fissati

| Requisito | Scelta |
|---|---|
| Versioni EnergyPlus | **tutte dalla 8.7 in poi** (8.7 → 25.x), con rilevamento automatico della versione |
| Machine learning | **solo PyTorch** (+ scikit-learn); TensorFlow/Keras vengono eliminati |
| Packaging | **libreria Python indipendente** (`pip install bdopt`), che gira su Linux/Windows/macOS con un'installazione locale di EnergyPlus. Le immagini Docker sono un artefatto **separato e opzionale** del repo, non una dipendenza |
| Ottimizzatore | **solo pymoo** (variabili miste, NSGA-II/III); Platypus eliminato |
| Nome / licenza | `bdopt`, **MIT** (come BESOS ed eppy: permissiva, adatta all'uso accademico e didattico) |

## 1. Perché

Analisi di `Smart_Builing_Educational_Platform_Dockers` (notebook in `backend/notebook_templates`,
`Surrogate modeling examples/Test 1..7`, `conda-environment/`):

**Cosa usiamo davvero di BESOS** (circa il 10% della libreria):

| API BESOS | Uso nella piattaforma |
|---|---|
| `eppy_funcs.get_building` | carica l'IDF (iniettato dal backend in `inject_data.py`) |
| `FieldSelector(class_name, object_name, field_name)` | anche con `object_name='*'` |
| `RangeParameter`, `CategoryParameter`, `Parameter` | continui + categorici (tipi di vetro, Test 6) |
| `EPProblem(parameters, objectives)` | obiettivi = meter (`Electricity:Facility`, `DistrictHeating:Facility`, `DistrictCooling:Facility`) |
| `sampling.dist_sampler(sampling.lhs, problem, n)` | LHS |
| `EvaluatorEP(...).df_apply(df)` | lancia EnergyPlus 9.6 su ogni riga |
| `EvaluatorGeneric(func, problem).df_apply(df)` | valuta il surrogato |

Importato ma **non usato**: `NSGAII` di BESOS (i notebook usano Platypus direttamente), `GenericSelector`,
`expand_plist`, `wwr`. Mai usato: PyEHub/EnergyHub, `EvaluatorEH`, rbfopt, editing della geometria, ecc.

**Cosa manca in BESOS: il design optimization con adaptive sampling (harlow).** Oggi negli
`Harlow_Example_*.ipynb` l'integrazione si fa a mano, ed è fragile:

1. i bound del dominio vanno **ricopiati a mano** e scalati a mano, nello stesso ordine dei parametri BESOS;
2. la `target_f` va scritta a mano: scala/de-scala, reshape, `DataFrame`, `df_apply`, di nuovo scala;
3. **monkey-patch** di `surrogate_model.update` per forzare `y` 2D;
4. le **variabili categoriche** si gestiscono con un `if glazing_value < 1.5 ...` scritto a mano, più `replace(win, i+1)` sui DataFrame;
5. il problema Platypus si ricostruisce a mano e si rompe con i categorici (`'CategoryParameter' object has no attribute 'min'` in Test 6);
6. data leakage negli scaler (`scaler.fit_transform(test_in)` invece di `transform`);
7. nessuna cache o persistenza strutturata delle run EnergyPlus (si usano `.pkl` salvati a mano);
8. legato a una sola versione di EnergyPlus (9.6) e a TensorFlow (dipendenza pesante).

## 2. Principi

- **Solo il necessario**: EnergyPlus + parametri + sampling (statico e adattivo) + surrogati + ottimizzazione multi-obiettivo.
- **Indipendente dall'ambiente**: nessuna assunzione su Docker, path o OS. EnergyPlus si trova da sé (vedi §3.1) oppure si indica esplicitamente.
- **Multi-versione EnergyPlus by design**: tutto ciò che dipende dalla versione (IDD, nomi di campi e meter, schema SQL, CLI) passa da un unico modulo `eplus/`.
- **Spazio normalizzato unico**: ogni `Problem` sa codificare e decodificare i parametri in `[0,1]^d` (i categorici con indici arrotondati). Harlow, i surrogati e gli ottimizzatori lavorano tutti in questo spazio, così niente bound o scaler a mano.
- **DataFrame in ingresso e in uscita**, con nomi di colonna = nomi dei parametri e degli obiettivi.
- **Strato di compatibilità** `bdopt.compat.besos`: i notebook esistenti migrano cambiando solo gli import.
- **Configurabile da JSON/YAML** (`DesignStudy.from_config`), così il backend genera uno *study config* invece di iniettare codice.
- Python ≥ 3.10. Core leggero (numpy, pandas, scipy, eppy, pymoo). Platypus non è una dipendenza: `optimize.py` espone un'interfaccia `Optimizer` minima, così un altro backend si può aggiungere in futuro senza toccare il resto. Gli extra sono opzionali: `bdopt[harlow]`, `bdopt[torch]`, `bdopt[viz]`.

## 3. Architettura

```
bdopt/
  eplus/                # TUTTO ciò che dipende dalla versione di EnergyPlus
    install.py          # trova le installazioni: arg esplicito > env ENERGYPLUS_DIR/EPLUS_PATH
                        # > PATH > percorsi standard per OS (C:\EnergyPlusV*, /usr/local/EnergyPlus-*,
                        # /Applications/EnergyPlus-*); legge la versione (`energyplus --version`/idd)
    version.py          # Version("8.7.0"), parsing dall'oggetto `Version` dell'IDF, confronti
    idd.py              # IDD dell'installazione corrispondente; gestione IDD globale di eppy
    transition.py       # upgrade IDF -> versione del motore (IDFVersionUpdater/Transition-V*
                        # inclusi in ogni installazione), opzionale e con cache
    aliases.py          # tabella dei nomi che cambiano tra versioni (campi, classi, meter)
    sql.py              # lettore eplusout.sql che si adatta allo schema della versione
  model.py              # load_idf(path, eplus=...) -> Building (eppy), copia, salvataggio
  parameters.py         # FieldSelector, GenericSelector(callable), RangeParameter,
                        # CategoryParameter, Parameter; encode/decode verso [0,1]
  problem.py            # Problem(parameters, objectives, constraints?)
                        #   .names(), .bounds(), .encode(df), .decode(X), .to_pymoo()
  objectives.py         # MeterReader, VariableReader, unità J->kWh, aggregazioni, callable custom
  simulation/
    runner.py           # Runner astratto
                        #   LocalRunner (default): subprocess `energyplus -w -d -r ...`, una cartella
                        #     per run, parallelismo (ProcessPool), timeout, gestione errori
                        #   PyEnergyPlusRunner (opzionale, E+ >= 9.3): API Python in-process
                        #   CommandRunner (opzionale): prefisso di comando configurabile, es.
                        #     `docker exec energyplus ...`; così la piattaforma usa il suo container
                        #     senza che la libreria dipenda da Docker
    cache.py            # cache su disco: hash(IDF + EPW + versione E+ + parametri) -> risultati
  evaluator.py          # EnergyPlusEvaluator, FunctionEvaluator: .evaluate(df) -> df
  sampling.py           # lhs, sobol, random, full factorial (scipy.stats.qmc)
  adaptive/
    harlow.py           # HarlowAdaptiveSampler: wrapper su FuzzyLolaVoronoi, LolaVoronoi,
                        # ProbabilisticSampler, CVVoronoi... gestisce target_f, scaling,
                        # categorici, shape 2D, stopping criteria, storico iterazioni
    native.py           # fallback senza harlow (max-varianza GP) per test e robustezza
  surrogate/
    sklearn.py          # GP (selezione del kernel), MLP sklearn: Pipeline con scaler interno
    torch.py            # MLP PyTorch (sostituisce il modello Keras dei notebook), early stopping,
                        # salvataggio/caricamento; opzionale GP in GPyTorch
    metrics.py          # RMSE, RMSE%, CV(RMSE), NMBE (ASHRAE G14), R2
  optimize.py           # NSGA-II/NSGA-III su pymoo (variabili miste) sul surrogato o su EP;
                        # pareto_df, selezione (distanza normalizzata dal punto ideale)
  study.py              # DesignStudy: pipeline end-to-end da config
  viz.py                # pairplot, heatmap correlazioni, parallel coords, pareto 3D (extra)
  compat/besos.py       # alias: FieldSelector, RangeParameter, EPProblem, EvaluatorEP,
                        # EvaluatorGeneric, sampling.dist_sampler/lhs, eppy_funcs.get_building
docker/                 # OPZIONALE, fuori dal pacchetto Python
  Dockerfile            # ARG EPLUS_VERSION -> immagini bdopt:eplus-8.7 ... bdopt:eplus-25.x
```

### 3.1 Supporto EnergyPlus 8.7+

Cosa cambia tra le versioni e come lo gestiamo:

| Aspetto | Gestione |
|---|---|
| **Dove sta E+ / quale versione** | `eplus.install.find()` restituisce tutte le installazioni trovate; `EnergyPlus(version="9.6")` oppure `EnergyPlus(path=...)`; di default sceglie quella che corrisponde alla versione dell'IDF |
| **IDD** | sempre quello dell'installazione usata. eppy ha un IDD globale per processo: lo resettiamo in modo controllato (`IDF.resetidd`) e le run parallele in processi separati evitano conflitti |
| **IDF di versione diversa dal motore** | errore chiaro, oppure `auto_upgrade=True` che usa i Transition dell'installazione (solo upgrade, non downgrade) |
| **CLI** | `energyplus -w EPW -d OUT -r?` è stabile dalla 8.3, quindi copre tutte le versioni ≥ 8.7. Il `ExpandObjects` si attiva quando l'IDF contiene HVACTemplate |
| **Output** | la libreria aggiunge `Output:SQLite` e i `Output:Meter`/`Output:Variable` necessari. Il lettore SQL si adatta allo schema: nelle versioni vecchie i meter stanno in tabelle separate (`ReportMeterData*`), in quelle recenti sono unificati in `ReportData*`. Fallback su ESO/CSV |
| **Nomi rinominati** | tabella `aliases.py` con test. Esempi da verificare caso per caso: `DistrictHeating:Facility` → `DistrictHeatingWater:Facility` (23.1+), `Watts per Zone Floor Area` → `Watts per Floor Area` nelle versioni recenti. Un `FieldSelector`/`MeterReader` accetta il nome "vecchio" e lo risolve per la versione attiva, con un warning |
| **Python API** | `PyEnergyPlusRunner` opzionale, solo per E+ ≥ 9.3; il default resta il subprocess, che funziona ovunque |

### 3.2 API obiettivo (quello che diventa un notebook)

```python
from bdopt import EnergyPlus, load_idf, FieldSelector, RangeParameter, CategoryParameter, Parameter, Problem
from bdopt.evaluator import EnergyPlusEvaluator
from bdopt.adaptive import HarlowAdaptiveSampler
from bdopt.surrogate import TorchMLP, GPSurrogate
from bdopt.optimize import nsga2

ep = EnergyPlus()                                   # autodetect; oppure EnergyPlus(version="23.2")
building = load_idf("Office_IDF.idf", eplus=ep)
params = [
    Parameter(FieldSelector("Material", "_InsulationEX_.01", "Thickness"), RangeParameter(0.1, 0.5), name="Wall Insulation"),
    Parameter(FieldSelector("FenestrationSurface:Detailed", "*", "Construction Name"),
              CategoryParameter(["single_glazing", "double_glazing", "triple_glazing"]), name="Windows types"),
    ...
]
problem = Problem(params, ["Electricity:Facility", "DistrictHeating:Facility", "DistrictCooling:Facility"])
ev = EnergyPlusEvaluator(problem, building, epw="Larnaca.epw", eplus=ep, out_dir="runs", processes=4, cache=True)

sampler = HarlowAdaptiveSampler(problem, ev, method="FuzzyLolaVoronoi")
data = sampler.run(n_initial=20, n_test=5, max_iter=40, batch=1, tol=0.01)   # DataFrame in unità reali

sur = TorchMLP(hidden=[32, 64, 32]).fit(data, problem)   # oppure GPSurrogate(); report metriche
pareto = nsga2(problem, sur, pop_size=200, n_gen=200)    # DataFrame parametri + obiettivi
best = pareto.closest_to_ideal()
```

## 4. Fasi e milestone

**Fase 0 – Setup (1 settimana)**
- struttura del repo (`pyproject.toml`, src layout, ruff, mypy, pytest, pre-commit), licenza
- CI GitHub Actions: lint + unit test su Linux/Windows/macOS, Python 3.10–3.13 (senza E+)
- script `tools/install_eplus.py` che scarica una versione specifica dalle release NREL (usato in CI e dagli utenti); cache degli installer in CI
- dataset di riferimento: IDF/EPW di Test 1..7 + risultati BESOS salvati come golden (E+ 9.6)
- verifica delle dipendenze di harlow: quali surrogati richiedono TF; si usano solo quelli torch/sklearn/gpytorch

**Fase 1 – Layer EnergyPlus multi-versione (2–3 settimane)**
- `eplus/` completo: discovery, versioni, IDD, transition, aliases, lettore SQL che si adatta allo schema
- `model`, `parameters` (selettori con `*`, categorici, GenericSelector), `problem` (encode/decode)
- `LocalRunner` + `EnergyPlusEvaluator` con parallelismo, cache, gestione errori (riga fallita a NaN più log)
- **matrice di integrazione E+**: 8.7, 8.9, 9.0, 9.3, 9.6, 22.2, 23.1, 24.2, ultima disponibile. Sulle PR girano 3 versioni (8.7, 9.6, ultima), la matrice completa ogni notte
- criteri di uscita: Test 1 su 9.6 uguale a BESOS (tolleranza < 0.1%); stesso study eseguibile su tutte le versioni della matrice con risultati plausibili (regressione per versione salvata)

**Fase 2 – Sampling e adaptive (2 settimane)**
- `sampling` (LHS/Sobol in spazio normalizzato, decodifica corretta dei categorici)
- `HarlowAdaptiveSampler`: bound automatici, target function, fix della shape, storico, salvataggio incrementale (riprende dopo un crash), pin di versione di harlow + fallback `native`
- criterio di uscita: Harlow_Example_1 e _6 (categorici) riprodotti senza codice di collegamento

**Fase 3 – Surrogati (torch) e ottimizzazione (2 settimane)**
- `surrogate`: GP e MLP sklearn, MLP PyTorch (equivalente al modello Keras attuale), split train/test corretto, metriche
- `optimize` con pymoo (variabili miste: niente errore `.min` sui categorici), utilità per il fronte di Pareto

**Fase 4 – Docker (opzionale) e integrazione nella piattaforma (1–2 settimane)**
- `docker/Dockerfile` parametrico su `EPLUS_VERSION`, immagini pubblicate su GHCR: è comodo, ma la libreria non ne ha bisogno
- `compat.besos` + migrazione di `Surrogate_template.ipynb` e degli esempi Test 1..7 (Keras → `TorchMLP`)
- `conda_environment.yml` della piattaforma: via `besos`, `tensorflow-probability[tf]`; dentro `bdopt[harlow,torch,viz]`; Python 3.9 → ≥ 3.10
- backend: `inject_data.py` scrive uno `study.json` (IDF, EPW, versione E+, parametri scelti dal frontend tramite `/get_surrogate_objects`) letto da `DesignStudy.from_config`; `CommandRunner` se si vuole continuare a eseguire E+ nel container `energyplus`

**Fase 5 – Rilascio (1 settimana)**
- docs (mkdocs): installazione di E+, tabella di compatibilità delle versioni, guida alla migrazione da BESOS, esempi
- release 0.1 su PyPI + immagini Docker opzionali

Durata stimata totale: **9–11 settimane** per una persona (il supporto multi-versione aggiunge circa 2 settimane).

## 5. Test e qualità

- unit test senza E+: encode/decode (proprietà `decode(encode(x)) == x`, categorici inclusi), aliases, parser della versione, lettore SQL su file `.sql` di fixture generati da ogni versione
- integrazione E+ in CI sulla matrice di versioni, con un IDF piccolo e una run breve (RunPeriod di pochi giorni)
- regressione numerica contro i golden BESOS (9.6) e golden propri per le altre versioni
- adaptive sampling testato su funzioni analitiche (Branin, Hartmann) senza E+
- test di installazione "pulita" (`pip install bdopt` in un venv, senza Docker) su tutti e 3 gli OS

## 6. Rischi

| Rischio | Mitigazione |
|---|---|
| 8.7 è vecchia: binari solo per OS datati (es. Ubuntu 16.04/18.04 per Linux) | test 8.x nella CI Linux su un container dell'epoca, oppure librerie di compatibilità; documentare i limiti per OS |
| differenze di schema/nome tra versioni non coperte | `aliases.py` + fixture SQL per ogni versione; errore esplicito con suggerimento invece di NaN silenziosi |
| eppy con IDD globale | un IDD per processo, run in sottoprocessi; valutare un parser IDF interno se diventa un limite |
| harlow poco manutenuto / dipendenze TF | wrapper isolato, versione pinnata, solo surrogati torch/sklearn, fallback `native` |
| differenze numeriche vs BESOS | golden test fase 1 |
| durata della matrice E+ in CI | matrice completa solo notturna, cache degli installer |

## 7. Decisioni prese

1. **Nome `bdopt`, licenza MIT.** Il nome è libero su PyPI. MIT è la stessa licenza di BESOS ed eppy, compatibile con EnergyPlus (BSD-3) e semplice per l'uso accademico.
2. **Solo pymoo, niente Platypus.** pymoo gestisce nativamente le variabili miste (risolve il problema dei categorici di Test 6), offre NSGA-II/NSGA-III ed è manutenuto attivamente. Una dipendenza in meno; se servirà un altro ottimizzatore, si aggiunge dietro l'interfaccia `Optimizer`.
3. **Downgrade degli IDF** (es. un IDF 23.x su un motore 9.6): non supportato, errore esplicito.
