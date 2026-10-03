# Piano – libreria alternativa a BESOS per la Smart Building Educational Platform

Nome di lavoro: **`bdopt`** (Building Design Optimization). Il nome è provvisorio.

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
`expand_plist`, `wwr`. Mai usato: PyEHub/EnergyHub, `EvaluatorEH`, rbfopt, editing della geometria, ecc.,
che però si portano dietro dipendenze pesanti e vincoli di versione.

**Cosa manca in BESOS: il design optimization con adaptive sampling (harlow).** Oggi negli
`Harlow_Example_*.ipynb` l'integrazione si fa a mano, ed è fragile:

1. i bound del dominio vanno **ricopiati a mano** (`domain_lower_bound = [...]`), devono restare nello stesso ordine dei parametri BESOS e vanno scalati a mano;
2. la `target_f` va scritta a mano: scala/de-scala, reshape, `DataFrame`, `df_apply`, di nuovo scala;
3. **monkey-patch** di `surrogate_model.update` per forzare `y` 2D;
4. le **variabili categoriche** si gestiscono con un `if glazing_value < 1.5 ...` scritto a mano nella `target_f`, più `replace(win, i+1)` sui DataFrame;
5. il problema Platypus si ricostruisce a mano da `param.value_descriptor.min/max`, e si rompe con i categorici (`'CategoryParameter' object has no attribute 'min'` in Test 6);
6. data leakage negli scaler (`scaler.fit_transform(test_in)` invece di `transform`), con due fasi di scaling diverse prima e dopo harlow;
7. nessuna cache o persistenza strutturata delle run EnergyPlus (si usano `.pkl` salvati a mano).

Obiettivo: una libreria **piccola**, che copra il 100% di quello che la piattaforma usa e renda
l'adaptive sampling con harlow una sola chiamata.

## 2. Principi

- **Solo il necessario**: EnergyPlus + parametri + sampling (statico e adattivo) + surrogati + ottimizzazione multi-obiettivo.
- **Spazio normalizzato unico**: ogni `Problem` sa codificare e decodificare i parametri in `[0,1]^d` (i categorici con indici arrotondati). Harlow, i surrogati e gli ottimizzatori lavorano tutti in questo spazio, così niente bound o scaler a mano.
- **DataFrame in ingresso e in uscita**, con nomi di colonna = nomi dei parametri e degli obiettivi (come in BESOS, per non cambiare le abitudini didattiche).
- **Strato di compatibilità** `bdopt.compat.besos`: i notebook esistenti migrano cambiando solo gli import.
- **Configurabile da JSON/YAML**, così il backend genera uno *study config* dalle scelte del frontend invece di iniettare codice nelle celle.
- Python ≥ 3.10, dipendenze minime nel core; harlow, TF e la visualizzazione come extra opzionali.

## 3. Architettura

```
bdopt/
  model.py          # load_idf(path, idd=env IDD_PATH), copy, save (eppy)
  parameters.py     # FieldSelector, GenericSelector(callable), RangeParameter,
                    # CategoryParameter, Parameter; encode/decode verso [0,1]
  problem.py        # Problem(parameters, objectives, constraints?)
                    #   .names(), .bounds(), .encode(df), .decode(X), .to_pymoo()
  objectives.py     # MeterReader, VariableReader (da eplusout.sql), unità J->kWh,
                    # aggregazioni (sum, max, mean), obiettivi custom (callable)
  simulation/
    runner.py       # EnergyPlusRunner: subprocess, una cartella per run,
                    # parallelismo (ProcessPool), timeout, gestione errori
    cache.py        # cache su disco: hash(IDF + EPW + valori parametri) -> risultati
  evaluator.py      # EnergyPlusEvaluator, FunctionEvaluator: .evaluate(df) -> df
  sampling.py       # lhs, sobol, random, full factorial (scipy.stats.qmc)
  adaptive/
    harlow.py       # HarlowAdaptiveSampler: wrapper su FuzzyLolaVoronoi, LolaVoronoi,
                    # ProbabilisticSampler, CVVoronoi... gestisce target_f, scaling,
                    # categorici, shape 2D, stopping criteria, storico iterazioni
    native.py       # fallback senza harlow (max-varianza GP) per test e robustezza
  surrogate.py      # GP / MLP (sklearn Pipeline con scaler interno, niente leakage),
                    # Keras opzionale; fit, CV, metriche (RMSE, RMSE%, CV(RMSE), R2)
  optimize.py       # NSGA-II/NSGA-III su pymoo (variabili miste) sul surrogato o su EP;
                    # pareto_df, selezione della soluzione (distanza dal punto ideale normalizzata)
  study.py          # DesignStudy: pipeline end-to-end da config
  viz.py            # pairplot, heatmap correlazioni, parallel coords, pareto 3D (extra)
  compat/besos.py   # alias: FieldSelector, RangeParameter, EPProblem, EvaluatorEP,
                    # EvaluatorGeneric, sampling.dist_sampler/lhs, eppy_funcs.get_building
```

### API obiettivo (quello che diventa un notebook)

```python
from bdopt import load_idf, FieldSelector, RangeParameter, CategoryParameter, Parameter, Problem
from bdopt.evaluator import EnergyPlusEvaluator
from bdopt.adaptive import HarlowAdaptiveSampler
from bdopt.surrogate import GPSurrogate
from bdopt.optimize import nsga2

building = load_idf("Office_IDF.idf")
params = [
    Parameter(FieldSelector("Material", "_InsulationEX_.01", "Thickness"), RangeParameter(0.1, 0.5), name="Wall Insulation"),
    Parameter(FieldSelector("FenestrationSurface:Detailed", "*", "Construction Name"),
              CategoryParameter(["single_glazing", "double_glazing", "triple_glazing"]), name="Windows types"),
    ...
]
problem = Problem(params, ["Electricity:Facility", "DistrictHeating:Facility", "DistrictCooling:Facility"])
ev = EnergyPlusEvaluator(problem, building, epw="Larnaca.epw", out_dir="runs", processes=4, cache=True)

sampler = HarlowAdaptiveSampler(problem, ev, method="FuzzyLolaVoronoi", surrogate="BatchIndependentGP")
data = sampler.run(n_initial=20, n_test=5, max_iter=40, batch=1, tol=0.01)   # DataFrame in unità reali

sur = GPSurrogate().fit(data, problem)            # scaling interno, report metriche
pareto = nsga2(problem, sur, pop_size=200, n_gen=200)   # DataFrame parametri + obiettivi
best = pareto.closest_to_ideal()
```

Oggi le celle tra la 20 e la 30 dei notebook Harlow fanno a mano scaling, bound, `target_f` e patch;
con questa API diventano 2 righe.

## 4. Fasi e milestone

**Fase 0 – Setup (1 settimana)**
- struttura del repo (`pyproject.toml`, src layout, ruff, pytest, pre-commit), licenza, CI GitHub Actions
- immagine Docker di test con EnergyPlus 9.6.0 (uguale a quella della piattaforma)
- dataset di riferimento: IDF/EPW di Test 1..7 + risultati BESOS salvati (`.pkl`) come golden

**Fase 1 – Core EnergyPlus (2 settimane)**
- `model`, `parameters` (selettori con `*`, categorici, GenericSelector), `problem` (encode/decode)
- `EnergyPlusRunner` + lettura dei meter da SQLite, unità coerenti con BESOS (J)
- `EnergyPlusEvaluator` con parallelismo, cache e gestione degli errori (la riga fallita va a NaN più un log, invece di interrompere il batch)
- test: unit senza E+ (runner mockato) + integrazione con E+ 9.6
- criterio di uscita: Test 1 LHS 60 campioni = risultati BESOS (tolleranza < 0.1%)

**Fase 2 – Sampling e adaptive (2 settimane)**
- `sampling` (LHS/Sobol in spazio normalizzato, decodifica corretta dei categorici)
- `HarlowAdaptiveSampler`: costruzione automatica di bound, target function, fix della shape, storico, salvataggio incrementale (riprende dopo un crash)
- verifica della compatibilità di versione con harlow (pin e test su più versioni) + fallback `native`
- criterio di uscita: Harlow_Example_1 e _6 (categorici) riprodotti senza codice di colla

**Fase 3 – Surrogati e ottimizzazione (1–2 settimane)**
- `surrogate` (GP con selezione del kernel, MLP, Keras opzionale) con split train/test corretto
- `optimize` con pymoo (supporta variabili miste, quindi niente errore `.min` sui categorici); adapter Platypus opzionale
- utilità per il fronte di Pareto e la selezione della soluzione

**Fase 4 – Integrazione nella piattaforma (1–2 settimane)**
- `compat.besos` + migrazione di `Surrogate_template.ipynb` e degli esempi Test 1..7
- `conda_environment.yml`: rimuovere `besos`, aggiungere `bdopt[harlow,viz]`
- backend: `inject_data.py` scrive un `study.json` (IDF, EPW, parametri scelti dal frontend tramite `/get_surrogate_objects`) letto da `DesignStudy.from_config`, al posto di iniettare codice
- documentazione didattica: notebook "BESOS vs adaptive" rifatti con la nuova API

**Fase 5 – Rilascio**
- docs (mkdocs), esempi, release 0.1 su PyPI/GitHub, changelog

Durata stimata totale: 7–9 settimane per una persona.

## 5. Test e qualità

- unit test per encode/decode (proprietà: `decode(encode(x)) == x`, categorici inclusi)
- integrazione E+ in CI (container 9.6) su un IDF piccolo, una run breve (RunPeriod di pochi giorni)
- regressione numerica contro i golden BESOS di Test 1..7
- test dell'adaptive sampling con funzioni analitiche (es. Branin), senza E+

## 6. Rischi

| Rischio | Mitigazione |
|---|---|
| harlow poco manutenuto / API instabile | wrapper isolato in `adaptive/harlow.py`, versione pinnata, fallback `native` |
| differenze numeriche vs BESOS (unità, aggregazioni) | golden test fase 1 |
| supporto a versioni E+ ≠ 9.6 | IDD e binario configurabili; 9.6 unica versione supportata ufficialmente in 0.1 |
| peso di TensorFlow nell'immagine | extra opzionale `bdopt[keras]` |

## 7. Decisioni aperte

1. Nome definitivo della libreria e licenza (BESOS è MIT)
2. Ottimizzatore: **pymoo** (consigliato: variabili miste, NSGA-III, manutenuto) oppure restare su Platypus
3. Supporto EnergyPlus: solo 9.6 o anche 23.x/24.x (epJSON)
4. Tenere Keras/TF nei notebook didattici o sostituirlo con sklearn/PyTorch-lite
5. Parallelismo: solo processi locali o anche esecuzione nel container `energyplus` (`docker exec`, come fa oggi `backend/methods.py`)
