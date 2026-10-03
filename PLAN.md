# bdopt – piano e stato

Libreria Python indipendente per l'ottimizzazione del progetto di edifici con EnergyPlus (dalla 8.7 in poi):
studi parametrici, **adaptive sampling**, modelli surrogati (scikit-learn e PyTorch) e ottimizzazione
multi-obiettivo. Nome `bdopt`, licenza MIT.

## Scelte

| Tema | Scelta |
|---|---|
| Versioni EnergyPlus | tutte dalla 8.7 (testate 8.7, 9.0, 9.6, 23.1, 24.2, 25.2) |
| Machine learning | scikit-learn + PyTorch; nessun TensorFlow/Keras |
| Ottimizzazione | solo pymoo (NSGA-II / NSGA-III), variabili miste |
| Docker | non è una dipendenza; `docker/Dockerfile` è facoltativo |
| Adaptive sampling | algoritmi propri (numpy/scipy); ponte facoltativo verso harlow |

## Stato

| Fase | Stato | Note |
|---|---|---|
| 0 Setup | fatto | `pyproject.toml`, ruff, pytest, CI GitHub Actions (YAML validato, **non eseguito**), `tools/install_eplus.py` (testato su 8.7 `.sh` e 8.8 `.tar.gz`) |
| 1 Layer EnergyPlus multi-versione | fatto | `bdopt/eplus/`: ricerca installazioni, versioni, parser IDD, alias, catena di transizioni; matrice di test in `tests/test_eplus_versions.py` |
| 2 Sampling e adaptive | fatto | FLOLA/LOLA-Voronoi, Voronoi, varianza GP; ponte harlow testato |
| 3 Surrogati e ottimizzazione | fatto | GP (ARD, incertezza), MLP sklearn, TorchMLP; NSGA-II/III con vincoli |
| 4 Studio da configurazione e CLI | fatto | `DesignStudy`, `bdopt run study.yaml`, `bdopt eplus list` |
| 5 Rilascio | parziale | README e docs scritti; **nessuna pubblicazione su PyPI**; il pacchetto si installa da git |

## Decisioni di progetto

* **harlow non è il motore di default.** `harlow 0.1.0` importa TensorFlow/TensorFlow-Probability a livello di modulo
  e oggi un `pip install harlow` produce un import rotto (serve `tf_keras`, non dichiarato). Gli algoritmi sono
  riscritti in `bdopt.adaptive`; `HarlowAdaptiveSampler` è un ponte facoltativo che esegue harlow vero.
* **Lo schema SQL dei risultati è uguale in tutte le versioni 8.7–25.2** (verificato): un solo lettore. Cambiano
  invece nomi di campi, classi e meter (tabella in `aliases.py`, ricavata dal confronto reale degli IDD).
* **Parser IDF/IDD propri**: nessun IDD globale per processo, quindi nessun conflitto tra versioni.
* **`CommandRunner`** (prefisso di comando, es. `docker exec`) permette di usare EnergyPlus in un container senza
  che la libreria dipenda da Docker. Un runner via API Python di EnergyPlus non è stato implementato (nessun
  vantaggio rispetto al subprocess).
* **Variabili discrete (categoriche/intere) nel cubo unitario con "snap"** al valore valido prima di ogni valutazione,
  in campionamento, surrogati e ottimizzazione.
* **GP robusto:** la massimizzazione della verosimiglianza, con pochi riavvii, a volte collassava su una soluzione
  "solo rumore" (errore ~1.0 su alcuni disegni LHS). Limiti sulle lunghezze di scala, 8 riavvii e nuovi tentativi se
  rileva il collasso (vedi `docs/ADAPTIVE.md`).

## Risultati misurati

* Riproduce valori di riferimento salvati (4 punti, EnergyPlus 9.6) con differenza relativa massima 4e-13.
* Adaptive vs LHS a parità di punti di training: −37% di errore su una risposta con caratteristica **localizzata**,
  nessun vantaggio su risposta liscia o su soglia obliqua (tabella in `README.md`).
* Aggiornamento automatico di un modello 9.6 alla 25.2 (8 passi di transizione): ~230 s una tantum (poi in cache);
  risultati diversi di −2.3% / +4.9% / −5.8% rispetto alla 9.6 (causa non isolata).

## Non verificato

CI su GitHub, esecuzione su Windows/macOS, build dell'immagine Docker, pubblicazione su PyPI.
