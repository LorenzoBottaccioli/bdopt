# EnergyPlus versions (8.7 and newer)

bdopt works with every EnergyPlus release from 8.7. Everything that depends on the version lives in
`bdopt/eplus/`; this page says what is handled and how.

| Aspect | How it is handled | Checked on |
|---|---|---|
| Finding the engine | `EnergyPlus.find(version=...)`, environment variables, `PATH`, standard folders; version read from the `!IDD_Version` line of `Energy+.idd` | 8.7, 9.0.1, 9.6, 23.1, 24.2, 25.2 |
| Command line | `energyplus -w EPW -d OUTDIR [-x] [-D] IDF`, identical in all versions; `-x` (ExpandObjects) is added when the model has `HVACTemplate:` objects | all six |
| Results | `eplusout.sql`: `ReportDataDictionary` / `ReportData` have the same layout from 8.7 to 25.2, so one reader serves all | all six |
| Field/class names | resolved through the IDD of the version in use; renames are aliased (`bdopt/eplus/aliases.py`) | see below |
| Meter names | `DistrictHeating:Facility` is requested as `DistrictHeating`, `DistrictHeatingWater` and `DistrictHeatingSteam`; the values reported are summed | 25.2 reports `DistrictHeatingWater` |
| Model older than the engine | `auto_upgrade=True` runs the `IDFVersionUpdater` transitions shipped with EnergyPlus, chained across installations (the 9.6 install covers 7.2 -> 9.6, the 25.2 install 9.0 -> 25.2); result cached | 8.7 -> 9.6 ... 25.2, 9.6 -> 25.2 |
| Model newer than the engine | refused (`TransitionError`) | |

Renames found by diffing the IDDs of 8.7, 9.0, 9.6, 23.1, 24.2 and 25.2 (`tools/idd_diff.py`):

* `Lights` / `*Equipment`: `Watts per Zone Floor Area` -> `Watts per Floor Area` (24.2)
* `ZoneInfiltration:DesignFlowRate`: `Flow per Zone Floor Area` -> `Flow Rate per Floor Area`,
  `Flow per Exterior Surface Area` -> `Flow Rate per Exterior Surface Area` (23.1)
* `ZoneVentilation:DesignFlowRate`: `Flow Rate per Zone Floor Area` -> `Flow Rate per Floor Area` (23.1)
* zone-name fields: `Zone or ZoneList Name` -> `Zone or ZoneList or Space or SpaceList Name` (9.6 / 23.1)
* `Output:Meter`: `Name` -> `Key Name` (9.0)
* class `WindowProperty:ShadingControl` -> `WindowShadingControl` (9.0)

Fields that were only added or removed need no alias, because fields are addressed by name, never by
position. When a new EnergyPlus version appears, run `tools/idd_diff.py` between it and the previous one and add
what it reports to `aliases.py`. The tables cover the classes the platform uses; for others the diff tool
shows what to add, and an unknown name fails early with suggestions.

Running a different version than the model's: the `auto_upgrade` step is slow (about 10-30 s per
version step, 230 s for the platform's Office model from 9.6 to 25.2) but happens once per model and target
version. Engine results also differ between versions (the same model gave -2.3 % / +4.9 % / -5.8 % on the three
objectives between 9.6 and 25.2); compare before mixing results from different versions.

Old binaries: the 8.7 Linux release is a generic `x86_64` build and runs on current distributions; the 8.7
Windows and macOS releases only ship installers (`.exe`, `.dmg`), which `tools/install_eplus.py` can run silently
only on Windows.
