# Objectives

An objective says how to read one number out of an EnergyPlus run. Results are read from the run's
`eplusout.sql`, whose layout is the same in every version from 8.7 to 25.2.

## Meters

A plain string is a meter name:

```python
Problem(parameters, ["Electricity:Facility", "DistrictHeating:Facility", "DistrictCooling:Facility"])
```

For units, aggregation or direction use {class}`~bdopt.MeterReader`:

```python
from bdopt import MeterReader

MeterReader("Electricity:Facility", name="Electricity [kWh]", unit="kWh")
MeterReader("Electricity:Facility", name="Peak electricity", aggregate="max", unit="kWh")
MeterReader("ElectricityProduced:Facility", name="Self-produced", direction="max")   # maximise
```

| Argument | Meaning | Default |
|---|---|---|
| `meter` | EnergyPlus meter name | |
| `name` | name of the result column | the meter name |
| `unit` | `J`, `kJ`, `MJ`, `GJ`, `kWh`, `MWh` (applies to energy meters in Joules) | `J` |
| `aggregate` | how the run-period values are combined (a model with several run periods reports several): `sum`, `max`, `mean` | `sum` |
| `missing` | what to do if EnergyPlus does not report the meter: `zero` (with a warning), `nan`, `error` | `zero` |
| `direction` | `min` or `max`; used by the optimizers | `min` |

:::{note}
A meter only exists in a run if something consumes that resource: a model with no cooling has no
`DistrictCooling:Facility` meter. With the default `missing="zero"` the objective is `0.0` and a warning says so,
which is also what a **misspelt meter name** produces. If that worries you, use `missing="error"`.
:::

### Meter names that changed between versions

`DistrictHeating:Facility` became `DistrictHeatingWater:Facility` (and `...Steam`) in newer versions. bdopt
requests all candidates and **sums the ones reported**, so one name works everywhere. The same applies to the
end-use meters (`Heating:DistrictHeating`).

## Output variables

{class}`~bdopt.VariableReader` reads an `Output:Variable`, aggregated over the run period or over a finer
frequency:

```python
from bdopt import VariableReader

VariableReader("*", "Zone Mean Air Temperature", name="Mean zone temperature", aggregate="mean",
               frequency="Hourly")
VariableReader("SPACE1-1", "Zone Air System Sensible Heating Energy", name="Heating, zone 1",
               aggregate="sum", frequency="RunPeriod")
```

`key_value` is the object the variable belongs to (`"*"`: all of them, aggregated together). `frequency` is
`RunPeriod`, `Annual`, `Monthly`, `Daily`, `Hourly`, `Timestep` or `Detailed`. The missing-output behaviour
defaults to `nan` here, because a wrong variable name usually means a mistake.

## Custom objectives

{class}`~bdopt.FunctionReader` calls your function with a {class}`~bdopt.simulation.RunResults`, which can query
the SQL file, and declares which outputs EnergyPlus must produce:

```python
import numpy as np
from bdopt import FunctionReader

def hours_above_26(run):
    temps, _ = run.series("Zone Operative Temperature", is_meter=False, key="SPACE1-1", frequency="Hourly")
    return float(np.sum(temps > 26.0))

FunctionReader(
    hours_above_26,
    name="Overheating hours",
    outputs=[("Output:Variable", "SPACE1-1", "Zone Operative Temperature", "Hourly")],
)
```

## Run-period results only

bdopt reports one number per objective per run (an annual total, a peak, a mean ...). Time series as such are not
returned; use a {class}`~bdopt.FunctionReader` to reduce them to the number you need.
