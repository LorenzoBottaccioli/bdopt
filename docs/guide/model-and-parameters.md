# Model and parameters

## Loading a model

```python
from bdopt import load_idf

idf = load_idf("Office_IDF.idf")
print(idf)                       # IDF(513 objects, version=9.6.0)
idf.version                      # Version('9.6.0'), from the Version object of the file
idf.by_class("Lights")           # all Lights objects (case-insensitive)
idf.find("Material", "_InsulationEX_.01")     # by class and object name; "*" selects all
```

{class}`~bdopt.IDF` is a plain description of the file: a list of {class}`~bdopt.IDFObject` (class name and
field values as strings). It is read and written without an IDD, so it works for any EnergyPlus version and
there is no global state shared between versions. Comments are not preserved when a model is written back
({meth}`IDF.save <bdopt.IDF.save>`); the results do not depend on them.

You rarely edit an {class}`~bdopt.IDF` yourself: a {class}`~bdopt.Problem` does it for every design.

## Selectors: where a value is written

A **selector** says which field of which objects receives the parameter's value.

```python
from bdopt import FieldSelector

# one object, by name
FieldSelector("Material", "_InsulationEX_.01", "Thickness")

# every object of a class
FieldSelector("Lights", "*", "Watts per Zone Floor Area")

# only some objects: `where` is {field name: required value} or a function of the object
FieldSelector("FenestrationSurface:Detailed", "*", "Construction Name",
              where={"Surface Type": "Window"})
FieldSelector("Zone", "*", "Multiplier", where=lambda obj: obj.name.startswith("Office"))
```

Class names, object names, field names and `where` values are compared **case-insensitively**. Class and field
names are resolved against the `Energy+.idd` of the EnergyPlus version that runs the model, so the same selector
works across versions: renamed fields are aliased (`Watts per Zone Floor Area` also finds `Watts per Floor Area`
in 24.2 and later; see {doc}`energyplus-versions`). Fields can also be given by position
(`field_index=5`) when you must.

When a name is wrong, the error appears **when the evaluator is created**, with suggestions:

```text
KeyError: field 'Lighting Levl' does not exist in class 'Lights' of EnergyPlus 9.6.0. Did you mean: Lighting Level?
ParameterError: selector 'Lights:NO SUCH LIGHT:Lighting Level' matches no object in the model (class 'Lights', object 'NO SUCH LIGHT', where=None)
```

For anything a field cannot express (geometry, schedules, a value written in several places) use a
{class}`~bdopt.GenericSelector`, a function of the model and the value:

```python
from bdopt import GenericSelector

def set_wwr(idf, value):
    ...                           # edit idf in place

GenericSelector(set_wwr, name="window-to-wall ratio")
```

## Value descriptors: which values are allowed

| Class | Values | Unit-cube mapping |
|---|---|---|
| {class}`~bdopt.RangeParameter` `(min, max)` | continuous | linear |
| {class}`~bdopt.IntegerParameter` `(min, max)` | integers, both ends included | one equal bin per integer |
| {class}`~bdopt.CategoryParameter` `(options)` | strings or numbers, at least two, all different | one equal bin per option |

## Parameters

```python
from bdopt import Parameter, RangeParameter, CategoryParameter

wall = Parameter(FieldSelector("Material", "_InsulationEX_.01", "Thickness"),
                 RangeParameter(0.1, 0.5), name="Wall insulation")
glazing = Parameter(
    FieldSelector("FenestrationSurface:Detailed", "*", "Construction Name", where={"Surface Type": "Window"}),
    CategoryParameter(["single_glazing", "double_glazing", "triple_glazing"]),
    name="Glazing",
)
```

`name` defaults to a description of the selector; names must be unique within a problem. The descriptor can also
be written as a `(min, max)` tuple or a list of options.

:::{admonition} One parameter, several objects
:class: tip
A selector with `"*"` (or a `where`) writes the *same* value into every object it matches: one design variable
for all windows. If you want an independent variable per window, create one {class}`~bdopt.Parameter` per
window with a distinct `name`; the dimension of the problem grows accordingly.
:::

## The problem

```python
from bdopt import Problem

problem = Problem(
    parameters=[wall, glazing],
    objectives=["Electricity:Facility", "DistrictHeating:Facility"],
    constraints=[lambda row: row["Wall insulation"] - 0.45],     # g(x) <= 0 is feasible
)
problem.names()               # parameter names followed by objective names
problem.decode(unit_points)   # unit cube -> table of real values
problem.encode(table)         # table -> unit cube
problem.bounds()              # real-valued bounds (a category: index 0..n-1)
```

A **constraint** is a function of a parameter row (`{name: value}`) returning a number, feasible when it is
`<= 0`. Samplers draw only feasible points and the optimizers treat them as constraints.
