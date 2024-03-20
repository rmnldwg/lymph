# Changelog

All notable changes to this project will be documented in this file.


<a name="1.1.0"></a>
## [1.1.0] - 2024-03-20

### Features

- (**utils**) Add `safe_set_params()` function.\
  This checks whether the params are a dict, list, or None and handles
  them accordingly. Just a convencience method that helped refactor some methods.
- Allow to pass state distributions to `posterior_state_dist()` and `risk()` methds. Fixes [#80].\
  With this, one can use precomputed state distributions to speed up
  computing the posterior or risk for multiple scenarios.

### Refactor

- Use `safe_set_params()` across models.

### Testing

- Add checks for midline risk. Related [#80].
- (**mid**) Fix wrong assumption in risk test.


<a name="1.0.0"></a>
## [1.0.0] - 2024-03-18

### Bug Fixes

- (**uni**) Catch error when `apply` to empty data. Fixes [#79].\
  For some reason, using `apply` on an empty `DataFrame` has an entirely
  different return type than when it is not empty. This caused the issue
  [#79] and has now been fixed.
- (**bi**) Data reload loads wrong side.\
  Now the data does not get reloaded anymore, which was actually
  unnecessary in the first place.
- (**uni**) Return correctly in `get_spread_params`.
- (**mid**) Consume & return params in same order.
- (**uni**) Allow `mapping=None` when loading data.

### Testing

- (**uni**) Check if loading empty data works. Related [#79].
- (**uni**) Make sure likelihood is deterministic.

### Change

- ⚠ **BREAKING** (**uni**) Shorten two (unused) method names.
- ⚠ **BREAKING** `helpers` are now `utils`.
- (**type**) Add type definition for graph dict.
- (**diag**) Use [partials] to save parametric dist.

[partials]: https://docs.python.org/3.10/library/functools.html#functools.partial

### Merge

- Branch 'main' into 'dev'.
- Branch '79-loading-an-empty-dataframe-raises-error' into 'dev'.


<a name="1.0.0.rc2"></a>
## [1.0.0.rc2] - 2024-03-06

Implementing the [lymixture] brought to light a shortcoming in the way the data and diagnose matrices are computed and stored. As mentioned in issue [#77], their rows are now aligned with the patient data, which may have some advantages for different use cases.

Also, since this is probably the last pre-release, I took the liberty to go over some method names once more and make them clearer.


### Bug Fixes

- Don't use fake T-stage for BN model. Related [#77].\
  Since we now have access to the full diagnose matrix by default, there
  is no need for the Bayesian network T-stage fix anymore.
- (**uni**) Reload data when modalities change.\
  Because we only store those diagnoses that are relevant to the model
  under the "_model" header in the `patient_data` table, we need to reload
  the patient data whenever we modify the modalities.


### Documentation

- Update to slightly changed API.
- (**bi**) Add bilateral quickstart to docs.

### Features

- (**mod**) Add utils to check for modality changes.

### Performance

- (**uni**) Make data & diagnose matrices faster. Related [#77].\
  The last change caused a dramatic slowdown (factor 500) of the data and
  diagnose matrix access, because it needed to index them from a
  `DataFrame`. Now, I implemented a basic caching scheme with a patient
  data cache version that brought back the original speed.
  Also, apparently `del dataframe[column]` is much slower than
  `dataframe.drop(columns)`. I replaced the former with the latter and now
  the tests are fast again.


### Refactor

- ⚠ **BREAKING** Rename methods for brevity & clarity.\
  Method names have been changed, e.g `comp_dist_evolution()` has been
  renamed to `state_dist_evo()` which is both shorter and (imho) clearer.
- (**uni**) Move data/diag matrix generation.

### Testing

- Update to slightly changed API.
- (**uni**) Check reset of data on modality change.\
  Added a test to make sure the patient data gets reloaded when the
  modalities change. This test is still failing.
- Finally suppress all `PerformanceWarnings`.

### Change

- ⚠ **BREAKING** Store data & diagnose matrices in data. Fixes [#77].\
  Instead of weird, dedicated `UserDict`s, I simply use the patient data
  to store the data encoding and diagnose probabilities for each patient.
  This has the advantage that the entire matrix (irrespective of T-stage)
  is aligned with the patients.
- ⚠ **BREAKING** (**bi**) Shorten kwargs.\
  The `(uni|ipsi|contra)lateral_kwargs` in the `Bilateral` constructor
  were shortened by removing the "lateral".


### Merge

- Branch 'main' into 'dev'.
- Branch '77-diagnose-matrices-not-aligned-with-data' into 'dev'.

### Remove

- Unused helpers.


<a name="1.0.0.rc1"></a>
## [1.0.0.rc1] - 2024-03-04

This release hopefully represents the last major change before releasing version 1.0.0. It was necessary because during the implementation of the midline model, managing the symmetries in a transparent and user-friendly way became impossible in the old implementation.

Now, a [composite pattern] is used for both the modalities and the distributions over diagnose times. This furhter separates the logic and will allow more hierarchical models based on the ones provided here to work seamlessly almost out of the box. This may become relevant with the mixture model.

[composite pattern]: https://refactoring.guru/design-patterns/composite


### Add

- First version of midline module added.

### Bug Fixes

- (**diag**) Delete frozen distributions when params change.
- (**diag**) Correct max time & params.\
  The `max_time` is now correctly accessed and set. Also, the distribution
  params are not used up by synched distributions, but only by the
  distributions in composite leafs.
- (**graph**) Avoid warning for micro mod setting.
- ⚠ **BREAKING** Make likelihood work with emcee again.\
  The way the likelihood was defined, it did not actually play nicely with
  how the emcee package works. This is now fixed.
- (**bi**) Fix uninitialized `is_symmetric` dict.
- (**mid**) Add missing dict in init.
- (**mid**) Update call to `transition_matrix()` & `state_list`.
- (**mid**) Finish `draw_patients` method.\
  Some bugs in the method for drawing synthetic patients from the
  `Midline` were fixed. This seems to be working now.


### Documentation

- (**mid**) Improve midline docstrings slightly.
- Go over `set_params()` docstrings.
- Update quickstart guide to new API.
- Adapt tests to new API (now passing).
- Update index & fix some docstrings.
- Fix some typos and cross-references.

### Features

- (**helper**) Add `popfirst()` and `flatten()`.\
  Two new helper function in relation to getting and setting params.
- (**type**) Add model ABC to inherit from.\
  I added an abstract base class from which all model-like classes should
  inherit. It defines all the methods that need to be present in a model.\
  The idea behind this is that any subclass of this can be part of a
  composite that correctly delegates getting/setting parameters,
  diagnose time distributions, and modalities.
- ⚠ **BREAKING** (**graph**) Add `__hash__` to edge, node, graph.\
  This replaces the dedicated `parameter_hash()` method.
- (**mod**) Add method to delete modality `del_modality()`.
- Add more get/set params methods.
- (**mid**) Implement `set_params`.
- (**mid**) Implement the `load_patient_data` meth.
- (**mid**) Finish midline (feature complete).
- Complete set/get methods on model classes.\
  The `Unilateral`, `Bilateral`, and `Midline` model now all have the six
  methods `set_tumor_spread_params`, `set_lnl_spread_params`,
  `set_spread_params`, `set_params`, `get_tumor_spread_params`,
  `get_lnl_spread_params`, `get_spread_params`, and `get_params`.
- (**mid**) Reimplement the midline evolution.\
  The midline evolution that Lars Widmer worked on is now reimplemented.
  However, although this implementation is analogous to the one used in
  previsou version of the code and should thus work, it is still untested
  at this point.
- Add helper to draw diagnoses.\
  The new helper function`draw_diagnoses` is a re-implementation of the
  `Unilateral` class's method with the same name for easier reusing.
- (**mid**) Allow marginalization over unknown midline extension.\
  This is implemented differently than before: If data with unknown
  midline extension is added, it gets loaded into an attribute named
  `unknown`, which is a `Bilateral` model only used to store that data and
  generate diagnose matrices.


### Miscellaneous Tasks

- Move timing data.
- Make changelog super detailed.

### Refactor

- (**mid**) Split likelihood method.

### Testing

- Fix long-running test.
- Add integration tests with emcee.
- Add checks for bilateral symmetries.
- (**mid**) Add first check of `set_params()` method.
- (**mid**) Check likelihood function.

### Add

- Added doc strings.

### Change

- Non-mixture midline implemented.\
  fixed the non mixture midline extension model and added documentation
- ⚠ **BREAKING** Make `get_params()` uniform and chainable.\
  The API of all `get_params()` methods is now nice and uniform, allowing
  arbitrary chaining of these methods.
- ⚠ **BREAKING** Make `set_params()` uniform and chainable.\
  The API of all `set_params()` methods is now nice and uniform,
  allowing arbitrary chaining of these methods.
- ⚠ **BREAKING** Make `set_params()` not return kwargs.\
  It does make sense to "use up" the positional arguments one by one in
  the `set_params()` methods, but doing the same thing with keyword
  arguments is pointless, difficult and error prone.
- ⚠ **BREAKING** (**graph**) Replace `name` with `get_name()`.\
  In the `Edge` class, the `name` property is replaced by a function
  `get_name()` that is more flexible and allows us to have edge names
  without underscores when we need it.
- ⚠ **BREAKING** (**bi**) Reintroduce `is_symmetric` attribute.\
  This will once again manage the symmetry of the `Bilateral` class's
  different ipsi- and contralateral attributes.
- ⚠ **BREAKING** (**diag**) Use composite for distributions.\
  Instead of a dict that holds the T-stages and corresponding
  distributions over diagnose times, this implements them as a composite
  pattern. This replaces the dict-like API entirely with methods. This has
  several advantages:
  1. It is more explicit and thus more readable
  2. The composite pattern is designed to work naturally with tree-like
  structures, which we have here when dealing with bilateral models.
- ⚠ **BREAKING** (**mod**) Use composite for modalities.\
  Instead of a dict that holds the names and corresponding
  sens/spec for diagnostic modalities, this implements them as a composite
  pattern. This replaces the dict-like API entirely with methods. This has
  several advantages:
  1. It is more explicit and thus more readable
  2. The composite pattern is designed to work naturally with tree-like
  structures, which we have here when dealing with bilateral models.
- ⚠ **BREAKING** (**uni**) Transform to composite pattern.\
  Use the new composite pattern for the distribution over diagnose times
  and modalities.
- (**bi**) Update for new composite API.
- ⚠ **BREAKING** (**mod**) Shorten to sens/spec.\
  Also, add a `clear_modalities()` and a `clear_distributions()` method to
  the respective composites.
- (**matrix**) Use hashables over arg0 cache.\
  Instead of using this weird `arg0_cache` for the observation and
  transition matrix, I use the necessary arguments only, which are all
  hashable now.
- ⚠ **BREAKING** Adapt risk to likelihood call signature.
- (**type**) Add risk to abstract methods.
- (**type**) Abstract methods raise error.

### Merge

- Branch 'yoel-dev' into 'dev'.
- Branch '74-synchronization-is-unreadable-and-error-prone' into 'dev'. Fixes [#74].
- Branch 'main' into 'dev'.
- Branch 'add-midext-evolution' into 'dev'.

### Remove

- Unused helper functions.


<a name="1.0.0.a6"></a>
## [1.0.0.a6] - 2024-02-15

With this (still alpha) release, we most notably fixed a long unnoticed bug in the computation of the Bayesian network likelihood.

### Bug Fixes

- (**uni**) Leftover `kwargs` now correctly returned in `assign_params()`
- ⚠ **BREAKING** (**uni**) Remove `is_<x>_shared` entirely, as it was unused anyways. Fixes [#72].
- T-stage mapping may be dictionary or callable
- (**uni**) Raise exception when there are no tumors or LNLs in graph

### Documentation

- Fix typo in modalities

### Testing

- (**uni**) Check constructor raises exceptions
- Check the Bayesian network likelihood

### Change

- (**uni**) Trinary params are shared by default
- (**uni**) Prohibit setting `max_time`
- ⚠ **BREAKING** Change `likelihood()` API: We don't allow setting the data via the `likelihood()` anymore. It convoluted the method and setting it beforehand is more explicit anyways.


<a name="1.0.0.a5"></a>
## [1.0.0.a5] - 2024-02-06

In this alpha release we fixed more bugs and issues that emerged during more rigorous testing.

Most notably, we backed away from storing the transition matrix in a model's instance. Because it created opaque and confusion calls to functions trying to delete them when parameters were updated.

Instead, the function computing the transition matrix is now globally cached using a hash function from the graph representation. This has the drawback of slightly more computation time when calculating the hash. But the advantage is that e.g. in a bilateral symmetric model, the transition matrix of the two sides is only ever computed once when (synched) parameters are updated.

### Bug Fixes

- (**graph**) Assume `nodes` is dictionary, not a list. Fixes [#64].
- (**uni**) Update `draw_patients()` method to output LyProX style data. Fixes [#65].
- (**bi**) Update bilateral data generation method to also generate LyProX style data. Fixes [#65].
- (**bi**) Syntax error in `init_synchronization`. Fixes [#69].
- (**uni**) Remove need for transition matrix deletion via a global cache. Fixes [#68].
- (**uni**) Use cached matrices & simplify stuff. Fixes [#68].
- (**uni**) Observation matrix only property, not cached anymore

### Documentation

- Fix typos & formatting errors in docstrings

### Features

- (**graph**) Implement graph hash for global cache of transition matrix
- (**helper**) Add an `arg0` cache decorator that caches based on the first argument only
- (**matrix**) Use cache for observation & diagnose matrices. Fixes [#68].

### Miscellaneous Tasks

- Update dependencies & classifiers

### Refactor

- Variables inside `generate_transition()`

### Testing

- Make doctests discoverable by unittest
- Update tests to changed API
- (**uni**) Assert format & distribution of drawn patients
- (**uni**) Allow larger delta for synthetic data distribution
- (**bi**) Check bilateral data generation method
- Check the bilateral model with symmetric tumor spread
- Make sure delete & recompute synced edges' tensor work
- Adapt tests to changed `Edge` API
- (**bi**) Evaluate transition matrix recomputation
- Update tests to match new transition matrix code
- Update trinary unilateral tests

### Change

- ⚠ **BREAKING** Compute transition tensor globally. Fixes [#69].
- ⚠ **BREAKING** Make transition matrix a method instead of a property. Fixes [#40].
- ⚠ **BREAKING** Make observation matrix a method instead of a property. Fixes [#40].

### Ci

- Add coverage test dependency back into project

### Remove

- Unused files and directories


<a name="1.0.0.a4"></a>
## [1.0.0.a4] - 2023-12-12

### Bug Fixes

- Use `lnls.keys()` consistently everywhere
- Warn about symmetric params in asymmetric graph
- Make `allowed_states` accessible
- Provide `base` keyword argument to `compute_encoding()`. This is necessary for the trinary model (see [#45])
- Ensure confusion matrix of trinary diagnostic modality has correct shape
- Make diagnostic encoding always binary
- Correct joint state/diagnose matrix (fixes [#61])
- Send kwargs to both `assign_params` methods (fixes [#60])
- Enable two-way sync between lookup dicts (fixes [#62])

### Documentation

- Add "see also" to get/set methods, thereby making them reference each other

### Features

- Add trinary & keywords in encoding: When computing the risk for a certain pattern in a trinary model, one may now provide different kewords like `"macro"` to differentiate between different involvements of interest.
- Add convenience constructors to create `binary` and `trinary` bilateral models
- Allow bilateral model with an asymmetric graph structure
- Add get/set methods to `DistributionsUserDict`, which makes all `get_params()` and `set_params()` methods consistent across their occurences

### Refactor

- Pull initialization of ipsi- & contralateral models out of `Bilateral` model's `__init__()`
- Restructure `Bilateral` model's `__init__()` method slightly

### Testing

- Cover bilateral risk computation
- Cover unilateral risk method
- Check asymmetric model implementation
- Check binary/trinary & `allowed_states`
- Add trinary likelihood test
- Add risk check for trinary model
- Add checks for delegation of attrbutes & setting of params
- Check `cached_property` delegation works
- Check param assign thoroughly

### Change

- Don't use custom subclass of `cached_property` that forbids setting and use the default `cached_property` instead
- Encode symmetries of `Bilateral` model in a special dict called `is_summetric` with keys `"tumor_spread"`, `"lnl_spread"`, and `"modalities"`


<a name="1.0.0.a3"></a>
## [1.0.0.a3] - 2023-12-06

Fourth alpha release. [@YoelPH](https://github.com/YoelPH) noticed some more bugs that have been fixed now. Most notably, the risk prediction raised exceptions, because of a missing transponed matrix `.T`.

### Bug Fixes

- Raise `ValueError` if diagnose time parameters are invalid (Fixes [#53])
- Use names of LNLs in unilateral `comp_encoding()` (Fixes [#56])
- Wrong shape in unilateral posterior computation (missing `.T`) (Fixes [#57])
- Wrong shape in bilateral joint posterior computation (missing `.T`) (Fixes [#57])

### Documentation

- Add info on diagnose time distribution's `ValueError`

### Testing

- `ValueError` raised in diagnose time distribution's `set_params`
- Check `comp_encoding_diagnoses()` for shape and dtype
- Test unilateral posterior state distribution for shape and sum
- Test bilateral posterior joint state distribution for shape and sum


<a name="1.0.0.a2"></a>
## [1.0.0.a2] - 2023-09-15

Third alpha release. I am pretty confident that the `lymph.models.Unilateral` class works as intended since it _does_  yield the same results as the `0.4.3` version.

The `lymph.models.Bilateral` class is presumably finished now as well, although there may still be issues with that. It does however compute a likelihood if asked to do so, and the results don't look implausible. So, it might be worth giving it a spin.

Also, I am now quite satisfied with the look and usability of the new API. Hopefully, this means only minor changes from here on.

### Bug Fixes

- (**bi**) Sync callback was wrong way around
- Assigning new `modalities` now preserves the `trigger_callbacks`
- Set `diag_time_dists` params won't fail anymore
- (**bi**) Don't change dict size during `modalities` sync
- (**bi**) Delegted generator attribute now resets
- (**bi**) Make `modalities`/`diag_time_dists` syncable
- (**uni**) Evolution is now running through all time-steps

### Documentation

- Switch to MyST style sphinx theme
- 🛠️ Start with bilateral quickstart guide
- (**uni**) Reproduce llh with old and new model

### Features

- Re-implement bilateral model class
- (**bi**) Continue rewriting bilateral class
- (**helper**) Add `DelegatorMixin` to helpers
- (**uni**) Use delegator to pull graph attrs up
- (**bi**) Add delegation of uni attrs to bilateral
- (**bi**) Reimplement joint state/obs dists & llh
- (**uni**) Allow global setting of micro & growth
- (**uni**) Reimplement Bayesian network model
- (**log**) Add logging to sync callback creation
- Get params also as iterator
- (**uni**) Get only edge/dist params

### Refactor

- `state_list` is now a member of the `graph.Representation` & computation of involvement pattern encoding is separate function now
- subclasses of `cached_property` are used for e.g. transition and observation matrix instead of convoluted descriptors

### Testing

- (**uni**) Add tests w.r.t. delegator mixin
- (**bi**) Check the delegation of ipsi attrs
- (**bi**) Check sync for bilateral model
- Refactor out fixtures from test suite
- Make sure bilateral llh is deterministic
- Catch warnings for cleaner output
- (**uni**) Add likelihood value tests

### Change

- `assign_params` & joint posterior
- ⚠ **BREAKING** (**graph**) Remove `edge_params` lookup in favour of an `edge` dictionary in the `graph.Representation`
- ⚠ **BREAKING** The edge's and dist's `get_params()` and `set_params()` methods now have the same function signature, making a combined loop over both possible
- (**bi**) Rewrite the bilateral risk method
- ⚠ **BREAKING** Allow setting params as positional & keyword arguments in both the likelihood and the risk method

### Ci

- Bump codecov action to v3

### Merge

- Branch 'main' into 'dev'
- Branch 'dev' into 'reimplement-bilateral'
- Branch 'delegation-pattern' into 'dev'
- Branch 'dev' into 'reimplement-bilateral'
- Branch 'remove-descriptors' into 'reimplement-bilateral'
- Branch 'reimplement-bilateral' into 'dev'


<a name="1.0.0.a1"></a>
## [1.0.0.a1] - 2023-08-30

Second alpha release, aimed at testing the all new implementation. See these [issues](https://github.com/rmnldwg/lymph/milestone/1) for an idea of what this tries to address.

### Bug Fixes
- (**matrix**) Wrong shape of observation matrix for trinary model

### Documentation
- Fix wrong python version in rtd config file
- Remove outdated sampling tutorial
- Remove deprecated read-the-docs config
- Tell read-the-docs to install extra requirements
- Execute quickstart notebook

### Testing
- Check correct shapes for trinary model matrices


<a name="1.0.0.a0"></a>
## [1.0.0.a0] - 2023-08-15

This alpha release is a reimplementation most of the package's API. It aims to solve some [issues](https://github.com/rmnldwg/lymph/milestone/1) that accumulated for a while.

### Features
- parameters can now be assigned centrally via a `assign_params()` method, either using args or keyword arguments. This resolves [#46]
- expensive operations generally look expensive now, and do not just appear as if they were attribute assignments. Fixes [#40]
- computations around the the likelihood and risk predictions are now more modular. I.e., several conditional and joint probability vectors/matrices can now be computed conveniently and are not burried in large methods. Resolves isse [#41]
- support for the trinary model was added. This means lymph node levels (LNLs) can be in one of three states (healthy, microscopic involvement, macroscopic metatsasis), instead of only two (healthy, involved). Resolves [#45]

### Documentation
- module, class, method, and attribute docstrings should now be more detailed and helpful. We switched from strictly adhering to Numpy-style docstrings to something more akin to Python's core library docstrings. I.e., parameters and behaviour are explained in natural language.
- quickstart guide has been adapted to the new API

### Code Refactoring
- all matrices related to the underlying hidden Markov model (HMM) have been decoupled from the `Unilateral` model class
- the representation of the directed acyclic graph (DAG) that determined the directions of spread from tumor to and among the LNLs has been implemented in a separate class of which an instance provides access to it as an attribute of `Unilateral`
- access to all parameters of the graph (i.e., the edges) is bundled in a descriptor holding a `UserDict`

### BREAKING CHANGES
Almost the entire API has changed. I'd therefore recommend to have a look at the [quickstart guide](https://lymph-model.readthedocs.io/en/1.0.0.a0/quickstart.html) to see how the new model is used. Although most of the core concepts are still the same.

<a name="0.4.3"></a>
## [0.4.3] - 2022-09-02

### Bug Fixes
- incomplete involvement for unilateral risk method does not raise KeyError anymore. Fixes issue [#38]

<a name="0.4.2"></a>
## [0.4.2] - 2022-08-24

### Documentation
- fix the issue of docs failing to build
- remove outdated line in install instructions
- move conf.py back into source dir
- bundle sphinx requirements
- update the quickstart & sampling notebooks
- more stable sphinx-build & update old index

### Maintenance
- fine-tune git-chglog settings to my needs
- start with a CHANGELOG
- add description to types of allowed commits


<a name="0.4.1"></a>
## [0.4.1] - 2022-08-23
### Bug Fixes
- pyproject.toml referenced wrong README & LICENSE


<a name="0.4.0"></a>
## [0.4.0] - 2022-08-23
### Code Refactoring
- delete unnecessary utils

### Maintenance
- fix pyproject.toml typo
- add pre-commit hook to check commit msg


[Unreleased]: https://github.com/rmnldwg/lymph/compare/1.1.0...HEAD
[1.1.0]: https://github.com/rmnldwg/lymph/compare/1.0.0...1.1.0
[1.0.0]: https://github.com/rmnldwg/lymph/compare/1.0.0.rc2...1.0.0
[1.0.0.rc2]: https://github.com/rmnldwg/lymph/compare/1.0.0.rc1...1.0.0.rc2
[1.0.0.rc1]: https://github.com/rmnldwg/lymph/compare/1.0.0.a6...1.0.0.rc1
[1.0.0.a6]: https://github.com/rmnldwg/lymph/compare/1.0.0.a5...1.0.0.a6
[1.0.0.a5]: https://github.com/rmnldwg/lymph/compare/1.0.0.a4...1.0.0.a5
[1.0.0.a4]: https://github.com/rmnldwg/lymph/compare/1.0.0.a3...1.0.0.a4
[1.0.0.a3]: https://github.com/rmnldwg/lymph/compare/1.0.0.a2...1.0.0.a3
[1.0.0.a2]: https://github.com/rmnldwg/lymph/compare/1.0.0.a1...1.0.0.a2
[1.0.0.a1]: https://github.com/rmnldwg/lymph/compare/1.0.0.a0...1.0.0.a1
[1.0.0.a0]: https://github.com/rmnldwg/lymph/compare/0.4.3...1.0.0.a0
[0.4.3]: https://github.com/rmnldwg/lymph/compare/0.4.2...0.4.3
[0.4.2]: https://github.com/rmnldwg/lymph/compare/0.4.1...0.4.2
[0.4.1]: https://github.com/rmnldwg/lymph/compare/0.4.0...0.4.1
[0.4.0]: https://github.com/rmnldwg/lymph/compare/0.3.10...0.4.0

[#80]: https://github.com/rmnldwg/lymph/issues/80
[#79]: https://github.com/rmnldwg/lymph/issues/79
[#77]: https://github.com/rmnldwg/lymph/issues/77
[#74]: https://github.com/rmnldwg/lymph/issues/74
[#72]: https://github.com/rmnldwg/lymph/issues/72
[#69]: https://github.com/rmnldwg/lymph/issues/69
[#68]: https://github.com/rmnldwg/lymph/issues/68
[#65]: https://github.com/rmnldwg/lymph/issues/65
[#64]: https://github.com/rmnldwg/lymph/issues/64
[#62]: https://github.com/rmnldwg/lymph/issues/62
[#61]: https://github.com/rmnldwg/lymph/issues/61
[#60]: https://github.com/rmnldwg/lymph/issues/60
[#57]: https://github.com/rmnldwg/lymph/issues/57
[#56]: https://github.com/rmnldwg/lymph/issues/56
[#53]: https://github.com/rmnldwg/lymph/issues/53
[#46]: https://github.com/rmnldwg/lymph/issues/46
[#45]: https://github.com/rmnldwg/lymph/issues/45
[#41]: https://github.com/rmnldwg/lymph/issues/41
[#40]: https://github.com/rmnldwg/lymph/issues/40
[#38]: https://github.com/rmnldwg/lymph/issues/38
