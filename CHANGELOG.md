# Changelog

All notable changes to this project will be documented in this file.

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


[Unreleased]: https://github.com/rmnldwg/lymph/compare/1.0.0.a5...HEAD
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
