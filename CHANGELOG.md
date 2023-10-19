# Changelog

All notable changes to this project will be documented in this file.


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


[Unreleased]: https://github.com/rmnldwg/lymph/compare/1.0.0.a2...HEAD
[1.0.0.a2]: https://github.com/rmnldwg/lymph/compare/1.0.0.a1...1.0.0.a2
[1.0.0.a1]: https://github.com/rmnldwg/lymph/compare/1.0.0.a0...1.0.0.a1
[1.0.0.a0]: https://github.com/rmnldwg/lymph/compare/0.4.3...1.0.0.a0
[0.4.3]: https://github.com/rmnldwg/lymph/compare/0.4.2...0.4.3
[0.4.2]: https://github.com/rmnldwg/lymph/compare/0.4.1...0.4.2
[0.4.1]: https://github.com/rmnldwg/lymph/compare/0.4.0...0.4.1
[0.4.0]: https://github.com/rmnldwg/lymph/compare/0.3.10...0.4.0

[#46]: https://github.com/rmnldwg/lymph/issues/46
[#45]: https://github.com/rmnldwg/lymph/issues/45
[#41]: https://github.com/rmnldwg/lymph/issues/41
[#40]: https://github.com/rmnldwg/lymph/issues/40
[#38]: https://github.com/rmnldwg/lymph/issues/38
