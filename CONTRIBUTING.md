# How to Contribute to the `lymph` Package

First, thanks for considering to help out on this project!

Before you start coding away and fixing some of the [issues], let me give you an introduction of the standards and best-practices that we have tried to follow so far:


## Git

We use [git] as source control manager and [GitHub] as the repository hosting service.

Our branching model in [git] looks like this:

```mermaid
%%{init: {'gitGraph': {'rotateCommitLabel': false}} }%%
gitGraph
    commit id: "initial commit"
    branch dev
    checkout dev
    commit id: "some dev stuff"
    branch feature-1
    checkout feature-1
    commit id: "add this"
    commit id: "fix this"
    commit id: "refactor"
    checkout dev
    merge feature-1
    branch release-0.1.0
    checkout release-0.1.0
    commit id: "update changelog"
    checkout main
    merge release-0.1.0 tag: "v0.1.0"
    checkout dev
    merge main
    branch feature-2
    commit id: "add feat"
    commit id: "new test"
```

So, we have a `main` branch where every tagged release lives. Then there's a `dev` branch that is the starting point for any feature branches or fixes. In these feature or fix branches, the actual work happens. When a feature or fix is ready, it gets merged into the `dev` branch, from where a new `release-x.y.z` branch is created to finalize and maybe also test a new release. When everything looks good, that `release-x.y.z` branch is merged into the `main` branch, tagged and a release is created from it.


### Commit Messages

Since not so long ago, we use [conventional commits] in this repository. This is a standard, defining how commit messages should be written. It is worth the time to read the quick introduction on their website. Using it makes it easier to auto-update a draft of the `CHANGELOG.md`.

To avoid accidentally committing anything with a non-standard commit message, you can install a commit: If you have installed the package with the `dev` dependencies (for example by typing `pip install -e .[dev]`), then you can install the commit hook(s) by typing:

```
pre-commit install
pre-commit install --hook-type commit-msg
```

make sure this happens inside the virtual environment you have set up.


## Documentation

Right now, we use [sphinx] to compile a documentation from the docstrings of the code and host it on [readthedocs].

However, I like the simplistic style of [pdoc] a lot more that [sphinx], which can be convoluted. So, I would like to switch to [pdoc] at some point in the foreseeable future.

What I like about [pdoc] is that for the documentation to turn out well and useful, the code is necessarily well documented. It should be easy to understand and well described even when only reading the source code.


### Docstrings

The code uses [Google style docstrings] (or at least tries to do so most of the time). Generally, the [Google Python style guide] is worth a read and handy for reference.

Here's an example:

```python
def my_well_named_function(
    first_parameter: bool,
    second_parameter: int,
    another_parameter: Dict[str, Any],
    optional_parameter: Optional[float] = None,
) -> np.ndarray:
    """This function is well named and does its job very well.
    
    After a blank line we can add a bit of a longer description. This can even span
    multiple lines if it is necessary. Afterwards, the arguments follow.
    
    Args:
        first_parameter: The first parameter is an integer.
        second_parameter: The second parameter is a string.
        another_parameter: The third parameter is a float.
        optional_parameter: The fourth parameter is optional.

    Returns:
        An array filled with some super useful values.

    Examples:
        >>> my_well_named_function(True, 42, {"a": 1, "b": 2}, 3.14)
        array([1, 2, 3])
    """
    # here comes the code
```


## Tests

For simple functions and methods, I really like writing short and illustrative examples that can be used by [doctest] to test some basic functionality.

For more complicated stuff, we used [pytest] and [hypothesis] to cover a broad range of input values and edge cases. For the core components, this is important and ideally, with every new version of the package released, the portion of the code covered in tests goes up.


[issues]: https://github.com/rmnldwg/lymph/issues
[git]: https://git-scm.com
[GitHub]: https://github.com
[conventional commits]: https://www.conventionalcommits.org/en/v1.0.0/#summary
[sphinx]: https://www.sphinx-doc.org/en/master/
[readthedocs]: https://readthedocs.org/
[pdoc]: https://pdoc.dev
[Google style docstrings]: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[Google Python style guide]: https://google.github.io/styleguide/pyguide.html
[doctest]: https://docs.python.org/3.8/library/doctest.html
[pytest]: https://docs.pytest.org/en/7.3.x/
[hypothesis]: https://hypothesis.readthedocs.io/en/latest/
