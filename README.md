# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/lycosystem/lymph/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                             |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------- | -------: | -------: | ------: | --------: |
| src/lymph/\_\_init\_\_.py        |       20 |        7 |     65% |     43-49 |
| src/lymph/\_version.py           |       13 |        0 |    100% |           |
| src/lymph/diagnosis\_times.py    |      247 |       41 |     83% |85, 140, 153, 166, 175-176, 187, 243-244, 277, 334, 346, 350, 358, 367, 370, 374-378, 410, 429-434, 441-448, 452-457, 461-470 |
| src/lymph/graph.py               |      313 |       49 |     84% |56-57, 81, 104, 120-123, 140, 166, 186, 198-214, 265, 276, 279, 296, 346, 349, 352, 365, 371, 555, 568, 634-638, 703-707, 716-721, 741-743 |
| src/lymph/matrix.py              |       76 |       11 |     86% |169, 177-178, 217-218, 234-248 |
| src/lymph/modalities.py          |      146 |       19 |     87% |29, 45, 51, 67, 70, 83, 86, 106, 120, 123, 126, 129, 132, 195, 215-216, 244, 293-294 |
| src/lymph/models/\_\_init\_\_.py |        5 |        0 |    100% |           |
| src/lymph/models/bilateral.py    |      213 |       35 |     84% |129-131, 141-143, 149, 157, 180, 188, 443-449, 469-472, 484-490, 497-503, 515, 562-563, 567-570, 648, 715, 718-719 |
| src/lymph/models/hpv.py          |      151 |      103 |     32% |29-33, 78-97, 107-121, 131-133, 143-145, 150-153, 158-161, 169-188, 201-206, 217-223, 240-246, 250-267, 271-282, 286-287, 291-292, 306-318, 326-335, 339-348, 377-389, 402, 415, 423, 431 |
| src/lymph/models/midline.py      |      381 |       97 |     75% |109, 128, 175, 195, 198, 207, 213, 221, 227, 245, 257, 282-285, 290, 317, 323, 343-346, 426-435, 464-474, 539-542, 547, 549, 584-586, 631, 643, 666-682, 690-732, 770, 777-780, 814-815, 820, 865, 900, 912-920, 984, 987-988, 991, 1017 |
| src/lymph/models/unilateral.py   |      264 |       42 |     84% |113, 127, 137, 165-166, 174, 297-306, 322-336, 531-536, 542-547, 587, 658, 688-690, 710, 750, 813, 843, 960-961 |
| src/lymph/types.py               |      129 |       12 |     91% |70, 295, 308, 324, 334, 392, 408, 419, 433, 450, 468, 479 |
| src/lymph/utils.py               |      169 |       28 |     83% |20, 24, 29, 34-38, 101-103, 118-120, 219, 225-232, 240-242, 246-247, 459 |
|                        **TOTAL** | **2127** |  **444** | **79%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/lycosystem/lymph/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/lycosystem/lymph/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/lycosystem/lymph/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/lycosystem/lymph/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Flycosystem%2Flymph%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/lycosystem/lymph/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.