# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/lycosystem/lymph/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                             |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------- | -------: | -------: | ------: | --------: |
| src/lymph/\_\_init\_\_.py        |       20 |        7 |     65% |     43-49 |
| src/lymph/\_version.py           |       13 |        0 |    100% |           |
| src/lymph/diagnosis\_times.py    |      252 |       45 |     82% |85, 140, 153, 166, 175-176, 187, 243-244, 264-267, 285, 342, 354, 358, 366, 375, 378, 382-386, 418, 437-442, 449-456, 460-465, 469-478 |
| src/lymph/graph.py               |      313 |       47 |     85% |56-57, 104, 120-123, 140, 166, 186, 198-214, 276, 279, 296, 346, 349, 352, 365, 371, 555, 568, 634-638, 703-707, 716-721, 741-743 |
| src/lymph/matrix.py              |       76 |       11 |     86% |172, 180-181, 220-221, 237-251 |
| src/lymph/modalities.py          |      146 |       19 |     87% |29, 45, 51, 67, 70, 83, 86, 106, 120, 123, 126, 129, 132, 195, 215-216, 244, 293-294 |
| src/lymph/models/\_\_init\_\_.py |        5 |        0 |    100% |           |
| src/lymph/models/bilateral.py    |      213 |       35 |     84% |129-131, 141-143, 149, 157, 180, 188, 443-449, 469-472, 484-490, 497-503, 515, 562-563, 567-570, 648, 715, 718-719 |
| src/lymph/models/hpv.py          |      183 |      136 |     26% |29-33, 83-115, 125-139, 149-151, 161-163, 168-171, 176-179, 187-206, 219-224, 235-241, 258-266, 270-287, 291-341, 345-346, 350-351, 365-377, 385-394, 398-407, 436-448, 461, 474, 482, 498-516 |
| src/lymph/models/midline.py      |      384 |       99 |     74% |109, 128, 175, 195, 198, 207, 213, 221, 227, 245, 257, 282-285, 290, 317, 323, 343-346, 426-435, 464-474, 539-542, 547, 549, 585, 587-589, 634, 646, 669-685, 694-732, 770, 777-780, 814-815, 820, 865, 900, 912-920, 984, 987-988, 991, 1017 |
| src/lymph/models/unilateral.py   |      264 |       42 |     84% |113, 127, 137, 165-166, 174, 297-306, 322-336, 532-537, 543-548, 588, 659, 689-691, 711, 751, 814, 844, 961-962 |
| src/lymph/types.py               |      134 |       12 |     91% |70, 295, 308, 324, 334, 397, 413, 424, 438, 455, 473, 484 |
| src/lymph/utils.py               |      169 |       28 |     83% |20, 24, 29, 34-38, 101-103, 118-120, 219, 225-232, 240-242, 246-247, 459 |
| **TOTAL**                        | **2172** |  **481** | **78%** |           |


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