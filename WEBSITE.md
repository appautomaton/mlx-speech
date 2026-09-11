# Website ownership

The public website remains at https://appautomaton.com/mlx-speech/.

Edit its independent HTML, styles, scripts, fonts, and recorded media in
[the frontend module](https://github.com/appautomaton/appautomaton.github.io/tree/main/sites/mlx-speech).
The frontend repository builds, checks, and publishes the website, including
its model catalog and discovery files.

This repository owns the speech library, model guides in `docs/`, examples,
and checkpoint conversion tools. Its former `site/` copy and Pages workflow
have been retired. README and model-guide assertions remain in the library's
unit suite; website catalog assertions run in the frontend test suite.
