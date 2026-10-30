# Murfy backup (not code; never merge into main)

Hourly copy of the team's two Murfy projects:

- `report/` = Murfy "ANLP final report" (main.tex, XeLaTeX)
- `slides/` = Murfy "ANLP final slides" (slides.tex, XeLaTeX)

The first commit is the projects as uploaded on 30 Oct (built from main 97713cb). Each later commit holds the
files edited on Murfy since the one before, so `git diff <first commit> HEAD` shows every Murfy edit.
To restore a file, copy it from here into Murfy. Paths inside are Murfy's (`artifacts/...`), not the repo's
(`../artifacts/...`).
