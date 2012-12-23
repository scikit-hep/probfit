API Change is coming
dist_fit
--------

Fitting distribution do regression and such in python.

Requirement
-----------

RTMinuit https://github.com/piti118/RTMinuit
numpy
matplotlib

Tutorial
--------

open tutorial.ipynb in ipython notebook(I created it in 1.3dev version though you might have problem if you are using 1.2). If you can't open it just copy cell by cell from tutorial.py.

Caveat
------

There is a name Clash for function call Normalize with matplotlib though so if Normalize doesn't work do

```python
import dist_fit as df
f = df.Normalize(f)
```