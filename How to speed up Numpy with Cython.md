<h1 style="text-align: center;">How to speed up Numpy with Cython</h1>

# 1. Introduce
**This article will describe how to use Cython to speed up numpy.**

The original code using Nump is as follows, it truncates array_1 with the interval [2, 10] via np.clip. After that, we do some simple arithmetic with array_1, array_2 and a, b and c, and finally return the result of the calculation.


```python
def compute_np(array_1, array_2, a, b, c):
     return np.clip(array_1, 2, 10) * a + array_2 * b + c
```

# 2 Generating random arrays
**This is the prefix for all the code below, please run this prefix before running the program below.**

It will generate two random two-dimensional arrays and set the values of a,b,c. To make it easier to compare the speed of the runs later, the dimension of the arrays generated here is set larger, to (3000x2000).


```python
import numpy as np

array_1 = np.random.uniform(0, 100, size=(3000, 2000)).astype(np.intc)
array_2 = np.random.uniform(0, 100, size=(3000, 2000)).astype(np.intc)
a = 4
b = 3
c = 9
```

# 3. The code for the Nump

**The following code is executed by numpy.**

Here we name the function compute_np. We use timeit here to show how fast it runs, and to be more precise we calculate the average of its ten runs and record this length of time at the end with compute_np_time.


```python
import timeit

def compute_np(array_1, array_2, a, b, c):
     return np.clip(array_1, 2, 10) * a + array_2 * b + c

print(compute_np(array_1, array_2, a, b, c))

compute_np_time = timeit.timeit(lambda: compute_np(array_1, array_2, a, b, c), number=10)/10

print("compute_np execution time:", compute_np_time)
```

    [[ 85 127 241 ... 328  52 160]
     [196 160 343 ... 286 127 280]
     [169 109  94 ... 175 214 222]
     ...
     [253 163 325 ...  94 259 229]
     [133 308 190 ... 325 151 202]
     [178 331 281 ...  52 124 263]]
    compute_np execution time: 0.05866470000000845
    

# 4. The code for pure Python

**For numpy form, which cannot be translated directly into cython form, we first need to expand it into pure python form.**

Expands the above code into a pure python function. It loops through two dimensions for each element in array_1 and finally combines them together. This means that a new object needs to be assigned to each element used.Let's call it compute_py


```python
def clip(a, min_value, max_value):
    return min(max(a, min_value), max_value)

def compute_py(array_1, array_2, a, b, c):

    x_max = array_1.shape[0]
    y_max = array_1.shape[1]

    assert array_1.shape == array_2.shape

    result = np.zeros((x_max, y_max), dtype=array_1.dtype)

    for x in range(x_max):
        for y in range(y_max):
            tmp = clip(array_1[x, y], 2, 10)
            tmp = tmp * a + array_2[x, y] * b
            result[x, y] = tmp + c

    return result

print(compute_py(array_1, array_2, a, b, c))

compute_py_time = timeit.timeit(lambda: compute_py(array_1, array_2, a, b, c), number=10)/10

print("compute_py execution time:", compute_py_time)
```

    [[ 85 127 241 ... 328  52 160]
     [196 160 343 ... 286 127 280]
     [169 109  94 ... 175 214 222]
     ...
     [253 163 325 ...  94 259 229]
     [133 308 190 ... 325 151 202]
     [178 331 281 ...  52 124 263]]
    compute_py execution time: 10.784624809999878
    

## 4.1 Compare

Now, we can compare the speed of these two methods.

It is clear that the pure python is much slower than numpy.


```python
import pandas as pd
from IPython.display import HTML

data = {
    'Methods': ['Numpy', 'Pure Python'],
    'Speed(s)': [compute_np_time, compute_py_time],
    'Percentage(%)': [100, compute_np_time/compute_py_time*100]
}
df = pd.DataFrame(data)

# Creating style functions
def add_border(val):
    return 'border: 1px solid black'

# Applying style functions to data boxes
styled_df = df.style.applymap(add_border)

# Defining CSS styles
table_style = [
    {'selector': 'table', 'props': [('border-collapse', 'collapse')]},
    {'selector': 'th, td', 'props': [('border', '1px solid black')]}
]

# Adding styles to stylised data boxes
styled_df.set_table_styles(table_style)

# Displaying stylised data boxes in Jupyter Notebook
HTML(styled_df.to_html())
```




<style type="text/css">
#T_53ec3 table {
  border-collapse: collapse;
}
#T_53ec3 th {
  border: 1px solid black;
}
#T_53ec3  td {
  border: 1px solid black;
}
#T_53ec3_row0_col0, #T_53ec3_row0_col1, #T_53ec3_row0_col2, #T_53ec3_row1_col0, #T_53ec3_row1_col1, #T_53ec3_row1_col2 {
  border: 1px solid black;
}
</style>
<table id="T_53ec3">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_53ec3_level0_col0" class="col_heading level0 col0" >Methods</th>
      <th id="T_53ec3_level0_col1" class="col_heading level0 col1" >Speed(s)</th>
      <th id="T_53ec3_level0_col2" class="col_heading level0 col2" >Percentage(%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_53ec3_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_53ec3_row0_col0" class="data row0 col0" >Numpy</td>
      <td id="T_53ec3_row0_col1" class="data row0 col1" >0.058665</td>
      <td id="T_53ec3_row0_col2" class="data row0 col2" >100.000000</td>
    </tr>
    <tr>
      <th id="T_53ec3_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_53ec3_row1_col0" class="data row1 col0" >Pure Python</td>
      <td id="T_53ec3_row1_col1" class="data row1 col1" >10.784625</td>
      <td id="T_53ec3_row1_col2" class="data row1 col2" >0.543966</td>
    </tr>
  </tbody>
</table>




# 5. The code for Cython

## 5.1 Cython
Pure Python is also valid Cython code, so the same code can run in cython


```python
%load_ext cython
```


```cython
%%cython
import numpy as np
import timeit

array_1 = np.random.uniform(0, 100, size=(3000, 2000)).astype(np.intc)
array_2 = np.random.uniform(0, 100, size=(3000, 2000)).astype(np.intc)
a = 4
b = 3
c = 9

def clip(a, min_value, max_value):
    return min(max(a, min_value), max_value)

def compute_cy(array_1, array_2, a, b, c):

    x_max = array_1.shape[0]
    y_max = array_1.shape[1]

    assert array_1.shape == array_2.shape

    result = np.zeros((x_max, y_max), dtype=array_1.dtype)

    for x in range(x_max):
        for y in range(y_max):
            tmp = clip(array_1[x, y], 2, 10)
            tmp = tmp * a + array_2[x, y] * b
            result[x, y] = tmp + c

    return result

print(compute_cy(array_1, array_2, a, b, c))

compute_cy_time = timeit.timeit(lambda: compute_cy(array_1, array_2, a, b, c), number=10)/10

print("compute_cy execution time:", compute_cy_time)
```

    [[205  70 121 ... 130  64 304]
     [169 262  85 ... 295 232 181]
     [338 346  79 ... 295 100 139]
     ...
     [241 301 166 ... 193 235 289]
     [120 205 105 ... 223  85 103]
     [178  70  94 ... 109 130  55]]
    compute_cy execution time: 8.321111910000036
    

### 5.1.1 Compare

Because the C code still does exactly what the Python interpreter does, there's not much of a difference in speed.


```python
import pandas as pd

data = {
    'Methods': ['Numpy', 'Pure Python','Cython'],
    'Speed(s)': [compute_np_time, compute_py_time, compute_cy_time],
    'Percentage(%)': [100, compute_np_time/compute_py_time*100,compute_np_time/compute_cy_time*100]
}
df = pd.DataFrame(data)

# Creating style functions
def add_border(val):
    return 'border: 1px solid black'

# Applying style functions to data boxes
styled_df = df.style.applymap(add_border)

# Defining CSS styles
table_style = [
    {'selector': 'table', 'props': [('border-collapse', 'collapse')]},
    {'selector': 'th, td', 'props': [('border', '1px solid black')]}
]

# Adding styles to stylised data boxes
styled_df.set_table_styles(table_style)

# Displaying stylised data boxes in Jupyter Notebook
HTML(styled_df.to_html())
```




<style type="text/css">
#T_873e0 table {
  border-collapse: collapse;
}
#T_873e0 th {
  border: 1px solid black;
}
#T_873e0  td {
  border: 1px solid black;
}
#T_873e0_row0_col0, #T_873e0_row0_col1, #T_873e0_row0_col2, #T_873e0_row1_col0, #T_873e0_row1_col1, #T_873e0_row1_col2, #T_873e0_row2_col0, #T_873e0_row2_col1, #T_873e0_row2_col2 {
  border: 1px solid black;
}
</style>
<table id="T_873e0">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_873e0_level0_col0" class="col_heading level0 col0" >Methods</th>
      <th id="T_873e0_level0_col1" class="col_heading level0 col1" >Speed(s)</th>
      <th id="T_873e0_level0_col2" class="col_heading level0 col2" >Percentage(%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_873e0_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_873e0_row0_col0" class="data row0 col0" >Numpy</td>
      <td id="T_873e0_row0_col1" class="data row0 col1" >0.058665</td>
      <td id="T_873e0_row0_col2" class="data row0 col2" >100.000000</td>
    </tr>
    <tr>
      <th id="T_873e0_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_873e0_row1_col0" class="data row1 col0" >Pure Python</td>
      <td id="T_873e0_row1_col1" class="data row1 col1" >10.784625</td>
      <td id="T_873e0_row1_col2" class="data row1 col2" >0.543966</td>
    </tr>
    <tr>
      <th id="T_873e0_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_873e0_row2_col0" class="data row2 col0" >Cython</td>
      <td id="T_873e0_row2_col1" class="data row2 col1" >8.321112</td>
      <td id="T_873e0_row2_col2" class="data row2 col2" >0.705010</td>
    </tr>
  </tbody>
</table>




### 5.1.2 Check

We can use %%cython -a, to generate html files.

As shown below, if a line is white, it means that the code generated doesn’t interact with Python, so will run as fast as normal C code. The darker the yellow, the more Python interaction there is in that line. Those yellow lines will usually operate on Python objects, raise exceptions, or do other kinds of higher-level operations than what can easily be translated into simple and fast C code.

So we can use it to check our code, and it lets us know what lines to improve so that it can run as fast as C.


```cython
%%cython -a
import numpy as np
import timeit

array_1 = np.random.uniform(0, 100, size=(3000, 2000)).astype(np.intc)
array_2 = np.random.uniform(0, 100, size=(3000, 2000)).astype(np.intc)
a = 4
b = 3
c = 9

def clip(a, min_value, max_value):
    return min(max(a, min_value), max_value)

def compute_cy(array_1, array_2, a, b, c):

    x_max = array_1.shape[0]
    y_max = array_1.shape[1]

    assert array_1.shape == array_2.shape

    result = np.zeros((x_max, y_max), dtype=array_1.dtype)

    for x in range(x_max):
        for y in range(y_max):
            tmp = clip(array_1[x, y], 2, 10)
            tmp = tmp * a + array_2[x, y] * b
            result[x, y] = tmp + c

    return result
```




<!DOCTYPE html>
<!-- Generated by Cython 0.29.33 -->
<html>
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <title>Cython: _cython_magic_c8a0d4053b6e33d70de8541b3a985ec3.pyx</title>
    <style type="text/css">

body.cython { font-family: courier; font-size: 12; }

.cython.tag  {  }
.cython.line { margin: 0em }
.cython.code { font-size: 9; color: #444444; display: none; margin: 0px 0px 0px 8px; border-left: 8px none; }

.cython.line .run { background-color: #B0FFB0; }
.cython.line .mis { background-color: #FFB0B0; }
.cython.code.run  { border-left: 8px solid #B0FFB0; }
.cython.code.mis  { border-left: 8px solid #FFB0B0; }

.cython.code .py_c_api  { color: red; }
.cython.code .py_macro_api  { color: #FF7000; }
.cython.code .pyx_c_api  { color: #FF3000; }
.cython.code .pyx_macro_api  { color: #FF7000; }
.cython.code .refnanny  { color: #FFA000; }
.cython.code .trace  { color: #FFA000; }
.cython.code .error_goto  { color: #FFA000; }

.cython.code .coerce  { color: #008000; border: 1px dotted #008000 }
.cython.code .py_attr { color: #FF0000; font-weight: bold; }
.cython.code .c_attr  { color: #0000FF; }
.cython.code .py_call { color: #FF0000; font-weight: bold; }
.cython.code .c_call  { color: #0000FF; }

.cython.score-0 {background-color: #FFFFff;}
.cython.score-1 {background-color: #FFFFe7;}
.cython.score-2 {background-color: #FFFFd4;}
.cython.score-3 {background-color: #FFFFc4;}
.cython.score-4 {background-color: #FFFFb6;}
.cython.score-5 {background-color: #FFFFaa;}
.cython.score-6 {background-color: #FFFF9f;}
.cython.score-7 {background-color: #FFFF96;}
.cython.score-8 {background-color: #FFFF8d;}
.cython.score-9 {background-color: #FFFF86;}
.cython.score-10 {background-color: #FFFF7f;}
.cython.score-11 {background-color: #FFFF79;}
.cython.score-12 {background-color: #FFFF73;}
.cython.score-13 {background-color: #FFFF6e;}
.cython.score-14 {background-color: #FFFF6a;}
.cython.score-15 {background-color: #FFFF66;}
.cython.score-16 {background-color: #FFFF62;}
.cython.score-17 {background-color: #FFFF5e;}
.cython.score-18 {background-color: #FFFF5b;}
.cython.score-19 {background-color: #FFFF57;}
.cython.score-20 {background-color: #FFFF55;}
.cython.score-21 {background-color: #FFFF52;}
.cython.score-22 {background-color: #FFFF4f;}
.cython.score-23 {background-color: #FFFF4d;}
.cython.score-24 {background-color: #FFFF4b;}
.cython.score-25 {background-color: #FFFF48;}
.cython.score-26 {background-color: #FFFF46;}
.cython.score-27 {background-color: #FFFF44;}
.cython.score-28 {background-color: #FFFF43;}
.cython.score-29 {background-color: #FFFF41;}
.cython.score-30 {background-color: #FFFF3f;}
.cython.score-31 {background-color: #FFFF3e;}
.cython.score-32 {background-color: #FFFF3c;}
.cython.score-33 {background-color: #FFFF3b;}
.cython.score-34 {background-color: #FFFF39;}
.cython.score-35 {background-color: #FFFF38;}
.cython.score-36 {background-color: #FFFF37;}
.cython.score-37 {background-color: #FFFF36;}
.cython.score-38 {background-color: #FFFF35;}
.cython.score-39 {background-color: #FFFF34;}
.cython.score-40 {background-color: #FFFF33;}
.cython.score-41 {background-color: #FFFF32;}
.cython.score-42 {background-color: #FFFF31;}
.cython.score-43 {background-color: #FFFF30;}
.cython.score-44 {background-color: #FFFF2f;}
.cython.score-45 {background-color: #FFFF2e;}
.cython.score-46 {background-color: #FFFF2d;}
.cython.score-47 {background-color: #FFFF2c;}
.cython.score-48 {background-color: #FFFF2b;}
.cython.score-49 {background-color: #FFFF2b;}
.cython.score-50 {background-color: #FFFF2a;}
.cython.score-51 {background-color: #FFFF29;}
.cython.score-52 {background-color: #FFFF29;}
.cython.score-53 {background-color: #FFFF28;}
.cython.score-54 {background-color: #FFFF27;}
.cython.score-55 {background-color: #FFFF27;}
.cython.score-56 {background-color: #FFFF26;}
.cython.score-57 {background-color: #FFFF26;}
.cython.score-58 {background-color: #FFFF25;}
.cython.score-59 {background-color: #FFFF24;}
.cython.score-60 {background-color: #FFFF24;}
.cython.score-61 {background-color: #FFFF23;}
.cython.score-62 {background-color: #FFFF23;}
.cython.score-63 {background-color: #FFFF22;}
.cython.score-64 {background-color: #FFFF22;}
.cython.score-65 {background-color: #FFFF22;}
.cython.score-66 {background-color: #FFFF21;}
.cython.score-67 {background-color: #FFFF21;}
.cython.score-68 {background-color: #FFFF20;}
.cython.score-69 {background-color: #FFFF20;}
.cython.score-70 {background-color: #FFFF1f;}
.cython.score-71 {background-color: #FFFF1f;}
.cython.score-72 {background-color: #FFFF1f;}
.cython.score-73 {background-color: #FFFF1e;}
.cython.score-74 {background-color: #FFFF1e;}
.cython.score-75 {background-color: #FFFF1e;}
.cython.score-76 {background-color: #FFFF1d;}
.cython.score-77 {background-color: #FFFF1d;}
.cython.score-78 {background-color: #FFFF1c;}
.cython.score-79 {background-color: #FFFF1c;}
.cython.score-80 {background-color: #FFFF1c;}
.cython.score-81 {background-color: #FFFF1c;}
.cython.score-82 {background-color: #FFFF1b;}
.cython.score-83 {background-color: #FFFF1b;}
.cython.score-84 {background-color: #FFFF1b;}
.cython.score-85 {background-color: #FFFF1a;}
.cython.score-86 {background-color: #FFFF1a;}
.cython.score-87 {background-color: #FFFF1a;}
.cython.score-88 {background-color: #FFFF1a;}
.cython.score-89 {background-color: #FFFF19;}
.cython.score-90 {background-color: #FFFF19;}
.cython.score-91 {background-color: #FFFF19;}
.cython.score-92 {background-color: #FFFF19;}
.cython.score-93 {background-color: #FFFF18;}
.cython.score-94 {background-color: #FFFF18;}
.cython.score-95 {background-color: #FFFF18;}
.cython.score-96 {background-color: #FFFF18;}
.cython.score-97 {background-color: #FFFF17;}
.cython.score-98 {background-color: #FFFF17;}
.cython.score-99 {background-color: #FFFF17;}
.cython.score-100 {background-color: #FFFF17;}
.cython.score-101 {background-color: #FFFF16;}
.cython.score-102 {background-color: #FFFF16;}
.cython.score-103 {background-color: #FFFF16;}
.cython.score-104 {background-color: #FFFF16;}
.cython.score-105 {background-color: #FFFF16;}
.cython.score-106 {background-color: #FFFF15;}
.cython.score-107 {background-color: #FFFF15;}
.cython.score-108 {background-color: #FFFF15;}
.cython.score-109 {background-color: #FFFF15;}
.cython.score-110 {background-color: #FFFF15;}
.cython.score-111 {background-color: #FFFF15;}
.cython.score-112 {background-color: #FFFF14;}
.cython.score-113 {background-color: #FFFF14;}
.cython.score-114 {background-color: #FFFF14;}
.cython.score-115 {background-color: #FFFF14;}
.cython.score-116 {background-color: #FFFF14;}
.cython.score-117 {background-color: #FFFF14;}
.cython.score-118 {background-color: #FFFF13;}
.cython.score-119 {background-color: #FFFF13;}
.cython.score-120 {background-color: #FFFF13;}
.cython.score-121 {background-color: #FFFF13;}
.cython.score-122 {background-color: #FFFF13;}
.cython.score-123 {background-color: #FFFF13;}
.cython.score-124 {background-color: #FFFF13;}
.cython.score-125 {background-color: #FFFF12;}
.cython.score-126 {background-color: #FFFF12;}
.cython.score-127 {background-color: #FFFF12;}
.cython.score-128 {background-color: #FFFF12;}
.cython.score-129 {background-color: #FFFF12;}
.cython.score-130 {background-color: #FFFF12;}
.cython.score-131 {background-color: #FFFF12;}
.cython.score-132 {background-color: #FFFF11;}
.cython.score-133 {background-color: #FFFF11;}
.cython.score-134 {background-color: #FFFF11;}
.cython.score-135 {background-color: #FFFF11;}
.cython.score-136 {background-color: #FFFF11;}
.cython.score-137 {background-color: #FFFF11;}
.cython.score-138 {background-color: #FFFF11;}
.cython.score-139 {background-color: #FFFF11;}
.cython.score-140 {background-color: #FFFF11;}
.cython.score-141 {background-color: #FFFF10;}
.cython.score-142 {background-color: #FFFF10;}
.cython.score-143 {background-color: #FFFF10;}
.cython.score-144 {background-color: #FFFF10;}
.cython.score-145 {background-color: #FFFF10;}
.cython.score-146 {background-color: #FFFF10;}
.cython.score-147 {background-color: #FFFF10;}
.cython.score-148 {background-color: #FFFF10;}
.cython.score-149 {background-color: #FFFF10;}
.cython.score-150 {background-color: #FFFF0f;}
.cython.score-151 {background-color: #FFFF0f;}
.cython.score-152 {background-color: #FFFF0f;}
.cython.score-153 {background-color: #FFFF0f;}
.cython.score-154 {background-color: #FFFF0f;}
.cython.score-155 {background-color: #FFFF0f;}
.cython.score-156 {background-color: #FFFF0f;}
.cython.score-157 {background-color: #FFFF0f;}
.cython.score-158 {background-color: #FFFF0f;}
.cython.score-159 {background-color: #FFFF0f;}
.cython.score-160 {background-color: #FFFF0f;}
.cython.score-161 {background-color: #FFFF0e;}
.cython.score-162 {background-color: #FFFF0e;}
.cython.score-163 {background-color: #FFFF0e;}
.cython.score-164 {background-color: #FFFF0e;}
.cython.score-165 {background-color: #FFFF0e;}
.cython.score-166 {background-color: #FFFF0e;}
.cython.score-167 {background-color: #FFFF0e;}
.cython.score-168 {background-color: #FFFF0e;}
.cython.score-169 {background-color: #FFFF0e;}
.cython.score-170 {background-color: #FFFF0e;}
.cython.score-171 {background-color: #FFFF0e;}
.cython.score-172 {background-color: #FFFF0e;}
.cython.score-173 {background-color: #FFFF0d;}
.cython.score-174 {background-color: #FFFF0d;}
.cython.score-175 {background-color: #FFFF0d;}
.cython.score-176 {background-color: #FFFF0d;}
.cython.score-177 {background-color: #FFFF0d;}
.cython.score-178 {background-color: #FFFF0d;}
.cython.score-179 {background-color: #FFFF0d;}
.cython.score-180 {background-color: #FFFF0d;}
.cython.score-181 {background-color: #FFFF0d;}
.cython.score-182 {background-color: #FFFF0d;}
.cython.score-183 {background-color: #FFFF0d;}
.cython.score-184 {background-color: #FFFF0d;}
.cython.score-185 {background-color: #FFFF0d;}
.cython.score-186 {background-color: #FFFF0d;}
.cython.score-187 {background-color: #FFFF0c;}
.cython.score-188 {background-color: #FFFF0c;}
.cython.score-189 {background-color: #FFFF0c;}
.cython.score-190 {background-color: #FFFF0c;}
.cython.score-191 {background-color: #FFFF0c;}
.cython.score-192 {background-color: #FFFF0c;}
.cython.score-193 {background-color: #FFFF0c;}
.cython.score-194 {background-color: #FFFF0c;}
.cython.score-195 {background-color: #FFFF0c;}
.cython.score-196 {background-color: #FFFF0c;}
.cython.score-197 {background-color: #FFFF0c;}
.cython.score-198 {background-color: #FFFF0c;}
.cython.score-199 {background-color: #FFFF0c;}
.cython.score-200 {background-color: #FFFF0c;}
.cython.score-201 {background-color: #FFFF0c;}
.cython.score-202 {background-color: #FFFF0c;}
.cython.score-203 {background-color: #FFFF0b;}
.cython.score-204 {background-color: #FFFF0b;}
.cython.score-205 {background-color: #FFFF0b;}
.cython.score-206 {background-color: #FFFF0b;}
.cython.score-207 {background-color: #FFFF0b;}
.cython.score-208 {background-color: #FFFF0b;}
.cython.score-209 {background-color: #FFFF0b;}
.cython.score-210 {background-color: #FFFF0b;}
.cython.score-211 {background-color: #FFFF0b;}
.cython.score-212 {background-color: #FFFF0b;}
.cython.score-213 {background-color: #FFFF0b;}
.cython.score-214 {background-color: #FFFF0b;}
.cython.score-215 {background-color: #FFFF0b;}
.cython.score-216 {background-color: #FFFF0b;}
.cython.score-217 {background-color: #FFFF0b;}
.cython.score-218 {background-color: #FFFF0b;}
.cython.score-219 {background-color: #FFFF0b;}
.cython.score-220 {background-color: #FFFF0b;}
.cython.score-221 {background-color: #FFFF0b;}
.cython.score-222 {background-color: #FFFF0a;}
.cython.score-223 {background-color: #FFFF0a;}
.cython.score-224 {background-color: #FFFF0a;}
.cython.score-225 {background-color: #FFFF0a;}
.cython.score-226 {background-color: #FFFF0a;}
.cython.score-227 {background-color: #FFFF0a;}
.cython.score-228 {background-color: #FFFF0a;}
.cython.score-229 {background-color: #FFFF0a;}
.cython.score-230 {background-color: #FFFF0a;}
.cython.score-231 {background-color: #FFFF0a;}
.cython.score-232 {background-color: #FFFF0a;}
.cython.score-233 {background-color: #FFFF0a;}
.cython.score-234 {background-color: #FFFF0a;}
.cython.score-235 {background-color: #FFFF0a;}
.cython.score-236 {background-color: #FFFF0a;}
.cython.score-237 {background-color: #FFFF0a;}
.cython.score-238 {background-color: #FFFF0a;}
.cython.score-239 {background-color: #FFFF0a;}
.cython.score-240 {background-color: #FFFF0a;}
.cython.score-241 {background-color: #FFFF0a;}
.cython.score-242 {background-color: #FFFF0a;}
.cython.score-243 {background-color: #FFFF0a;}
.cython.score-244 {background-color: #FFFF0a;}
.cython.score-245 {background-color: #FFFF0a;}
.cython.score-246 {background-color: #FFFF09;}
.cython.score-247 {background-color: #FFFF09;}
.cython.score-248 {background-color: #FFFF09;}
.cython.score-249 {background-color: #FFFF09;}
.cython.score-250 {background-color: #FFFF09;}
.cython.score-251 {background-color: #FFFF09;}
.cython.score-252 {background-color: #FFFF09;}
.cython.score-253 {background-color: #FFFF09;}
.cython.score-254 {background-color: #FFFF09;}
pre { line-height: 125%; }
td.linenos .normal { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
span.linenos { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
td.linenos .special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
span.linenos.special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
.cython .hll { background-color: #ffffcc }
.cython { background: #f8f8f8; }
.cython .c { color: #3D7B7B; font-style: italic } /* Comment */
.cython .err { border: 1px solid #FF0000 } /* Error */
.cython .k { color: #008000; font-weight: bold } /* Keyword */
.cython .o { color: #666666 } /* Operator */
.cython .ch { color: #3D7B7B; font-style: italic } /* Comment.Hashbang */
.cython .cm { color: #3D7B7B; font-style: italic } /* Comment.Multiline */
.cython .cp { color: #9C6500 } /* Comment.Preproc */
.cython .cpf { color: #3D7B7B; font-style: italic } /* Comment.PreprocFile */
.cython .c1 { color: #3D7B7B; font-style: italic } /* Comment.Single */
.cython .cs { color: #3D7B7B; font-style: italic } /* Comment.Special */
.cython .gd { color: #A00000 } /* Generic.Deleted */
.cython .ge { font-style: italic } /* Generic.Emph */
.cython .gr { color: #E40000 } /* Generic.Error */
.cython .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.cython .gi { color: #008400 } /* Generic.Inserted */
.cython .go { color: #717171 } /* Generic.Output */
.cython .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.cython .gs { font-weight: bold } /* Generic.Strong */
.cython .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.cython .gt { color: #0044DD } /* Generic.Traceback */
.cython .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.cython .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.cython .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.cython .kp { color: #008000 } /* Keyword.Pseudo */
.cython .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.cython .kt { color: #B00040 } /* Keyword.Type */
.cython .m { color: #666666 } /* Literal.Number */
.cython .s { color: #BA2121 } /* Literal.String */
.cython .na { color: #687822 } /* Name.Attribute */
.cython .nb { color: #008000 } /* Name.Builtin */
.cython .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.cython .no { color: #880000 } /* Name.Constant */
.cython .nd { color: #AA22FF } /* Name.Decorator */
.cython .ni { color: #717171; font-weight: bold } /* Name.Entity */
.cython .ne { color: #CB3F38; font-weight: bold } /* Name.Exception */
.cython .nf { color: #0000FF } /* Name.Function */
.cython .nl { color: #767600 } /* Name.Label */
.cython .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.cython .nt { color: #008000; font-weight: bold } /* Name.Tag */
.cython .nv { color: #19177C } /* Name.Variable */
.cython .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.cython .w { color: #bbbbbb } /* Text.Whitespace */
.cython .mb { color: #666666 } /* Literal.Number.Bin */
.cython .mf { color: #666666 } /* Literal.Number.Float */
.cython .mh { color: #666666 } /* Literal.Number.Hex */
.cython .mi { color: #666666 } /* Literal.Number.Integer */
.cython .mo { color: #666666 } /* Literal.Number.Oct */
.cython .sa { color: #BA2121 } /* Literal.String.Affix */
.cython .sb { color: #BA2121 } /* Literal.String.Backtick */
.cython .sc { color: #BA2121 } /* Literal.String.Char */
.cython .dl { color: #BA2121 } /* Literal.String.Delimiter */
.cython .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.cython .s2 { color: #BA2121 } /* Literal.String.Double */
.cython .se { color: #AA5D1F; font-weight: bold } /* Literal.String.Escape */
.cython .sh { color: #BA2121 } /* Literal.String.Heredoc */
.cython .si { color: #A45A77; font-weight: bold } /* Literal.String.Interpol */
.cython .sx { color: #008000 } /* Literal.String.Other */
.cython .sr { color: #A45A77 } /* Literal.String.Regex */
.cython .s1 { color: #BA2121 } /* Literal.String.Single */
.cython .ss { color: #19177C } /* Literal.String.Symbol */
.cython .bp { color: #008000 } /* Name.Builtin.Pseudo */
.cython .fm { color: #0000FF } /* Name.Function.Magic */
.cython .vc { color: #19177C } /* Name.Variable.Class */
.cython .vg { color: #19177C } /* Name.Variable.Global */
.cython .vi { color: #19177C } /* Name.Variable.Instance */
.cython .vm { color: #19177C } /* Name.Variable.Magic */
.cython .il { color: #666666 } /* Literal.Number.Integer.Long */
    </style>
</head>
<body class="cython">
<p><span style="border-bottom: solid 1px grey;">Generated by Cython 0.29.33</span></p>
<p>
    <span style="background-color: #FFFF00">Yellow lines</span> hint at Python interaction.<br />
    Click on a line that starts with a "<code>+</code>" to see the C code that Cython generated for it.
</p>
<div class="cython"><pre class="cython line score-8" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">01</span>: <span class="k">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span></pre>
<pre class='cython code score-8 '>  __pyx_t_1 = <span class='pyx_c_api'>__Pyx_Import</span>(__pyx_n_s_numpy, 0, 0);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 1, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_np, __pyx_t_1) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 1, __pyx_L1_error)</span>
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
</pre><pre class="cython line score-8" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">02</span>: <span class="k">import</span> <span class="nn">timeit</span></pre>
<pre class='cython code score-8 '>  __pyx_t_1 = <span class='pyx_c_api'>__Pyx_Import</span>(__pyx_n_s_timeit, 0, 0);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 2, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_timeit, __pyx_t_1) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 2, __pyx_L1_error)</span>
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
</pre><pre class="cython line score-0">&#xA0;<span class="">03</span>: </pre>
<pre class="cython line score-47" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">04</span>: <span class="n">array_1</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">uniform</span><span class="p">(</span><span class="mf">0</span><span class="p">,</span> <span class="mf">100</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">(</span><span class="mf">3000</span><span class="p">,</span> <span class="mf">2000</span><span class="p">))</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">intc</span><span class="p">)</span></pre>
<pre class='cython code score-47 '>  <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_t_1, __pyx_n_s_np);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 4, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  __pyx_t_2 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_t_1, __pyx_n_s_random);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 4, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_t_2, __pyx_n_s_uniform);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 4, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
  __pyx_t_2 = <span class='pyx_c_api'>__Pyx_PyDict_NewPresized</span>(1);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 4, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
/* … */
  __pyx_tuple_ = <span class='py_c_api'>PyTuple_Pack</span>(2, __pyx_int_0, __pyx_int_100);<span class='error_goto'> if (unlikely(!__pyx_tuple_)) __PYX_ERR(0, 4, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_tuple_);
  <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_tuple_);
  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_t_2, __pyx_n_s_size, __pyx_tuple__2) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 4, __pyx_L1_error)</span>
  __pyx_t_3 = <span class='pyx_c_api'>__Pyx_PyObject_Call</span>(__pyx_t_1, __pyx_tuple_, __pyx_t_2);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 4, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
  __pyx_t_2 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_t_3, __pyx_n_s_astype);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 4, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
  <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_t_3, __pyx_n_s_np);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 4, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  __pyx_t_1 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_t_3, __pyx_n_s_intc);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 4, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
  __pyx_t_3 = <span class='pyx_c_api'>__Pyx_PyObject_CallOneArg</span>(__pyx_t_2, __pyx_t_1);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 4, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_array_1, __pyx_t_3) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 4, __pyx_L1_error)</span>
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
  __pyx_tuple__2 = <span class='py_c_api'>PyTuple_Pack</span>(2, __pyx_int_3000, __pyx_int_2000);<span class='error_goto'> if (unlikely(!__pyx_tuple__2)) __PYX_ERR(0, 4, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_tuple__2);
  <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_tuple__2);
</pre><pre class="cython line score-37" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">05</span>: <span class="n">array_2</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">uniform</span><span class="p">(</span><span class="mf">0</span><span class="p">,</span> <span class="mf">100</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">(</span><span class="mf">3000</span><span class="p">,</span> <span class="mf">2000</span><span class="p">))</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">intc</span><span class="p">)</span></pre>
<pre class='cython code score-37 '>  <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_t_3, __pyx_n_s_np);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 5, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  __pyx_t_1 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_t_3, __pyx_n_s_random);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 5, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
  __pyx_t_3 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_t_1, __pyx_n_s_uniform);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 5, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = <span class='pyx_c_api'>__Pyx_PyDict_NewPresized</span>(1);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 5, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_t_1, __pyx_n_s_size, __pyx_tuple__2) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 5, __pyx_L1_error)</span>
  __pyx_t_2 = <span class='pyx_c_api'>__Pyx_PyObject_Call</span>(__pyx_t_3, __pyx_tuple_, __pyx_t_1);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 5, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_t_2, __pyx_n_s_astype);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 5, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
  <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_t_2, __pyx_n_s_np);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 5, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
  __pyx_t_3 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_t_2, __pyx_n_s_intc);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 5, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
  __pyx_t_2 = <span class='pyx_c_api'>__Pyx_PyObject_CallOneArg</span>(__pyx_t_1, __pyx_t_3);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 5, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_array_2, __pyx_t_2) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 5, __pyx_L1_error)</span>
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
</pre><pre class="cython line score-5" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">06</span>: <span class="n">a</span> <span class="o">=</span> <span class="mf">4</span></pre>
<pre class='cython code score-5 '>  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_a, __pyx_int_4) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 6, __pyx_L1_error)</span>
</pre><pre class="cython line score-5" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">07</span>: <span class="n">b</span> <span class="o">=</span> <span class="mf">3</span></pre>
<pre class='cython code score-5 '>  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_b, __pyx_int_3) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 7, __pyx_L1_error)</span>
</pre><pre class="cython line score-5" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">08</span>: <span class="n">c</span> <span class="o">=</span> <span class="mf">9</span></pre>
<pre class='cython code score-5 '>  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_c, __pyx_int_9) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 8, __pyx_L1_error)</span>
</pre><pre class="cython line score-0">&#xA0;<span class="">09</span>: </pre>
<pre class="cython line score-50" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">10</span>: <span class="k">def</span> <span class="nf">clip</span><span class="p">(</span><span class="n">a</span><span class="p">,</span> <span class="n">min_value</span><span class="p">,</span> <span class="n">max_value</span><span class="p">):</span></pre>
<pre class='cython code score-50 '>/* Python wrapper */
static PyObject *__pyx_pw_46_cython_magic_c8a0d4053b6e33d70de8541b3a985ec3_1clip(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyMethodDef __pyx_mdef_46_cython_magic_c8a0d4053b6e33d70de8541b3a985ec3_1clip = {"clip", (PyCFunction)(void*)(PyCFunctionWithKeywords)__pyx_pw_46_cython_magic_c8a0d4053b6e33d70de8541b3a985ec3_1clip, METH_VARARGS|METH_KEYWORDS, 0};
static PyObject *__pyx_pw_46_cython_magic_c8a0d4053b6e33d70de8541b3a985ec3_1clip(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_a = 0;
  PyObject *__pyx_v_min_value = 0;
  PyObject *__pyx_v_max_value = 0;
  PyObject *__pyx_r = 0;
  <span class='refnanny'>__Pyx_RefNannyDeclarations</span>
  <span class='refnanny'>__Pyx_RefNannySetupContext</span>("clip (wrapper)", 0);
  {
    static PyObject **__pyx_pyargnames[] = {&amp;__pyx_n_s_a,&amp;__pyx_n_s_min_value,&amp;__pyx_n_s_max_value,0};
    PyObject* values[3] = {0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      const Py_ssize_t pos_args = <span class='py_macro_api'>PyTuple_GET_SIZE</span>(__pyx_args);
      switch (pos_args) {
        case  3: values[2] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 2);
        CYTHON_FALLTHROUGH;
        case  2: values[1] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 1);
        CYTHON_FALLTHROUGH;
        case  1: values[0] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 0);
        CYTHON_FALLTHROUGH;
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = <span class='py_c_api'>PyDict_Size</span>(__pyx_kwds);
      switch (pos_args) {
        case  0:
        if (likely((values[0] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_a)) != 0)) kw_args--;
        else goto __pyx_L5_argtuple_error;
        CYTHON_FALLTHROUGH;
        case  1:
        if (likely((values[1] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_min_value)) != 0)) kw_args--;
        else {
          <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("clip", 1, 3, 3, 1); <span class='error_goto'>__PYX_ERR(0, 10, __pyx_L3_error)</span>
        }
        CYTHON_FALLTHROUGH;
        case  2:
        if (likely((values[2] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_max_value)) != 0)) kw_args--;
        else {
          <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("clip", 1, 3, 3, 2); <span class='error_goto'>__PYX_ERR(0, 10, __pyx_L3_error)</span>
        }
      }
      if (unlikely(kw_args &gt; 0)) {
        if (unlikely(<span class='pyx_c_api'>__Pyx_ParseOptionalKeywords</span>(__pyx_kwds, __pyx_pyargnames, 0, values, pos_args, "clip") &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 10, __pyx_L3_error)</span>
      }
    } else if (<span class='py_macro_api'>PyTuple_GET_SIZE</span>(__pyx_args) != 3) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 0);
      values[1] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 1);
      values[2] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 2);
    }
    __pyx_v_a = values[0];
    __pyx_v_min_value = values[1];
    __pyx_v_max_value = values[2];
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("clip", 1, 3, 3, <span class='py_macro_api'>PyTuple_GET_SIZE</span>(__pyx_args)); <span class='error_goto'>__PYX_ERR(0, 10, __pyx_L3_error)</span>
  __pyx_L3_error:;
  <span class='pyx_c_api'>__Pyx_AddTraceback</span>("_cython_magic_c8a0d4053b6e33d70de8541b3a985ec3.clip", __pyx_clineno, __pyx_lineno, __pyx_filename);
  <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __pyx_r = __pyx_pf_46_cython_magic_c8a0d4053b6e33d70de8541b3a985ec3_clip(__pyx_self, __pyx_v_a, __pyx_v_min_value, __pyx_v_max_value);
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;

  /* function exit code */
  <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
  return __pyx_r;
}

static PyObject *__pyx_pf_46_cython_magic_c8a0d4053b6e33d70de8541b3a985ec3_clip(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_a, PyObject *__pyx_v_min_value, PyObject *__pyx_v_max_value) {
  PyObject *__pyx_r = NULL;
  <span class='refnanny'>__Pyx_RefNannyDeclarations</span>
  <span class='refnanny'>__Pyx_RefNannySetupContext</span>("clip", 0);
/* … */
  /* function exit code */
  __pyx_L1_error:;
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_1);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_2);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_3);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_4);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_5);
  <span class='pyx_c_api'>__Pyx_AddTraceback</span>("_cython_magic_c8a0d4053b6e33d70de8541b3a985ec3.clip", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  <span class='refnanny'>__Pyx_XGIVEREF</span>(__pyx_r);
  <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
  return __pyx_r;
}
/* … */
  __pyx_tuple__3 = <span class='py_c_api'>PyTuple_Pack</span>(3, __pyx_n_s_a, __pyx_n_s_min_value, __pyx_n_s_max_value);<span class='error_goto'> if (unlikely(!__pyx_tuple__3)) __PYX_ERR(0, 10, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_tuple__3);
  <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_tuple__3);
/* … */
  __pyx_t_2 = PyCFunction_NewEx(&amp;__pyx_mdef_46_cython_magic_c8a0d4053b6e33d70de8541b3a985ec3_1clip, NULL, __pyx_n_s_cython_magic_c8a0d4053b6e33d70d);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 10, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_clip, __pyx_t_2) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 10, __pyx_L1_error)</span>
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
  __pyx_codeobj__4 = (PyObject*)<span class='pyx_c_api'>__Pyx_PyCode_New</span>(3, 0, 3, 0, CO_OPTIMIZED|CO_NEWLOCALS, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple__3, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_C_Users_DELL_ipython_cython__cyt, __pyx_n_s_clip, 10, __pyx_empty_bytes);<span class='error_goto'> if (unlikely(!__pyx_codeobj__4)) __PYX_ERR(0, 10, __pyx_L1_error)</span>
</pre><pre class="cython line score-32" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">11</span>:     <span class="k">return</span> <span class="nb">min</span><span class="p">(</span><span class="nb">max</span><span class="p">(</span><span class="n">a</span><span class="p">,</span> <span class="n">min_value</span><span class="p">),</span> <span class="n">max_value</span><span class="p">)</span></pre>
<pre class='cython code score-32 '>  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_r);
  <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_v_max_value);
  __pyx_t_1 = __pyx_v_max_value;
  <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_v_min_value);
  __pyx_t_2 = __pyx_v_min_value;
  <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_v_a);
  __pyx_t_3 = __pyx_v_a;
  __pyx_t_5 = <span class='py_c_api'>PyObject_RichCompare</span>(__pyx_t_2, __pyx_t_3, Py_GT); <span class='refnanny'>__Pyx_XGOTREF</span>(__pyx_t_5);<span class='error_goto'> if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 11, __pyx_L1_error)</span>
  __pyx_t_6 = <span class='pyx_c_api'>__Pyx_PyObject_IsTrue</span>(__pyx_t_5); if (unlikely(__pyx_t_6 &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 11, __pyx_L1_error)</span>
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_5); __pyx_t_5 = 0;
  if (__pyx_t_6) {
    <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_t_2);
    __pyx_t_4 = __pyx_t_2;
  } else {
    <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_t_3);
    __pyx_t_4 = __pyx_t_3;
  }
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
  <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_t_4);
  __pyx_t_2 = __pyx_t_4;
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_4); __pyx_t_4 = 0;
  __pyx_t_3 = <span class='py_c_api'>PyObject_RichCompare</span>(__pyx_t_1, __pyx_t_2, Py_LT); <span class='refnanny'>__Pyx_XGOTREF</span>(__pyx_t_3);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 11, __pyx_L1_error)</span>
  __pyx_t_6 = <span class='pyx_c_api'>__Pyx_PyObject_IsTrue</span>(__pyx_t_3); if (unlikely(__pyx_t_6 &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 11, __pyx_L1_error)</span>
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
  if (__pyx_t_6) {
    <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_t_1);
    __pyx_t_4 = __pyx_t_1;
  } else {
    <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_t_2);
    __pyx_t_4 = __pyx_t_2;
  }
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
  <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_t_4);
  __pyx_r = __pyx_t_4;
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_4); __pyx_t_4 = 0;
  goto __pyx_L0;
</pre><pre class="cython line score-0">&#xA0;<span class="">12</span>: </pre>
<pre class="cython line score-68" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">13</span>: <span class="k">def</span> <span class="nf">compute_cy</span><span class="p">(</span><span class="n">array_1</span><span class="p">,</span> <span class="n">array_2</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">c</span><span class="p">):</span></pre>
<pre class='cython code score-68 '>/* Python wrapper */
static PyObject *__pyx_pw_46_cython_magic_c8a0d4053b6e33d70de8541b3a985ec3_3compute_cy(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyMethodDef __pyx_mdef_46_cython_magic_c8a0d4053b6e33d70de8541b3a985ec3_3compute_cy = {"compute_cy", (PyCFunction)(void*)(PyCFunctionWithKeywords)__pyx_pw_46_cython_magic_c8a0d4053b6e33d70de8541b3a985ec3_3compute_cy, METH_VARARGS|METH_KEYWORDS, 0};
static PyObject *__pyx_pw_46_cython_magic_c8a0d4053b6e33d70de8541b3a985ec3_3compute_cy(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_array_1 = 0;
  PyObject *__pyx_v_array_2 = 0;
  PyObject *__pyx_v_a = 0;
  PyObject *__pyx_v_b = 0;
  PyObject *__pyx_v_c = 0;
  PyObject *__pyx_r = 0;
  <span class='refnanny'>__Pyx_RefNannyDeclarations</span>
  <span class='refnanny'>__Pyx_RefNannySetupContext</span>("compute_cy (wrapper)", 0);
  {
    static PyObject **__pyx_pyargnames[] = {&amp;__pyx_n_s_array_1,&amp;__pyx_n_s_array_2,&amp;__pyx_n_s_a,&amp;__pyx_n_s_b,&amp;__pyx_n_s_c,0};
    PyObject* values[5] = {0,0,0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      const Py_ssize_t pos_args = <span class='py_macro_api'>PyTuple_GET_SIZE</span>(__pyx_args);
      switch (pos_args) {
        case  5: values[4] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 4);
        CYTHON_FALLTHROUGH;
        case  4: values[3] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 3);
        CYTHON_FALLTHROUGH;
        case  3: values[2] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 2);
        CYTHON_FALLTHROUGH;
        case  2: values[1] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 1);
        CYTHON_FALLTHROUGH;
        case  1: values[0] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 0);
        CYTHON_FALLTHROUGH;
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = <span class='py_c_api'>PyDict_Size</span>(__pyx_kwds);
      switch (pos_args) {
        case  0:
        if (likely((values[0] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_array_1)) != 0)) kw_args--;
        else goto __pyx_L5_argtuple_error;
        CYTHON_FALLTHROUGH;
        case  1:
        if (likely((values[1] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_array_2)) != 0)) kw_args--;
        else {
          <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("compute_cy", 1, 5, 5, 1); <span class='error_goto'>__PYX_ERR(0, 13, __pyx_L3_error)</span>
        }
        CYTHON_FALLTHROUGH;
        case  2:
        if (likely((values[2] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_a)) != 0)) kw_args--;
        else {
          <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("compute_cy", 1, 5, 5, 2); <span class='error_goto'>__PYX_ERR(0, 13, __pyx_L3_error)</span>
        }
        CYTHON_FALLTHROUGH;
        case  3:
        if (likely((values[3] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_b)) != 0)) kw_args--;
        else {
          <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("compute_cy", 1, 5, 5, 3); <span class='error_goto'>__PYX_ERR(0, 13, __pyx_L3_error)</span>
        }
        CYTHON_FALLTHROUGH;
        case  4:
        if (likely((values[4] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_c)) != 0)) kw_args--;
        else {
          <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("compute_cy", 1, 5, 5, 4); <span class='error_goto'>__PYX_ERR(0, 13, __pyx_L3_error)</span>
        }
      }
      if (unlikely(kw_args &gt; 0)) {
        if (unlikely(<span class='pyx_c_api'>__Pyx_ParseOptionalKeywords</span>(__pyx_kwds, __pyx_pyargnames, 0, values, pos_args, "compute_cy") &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 13, __pyx_L3_error)</span>
      }
    } else if (<span class='py_macro_api'>PyTuple_GET_SIZE</span>(__pyx_args) != 5) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 0);
      values[1] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 1);
      values[2] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 2);
      values[3] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 3);
      values[4] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 4);
    }
    __pyx_v_array_1 = values[0];
    __pyx_v_array_2 = values[1];
    __pyx_v_a = values[2];
    __pyx_v_b = values[3];
    __pyx_v_c = values[4];
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("compute_cy", 1, 5, 5, <span class='py_macro_api'>PyTuple_GET_SIZE</span>(__pyx_args)); <span class='error_goto'>__PYX_ERR(0, 13, __pyx_L3_error)</span>
  __pyx_L3_error:;
  <span class='pyx_c_api'>__Pyx_AddTraceback</span>("_cython_magic_c8a0d4053b6e33d70de8541b3a985ec3.compute_cy", __pyx_clineno, __pyx_lineno, __pyx_filename);
  <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __pyx_r = __pyx_pf_46_cython_magic_c8a0d4053b6e33d70de8541b3a985ec3_2compute_cy(__pyx_self, __pyx_v_array_1, __pyx_v_array_2, __pyx_v_a, __pyx_v_b, __pyx_v_c);
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;

  /* function exit code */
  <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
  return __pyx_r;
}

static PyObject *__pyx_pf_46_cython_magic_c8a0d4053b6e33d70de8541b3a985ec3_2compute_cy(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_array_1, PyObject *__pyx_v_array_2, PyObject *__pyx_v_a, PyObject *__pyx_v_b, PyObject *__pyx_v_c) {
  PyObject *__pyx_v_x_max = NULL;
  PyObject *__pyx_v_y_max = NULL;
  PyObject *__pyx_v_result = NULL;
  PyObject *__pyx_v_x = NULL;
  PyObject *__pyx_v_y = NULL;
  PyObject *__pyx_v_tmp = NULL;
  PyObject *__pyx_r = NULL;
  <span class='refnanny'>__Pyx_RefNannyDeclarations</span>
  <span class='refnanny'>__Pyx_RefNannySetupContext</span>("compute_cy", 0);
/* … */
  /* function exit code */
  __pyx_L1_error:;
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_1);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_2);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_3);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_5);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_10);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_11);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_13);
  <span class='pyx_c_api'>__Pyx_AddTraceback</span>("_cython_magic_c8a0d4053b6e33d70de8541b3a985ec3.compute_cy", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_v_x_max);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_v_y_max);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_v_result);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_v_x);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_v_y);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_v_tmp);
  <span class='refnanny'>__Pyx_XGIVEREF</span>(__pyx_r);
  <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
  return __pyx_r;
}
/* … */
  __pyx_tuple__5 = <span class='py_c_api'>PyTuple_Pack</span>(11, __pyx_n_s_array_1, __pyx_n_s_array_2, __pyx_n_s_a, __pyx_n_s_b, __pyx_n_s_c, __pyx_n_s_x_max, __pyx_n_s_y_max, __pyx_n_s_result, __pyx_n_s_x, __pyx_n_s_y, __pyx_n_s_tmp);<span class='error_goto'> if (unlikely(!__pyx_tuple__5)) __PYX_ERR(0, 13, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_tuple__5);
  <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_tuple__5);
/* … */
  __pyx_t_2 = PyCFunction_NewEx(&amp;__pyx_mdef_46_cython_magic_c8a0d4053b6e33d70de8541b3a985ec3_3compute_cy, NULL, __pyx_n_s_cython_magic_c8a0d4053b6e33d70d);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 13, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_compute_cy, __pyx_t_2) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 13, __pyx_L1_error)</span>
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
</pre><pre class="cython line score-0">&#xA0;<span class="">14</span>: </pre>
<pre class="cython line score-5" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">15</span>:     <span class="n">x_max</span> <span class="o">=</span> <span class="n">array_1</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mf">0</span><span class="p">]</span></pre>
<pre class='cython code score-5 '>  __pyx_t_1 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_v_array_1, __pyx_n_s_shape);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 15, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  __pyx_t_2 = <span class='pyx_c_api'>__Pyx_GetItemInt</span>(__pyx_t_1, 0, long, 1, __Pyx_PyInt_From_long, 0, 0, 1);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 15, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_v_x_max = __pyx_t_2;
  __pyx_t_2 = 0;
</pre><pre class="cython line score-5" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">16</span>:     <span class="n">y_max</span> <span class="o">=</span> <span class="n">array_1</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mf">1</span><span class="p">]</span></pre>
<pre class='cython code score-5 '>  __pyx_t_2 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_v_array_1, __pyx_n_s_shape);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
  __pyx_t_1 = <span class='pyx_c_api'>__Pyx_GetItemInt</span>(__pyx_t_2, 1, long, 1, __Pyx_PyInt_From_long, 0, 0, 1);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
  __pyx_v_y_max = __pyx_t_1;
  __pyx_t_1 = 0;
</pre><pre class="cython line score-0">&#xA0;<span class="">17</span>: </pre>
<pre class="cython line score-19" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">18</span>:     <span class="k">assert</span> <span class="n">array_1</span><span class="o">.</span><span class="n">shape</span> <span class="o">==</span> <span class="n">array_2</span><span class="o">.</span><span class="n">shape</span></pre>
<pre class='cython code score-19 '>  #ifndef CYTHON_WITHOUT_ASSERTIONS
  if (unlikely(!Py_OptimizeFlag)) {
    __pyx_t_1 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_v_array_1, __pyx_n_s_shape);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 18, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
    __pyx_t_2 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_v_array_2, __pyx_n_s_shape);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 18, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
    __pyx_t_3 = <span class='py_c_api'>PyObject_RichCompare</span>(__pyx_t_1, __pyx_t_2, Py_EQ); <span class='refnanny'>__Pyx_XGOTREF</span>(__pyx_t_3);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 18, __pyx_L1_error)</span>
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
    __pyx_t_4 = <span class='pyx_c_api'>__Pyx_PyObject_IsTrue</span>(__pyx_t_3); if (unlikely(__pyx_t_4 &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 18, __pyx_L1_error)</span>
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
    if (unlikely(!__pyx_t_4)) {
      <span class='py_c_api'>PyErr_SetNone</span>(PyExc_AssertionError);
      <span class='error_goto'>__PYX_ERR(0, 18, __pyx_L1_error)</span>
    }
  }
  #endif
</pre><pre class="cython line score-0">&#xA0;<span class="">19</span>: </pre>
<pre class="cython line score-35" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">20</span>:     <span class="n">result</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">((</span><span class="n">x_max</span><span class="p">,</span> <span class="n">y_max</span><span class="p">),</span> <span class="n">dtype</span><span class="o">=</span><span class="n">array_1</span><span class="o">.</span><span class="n">dtype</span><span class="p">)</span></pre>
<pre class='cython code score-35 '>  <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_t_3, __pyx_n_s_np);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 20, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  __pyx_t_2 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_t_3, __pyx_n_s_zeros);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 20, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
  __pyx_t_3 = <span class='py_c_api'>PyTuple_New</span>(2);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 20, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_v_x_max);
  <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_v_x_max);
  <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_3, 0, __pyx_v_x_max);
  <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_v_y_max);
  <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_v_y_max);
  <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_3, 1, __pyx_v_y_max);
  __pyx_t_1 = <span class='py_c_api'>PyTuple_New</span>(1);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 20, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_t_3);
  <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_1, 0, __pyx_t_3);
  __pyx_t_3 = 0;
  __pyx_t_3 = <span class='pyx_c_api'>__Pyx_PyDict_NewPresized</span>(1);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 20, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  __pyx_t_5 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_v_array_1, __pyx_n_s_dtype);<span class='error_goto'> if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 20, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_t_3, __pyx_n_s_dtype, __pyx_t_5) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 20, __pyx_L1_error)</span>
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_5); __pyx_t_5 = 0;
  __pyx_t_5 = <span class='pyx_c_api'>__Pyx_PyObject_Call</span>(__pyx_t_2, __pyx_t_1, __pyx_t_3);<span class='error_goto'> if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 20, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
  __pyx_v_result = __pyx_t_5;
  __pyx_t_5 = 0;
</pre><pre class="cython line score-0">&#xA0;<span class="">21</span>: </pre>
<pre class="cython line score-46" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">22</span>:     <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">x_max</span><span class="p">):</span></pre>
<pre class='cython code score-46 '>  __pyx_t_5 = <span class='pyx_c_api'>__Pyx_PyObject_CallOneArg</span>(__pyx_builtin_range, __pyx_v_x_max);<span class='error_goto'> if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 22, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
  if (likely(<span class='py_c_api'>PyList_CheckExact</span>(__pyx_t_5)) || <span class='py_c_api'>PyTuple_CheckExact</span>(__pyx_t_5)) {
    __pyx_t_3 = __pyx_t_5; <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_t_3); __pyx_t_6 = 0;
    __pyx_t_7 = NULL;
  } else {
    __pyx_t_6 = -1; __pyx_t_3 = <span class='py_c_api'>PyObject_GetIter</span>(__pyx_t_5);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 22, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
    __pyx_t_7 = Py_TYPE(__pyx_t_3)-&gt;tp_iternext;<span class='error_goto'> if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 22, __pyx_L1_error)</span>
  }
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_5); __pyx_t_5 = 0;
  for (;;) {
    if (likely(!__pyx_t_7)) {
      if (likely(<span class='py_c_api'>PyList_CheckExact</span>(__pyx_t_3))) {
        if (__pyx_t_6 &gt;= <span class='py_macro_api'>PyList_GET_SIZE</span>(__pyx_t_3)) break;
        #if CYTHON_ASSUME_SAFE_MACROS &amp;&amp; !CYTHON_AVOID_BORROWED_REFS
        __pyx_t_5 = <span class='py_macro_api'>PyList_GET_ITEM</span>(__pyx_t_3, __pyx_t_6); <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_t_5); __pyx_t_6++; if (unlikely(0 &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 22, __pyx_L1_error)</span>
        #else
        __pyx_t_5 = <span class='py_macro_api'>PySequence_ITEM</span>(__pyx_t_3, __pyx_t_6); __pyx_t_6++;<span class='error_goto'> if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 22, __pyx_L1_error)</span>
        <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
        #endif
      } else {
        if (__pyx_t_6 &gt;= <span class='py_macro_api'>PyTuple_GET_SIZE</span>(__pyx_t_3)) break;
        #if CYTHON_ASSUME_SAFE_MACROS &amp;&amp; !CYTHON_AVOID_BORROWED_REFS
        __pyx_t_5 = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_t_3, __pyx_t_6); <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_t_5); __pyx_t_6++; if (unlikely(0 &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 22, __pyx_L1_error)</span>
        #else
        __pyx_t_5 = <span class='py_macro_api'>PySequence_ITEM</span>(__pyx_t_3, __pyx_t_6); __pyx_t_6++;<span class='error_goto'> if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 22, __pyx_L1_error)</span>
        <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
        #endif
      }
    } else {
      __pyx_t_5 = __pyx_t_7(__pyx_t_3);
      if (unlikely(!__pyx_t_5)) {
        PyObject* exc_type = <span class='py_c_api'>PyErr_Occurred</span>();
        if (exc_type) {
          if (likely(<span class='pyx_c_api'>__Pyx_PyErr_GivenExceptionMatches</span>(exc_type, PyExc_StopIteration))) <span class='py_c_api'>PyErr_Clear</span>();
          else <span class='error_goto'>__PYX_ERR(0, 22, __pyx_L1_error)</span>
        }
        break;
      }
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
    }
    <span class='pyx_macro_api'>__Pyx_XDECREF_SET</span>(__pyx_v_x, __pyx_t_5);
    __pyx_t_5 = 0;
/* … */
  }
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
</pre><pre class="cython line score-46" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">23</span>:         <span class="k">for</span> <span class="n">y</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">y_max</span><span class="p">):</span></pre>
<pre class='cython code score-46 '>    __pyx_t_5 = <span class='pyx_c_api'>__Pyx_PyObject_CallOneArg</span>(__pyx_builtin_range, __pyx_v_y_max);<span class='error_goto'> if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 23, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
    if (likely(<span class='py_c_api'>PyList_CheckExact</span>(__pyx_t_5)) || <span class='py_c_api'>PyTuple_CheckExact</span>(__pyx_t_5)) {
      __pyx_t_1 = __pyx_t_5; <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_t_1); __pyx_t_8 = 0;
      __pyx_t_9 = NULL;
    } else {
      __pyx_t_8 = -1; __pyx_t_1 = <span class='py_c_api'>PyObject_GetIter</span>(__pyx_t_5);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 23, __pyx_L1_error)</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
      __pyx_t_9 = Py_TYPE(__pyx_t_1)-&gt;tp_iternext;<span class='error_goto'> if (unlikely(!__pyx_t_9)) __PYX_ERR(0, 23, __pyx_L1_error)</span>
    }
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_5); __pyx_t_5 = 0;
    for (;;) {
      if (likely(!__pyx_t_9)) {
        if (likely(<span class='py_c_api'>PyList_CheckExact</span>(__pyx_t_1))) {
          if (__pyx_t_8 &gt;= <span class='py_macro_api'>PyList_GET_SIZE</span>(__pyx_t_1)) break;
          #if CYTHON_ASSUME_SAFE_MACROS &amp;&amp; !CYTHON_AVOID_BORROWED_REFS
          __pyx_t_5 = <span class='py_macro_api'>PyList_GET_ITEM</span>(__pyx_t_1, __pyx_t_8); <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_t_5); __pyx_t_8++; if (unlikely(0 &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 23, __pyx_L1_error)</span>
          #else
          __pyx_t_5 = <span class='py_macro_api'>PySequence_ITEM</span>(__pyx_t_1, __pyx_t_8); __pyx_t_8++;<span class='error_goto'> if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 23, __pyx_L1_error)</span>
          <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
          #endif
        } else {
          if (__pyx_t_8 &gt;= <span class='py_macro_api'>PyTuple_GET_SIZE</span>(__pyx_t_1)) break;
          #if CYTHON_ASSUME_SAFE_MACROS &amp;&amp; !CYTHON_AVOID_BORROWED_REFS
          __pyx_t_5 = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_t_1, __pyx_t_8); <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_t_5); __pyx_t_8++; if (unlikely(0 &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 23, __pyx_L1_error)</span>
          #else
          __pyx_t_5 = <span class='py_macro_api'>PySequence_ITEM</span>(__pyx_t_1, __pyx_t_8); __pyx_t_8++;<span class='error_goto'> if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 23, __pyx_L1_error)</span>
          <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
          #endif
        }
      } else {
        __pyx_t_5 = __pyx_t_9(__pyx_t_1);
        if (unlikely(!__pyx_t_5)) {
          PyObject* exc_type = <span class='py_c_api'>PyErr_Occurred</span>();
          if (exc_type) {
            if (likely(<span class='pyx_c_api'>__Pyx_PyErr_GivenExceptionMatches</span>(exc_type, PyExc_StopIteration))) <span class='py_c_api'>PyErr_Clear</span>();
            else <span class='error_goto'>__PYX_ERR(0, 23, __pyx_L1_error)</span>
          }
          break;
        }
        <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
      }
      <span class='pyx_macro_api'>__Pyx_XDECREF_SET</span>(__pyx_v_y, __pyx_t_5);
      __pyx_t_5 = 0;
/* … */
    }
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
</pre><pre class="cython line score-55" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">24</span>:             <span class="n">tmp</span> <span class="o">=</span> <span class="n">clip</span><span class="p">(</span><span class="n">array_1</span><span class="p">[</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">],</span> <span class="mf">2</span><span class="p">,</span> <span class="mf">10</span><span class="p">)</span></pre>
<pre class='cython code score-55 '>      <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_t_2, __pyx_n_s_clip);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 24, __pyx_L1_error)</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
      __pyx_t_10 = <span class='py_c_api'>PyTuple_New</span>(2);<span class='error_goto'> if (unlikely(!__pyx_t_10)) __PYX_ERR(0, 24, __pyx_L1_error)</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_10);
      <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_v_x);
      <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_v_x);
      <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_10, 0, __pyx_v_x);
      <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_v_y);
      <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_v_y);
      <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_10, 1, __pyx_v_y);
      __pyx_t_11 = <span class='pyx_c_api'>__Pyx_PyObject_GetItem</span>(__pyx_v_array_1, __pyx_t_10);<span class='error_goto'> if (unlikely(!__pyx_t_11)) __PYX_ERR(0, 24, __pyx_L1_error)</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_11);
      <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_10); __pyx_t_10 = 0;
      __pyx_t_10 = NULL;
      __pyx_t_12 = 0;
      if (CYTHON_UNPACK_METHODS &amp;&amp; unlikely(<span class='py_c_api'>PyMethod_Check</span>(__pyx_t_2))) {
        __pyx_t_10 = <span class='py_macro_api'>PyMethod_GET_SELF</span>(__pyx_t_2);
        if (likely(__pyx_t_10)) {
          PyObject* function = <span class='py_macro_api'>PyMethod_GET_FUNCTION</span>(__pyx_t_2);
          <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_t_10);
          <span class='pyx_macro_api'>__Pyx_INCREF</span>(function);
          <span class='pyx_macro_api'>__Pyx_DECREF_SET</span>(__pyx_t_2, function);
          __pyx_t_12 = 1;
        }
      }
      #if CYTHON_FAST_PYCALL
      if (<span class='py_c_api'>PyFunction_Check</span>(__pyx_t_2)) {
        PyObject *__pyx_temp[4] = {__pyx_t_10, __pyx_t_11, __pyx_int_2, __pyx_int_10};
        __pyx_t_5 = <span class='pyx_c_api'>__Pyx_PyFunction_FastCall</span>(__pyx_t_2, __pyx_temp+1-__pyx_t_12, 3+__pyx_t_12);<span class='error_goto'> if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 24, __pyx_L1_error)</span>
        <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_10); __pyx_t_10 = 0;
        <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
        <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_11); __pyx_t_11 = 0;
      } else
      #endif
      #if CYTHON_FAST_PYCCALL
      if (<span class='pyx_c_api'>__Pyx_PyFastCFunction_Check</span>(__pyx_t_2)) {
        PyObject *__pyx_temp[4] = {__pyx_t_10, __pyx_t_11, __pyx_int_2, __pyx_int_10};
        __pyx_t_5 = <span class='pyx_c_api'>__Pyx_PyCFunction_FastCall</span>(__pyx_t_2, __pyx_temp+1-__pyx_t_12, 3+__pyx_t_12);<span class='error_goto'> if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 24, __pyx_L1_error)</span>
        <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_10); __pyx_t_10 = 0;
        <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
        <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_11); __pyx_t_11 = 0;
      } else
      #endif
      {
        __pyx_t_13 = <span class='py_c_api'>PyTuple_New</span>(3+__pyx_t_12);<span class='error_goto'> if (unlikely(!__pyx_t_13)) __PYX_ERR(0, 24, __pyx_L1_error)</span>
        <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_13);
        if (__pyx_t_10) {
          <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_t_10); <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_13, 0, __pyx_t_10); __pyx_t_10 = NULL;
        }
        <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_t_11);
        <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_13, 0+__pyx_t_12, __pyx_t_11);
        <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_int_2);
        <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_int_2);
        <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_13, 1+__pyx_t_12, __pyx_int_2);
        <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_int_10);
        <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_int_10);
        <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_13, 2+__pyx_t_12, __pyx_int_10);
        __pyx_t_11 = 0;
        __pyx_t_5 = <span class='pyx_c_api'>__Pyx_PyObject_Call</span>(__pyx_t_2, __pyx_t_13, NULL);<span class='error_goto'> if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 24, __pyx_L1_error)</span>
        <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
        <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_13); __pyx_t_13 = 0;
      }
      <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
      <span class='pyx_macro_api'>__Pyx_XDECREF_SET</span>(__pyx_v_tmp, __pyx_t_5);
      __pyx_t_5 = 0;
</pre><pre class="cython line score-31" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">25</span>:             <span class="n">tmp</span> <span class="o">=</span> <span class="n">tmp</span> <span class="o">*</span> <span class="n">a</span> <span class="o">+</span> <span class="n">array_2</span><span class="p">[</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">]</span> <span class="o">*</span> <span class="n">b</span></pre>
<pre class='cython code score-31 '>      __pyx_t_5 = <span class='py_c_api'>PyNumber_Multiply</span>(__pyx_v_tmp, __pyx_v_a);<span class='error_goto'> if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 25, __pyx_L1_error)</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
      __pyx_t_2 = <span class='py_c_api'>PyTuple_New</span>(2);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 25, __pyx_L1_error)</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
      <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_v_x);
      <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_v_x);
      <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_2, 0, __pyx_v_x);
      <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_v_y);
      <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_v_y);
      <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_2, 1, __pyx_v_y);
      __pyx_t_13 = <span class='pyx_c_api'>__Pyx_PyObject_GetItem</span>(__pyx_v_array_2, __pyx_t_2);<span class='error_goto'> if (unlikely(!__pyx_t_13)) __PYX_ERR(0, 25, __pyx_L1_error)</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_13);
      <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
      __pyx_t_2 = <span class='py_c_api'>PyNumber_Multiply</span>(__pyx_t_13, __pyx_v_b);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 25, __pyx_L1_error)</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
      <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_13); __pyx_t_13 = 0;
      __pyx_t_13 = <span class='py_c_api'>PyNumber_Add</span>(__pyx_t_5, __pyx_t_2);<span class='error_goto'> if (unlikely(!__pyx_t_13)) __PYX_ERR(0, 25, __pyx_L1_error)</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_13);
      <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_5); __pyx_t_5 = 0;
      <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
      <span class='pyx_macro_api'>__Pyx_DECREF_SET</span>(__pyx_v_tmp, __pyx_t_13);
      __pyx_t_13 = 0;
</pre><pre class="cython line score-21" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">26</span>:             <span class="n">result</span><span class="p">[</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">]</span> <span class="o">=</span> <span class="n">tmp</span> <span class="o">+</span> <span class="n">c</span></pre>
<pre class='cython code score-21 '>      __pyx_t_13 = <span class='py_c_api'>PyNumber_Add</span>(__pyx_v_tmp, __pyx_v_c);<span class='error_goto'> if (unlikely(!__pyx_t_13)) __PYX_ERR(0, 26, __pyx_L1_error)</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_13);
      __pyx_t_2 = <span class='py_c_api'>PyTuple_New</span>(2);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 26, __pyx_L1_error)</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
      <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_v_x);
      <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_v_x);
      <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_2, 0, __pyx_v_x);
      <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_v_y);
      <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_v_y);
      <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_2, 1, __pyx_v_y);
      if (unlikely(<span class='py_c_api'>PyObject_SetItem</span>(__pyx_v_result, __pyx_t_2, __pyx_t_13) &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 26, __pyx_L1_error)</span>
      <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
      <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_13); __pyx_t_13 = 0;
</pre><pre class="cython line score-0">&#xA0;<span class="">27</span>: </pre>
<pre class="cython line score-2" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">28</span>:     <span class="k">return</span> <span class="n">result</span></pre>
<pre class='cython code score-2 '>  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_r);
  <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_v_result);
  __pyx_r = __pyx_v_result;
  goto __pyx_L0;
</pre></div></body></html>



## 5.2 Add Types

The first thing we need to do is define the type, we named it **compute_cy_t**. This means that the object passed will be converted to a C type.

With the results we can see that the speed has improved compared to the previous.


```cython
%%cython --compile-args=-O3

import numpy as np
import timeit

array_1 = np.random.uniform(0, 100, size=(3000, 2000)).astype(np.intc)
array_2 = np.random.uniform(0, 100, size=(3000, 2000)).astype(np.intc)
a = 4
b = 3
c = 9

DTYPE = np.intc   # numpy.intc ---- int. Otherwise,they are implicitly typed as Python objects

cdef int clip(int a, int min_value, int max_value):
    return min(max(a, min_value), max_value)


def compute_cy_t(array_1, array_2, int a, int b, int c):
    cdef Py_ssize_t x_max = array_1.shape[0]
    cdef Py_ssize_t y_max = array_1.shape[1]
    
    assert array_1.shape == array_2.shape
    assert array_1.dtype == DTYPE
    assert array_2.dtype == DTYPE

    result = np.zeros((x_max, y_max), dtype=DTYPE)
    
    cdef int tmp

    cdef Py_ssize_t x, y

    for x in range(x_max):
        for y in range(y_max):

            tmp = clip(array_1[x, y], 2, 10)
            tmp = tmp * a + array_2[x, y] * b
            result[x, y] = tmp + c

    return result

print(compute_cy_t(array_1, array_2, a, b, c))

compute_cy_t_time = timeit.timeit(lambda: compute_cy_t(array_1, array_2, a, b, c), number=10)/10

print("compute_cy_t execution time:", compute_cy_t_time)
```

    [[100 250 319 ...  63 334 283]
     [191 250 172 ... 346 136 240]
     [298 337 281 ... 139 106 115]
     ...
     [238  97 166 ...  73  70 250]
     [127 289 226 ... 196  73 256]
     [109 217 163 ...  21  41 160]]
    compute_cy_t execution time: 5.123431210000126
    

## 5.3 Efficient indexing with memoryviews

Adding types does make the code faster, but not nearly as fast as Numpy.

array_1 and array_2 are still NumPy arrays, so Python objects, and expect Python integers as indexes.
```
tmp = clip(array_1[x, y], 2, 10)\
tmp = tmp * a + array_2[x, y] * b
result[x, y] = tmp + c
```
Here we pass C int values. So every time Cython reaches this line, it has to convert all the C integers to Python int objects. Since this line is called very often, it outweighs the speed benefits of the pure C loops that were created from the range() earlier.

Furthermore, tmp * a + array_2[x, y] * b returns a Python integer and tmp is a C integer, so Cython has to do type conversions again. In the end those types conversions add up. And made our computation really slow. But this problem can be solved easily by using memoryviews.

memoryviews are C structures that can hold a pointer to the data of a NumPy array and all the necessary buffer metadata to provide efficient and safe access: dimensions, strides, item size, item type information, etc… They also support slices, so they work even if the NumPy array isn’t contiguous in memory. They can be indexed by C integers, thus allowing fast access to the NumPy array data.

Here is how to declare a memoryview of integers:
```
cdef int [:] foo         # 1D memoryview
cdef int [:, :] foo      # 2D memoryview
cdef int [:, :, :] foo   # 3D memoryview
```
Here is how to use them in our code,we named it **compute_cy_m**:


```cython
%%cython

import numpy as np
import timeit

array_1 = np.random.uniform(0, 100, size=(3000, 2000)).astype(np.intc)
array_2 = np.random.uniform(0, 100, size=(3000, 2000)).astype(np.intc)
a = 4
b = 3
c = 9

DTYPE = np.intc   # numpy.intc ---- int. Otherwise,they are implicitly typed as Python objects

cdef int clip(int a, int min_value, int max_value):
    return min(max(a, min_value), max_value)

def compute_cy_m(int[:, :] array_1, int[:, :] array_2, int a, int b, int c):
     
    cdef Py_ssize_t x_max = array_1.shape[0]
    cdef Py_ssize_t y_max = array_1.shape[1]

    assert tuple(array_1.shape) == tuple(array_2.shape)

    result = np.zeros((x_max, y_max), dtype=DTYPE)
    cdef int[:, :] result_view = result

    cdef int tmp
    cdef Py_ssize_t x, y

    for x in range(x_max):
        for y in range(y_max):

            tmp = clip(array_1[x, y], 2, 10)
            tmp = tmp * a + array_2[x, y] * b
            result_view[x, y] = tmp + c

    return result

print(compute_cy_m(array_1, array_2, a, b, c))

compute_cy_m_time = timeit.timeit(lambda: compute_cy_m(array_1, array_2, a, b, c), number=10)/10

print("compute_cy_m execution time:", compute_cy_m_time)
```

    [[235 173 139 ... 133  82 286]
     [301 322 256 ...  91  36 235]
     [343  85 235 ... 317 283 190]
     ...
     [ 55  76 253 ... 193 217 235]
     [292 238 292 ... 199 103 301]
     [133 124 289 ...  68 232 310]]
    compute_cy_m execution time: 0.037777249999999186
    

Here we can see how much faster it has become。

## 5.2 Tuning indexing further

The array lookups are still slowed down by two factors:

Bounds checking is performed.

Negative indices are checked for and handled correctly. The code above is explicitly coded so that it doesn’t use negative indices, and it (hopefully) always access within bounds.

With decorators, we can deactivate those checks:
```
...
cimport cython
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def compute(int[:, :] array_1, int[:, :] array_2, int a, int b, int c):
...
```
Note that these lines of code are to be placed above the specified function, not at the beginning.Here we named it **compute_cy_i**.


```cython
%%cython

import numpy as np
import timeit


array_1 = np.random.uniform(0, 100, size=(3000, 2000)).astype(np.intc)
array_2 = np.random.uniform(0, 100, size=(3000, 2000)).astype(np.intc)
a = 4
b = 3
c = 9

DTYPE = np.intc   # numpy.intc ---- int. Otherwise,they are implicitly typed as Python objects

cdef int clip(int a, int min_value, int max_value):
    return min(max(a, min_value), max_value)

cimport cython
@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.

def compute_cy_i(int[:, :] array_1, int[:, :] array_2, int a, int b, int c):
     
    cdef Py_ssize_t x_max = array_1.shape[0]
    cdef Py_ssize_t y_max = array_1.shape[1]

    assert tuple(array_1.shape) == tuple(array_2.shape)

    result = np.zeros((x_max, y_max), dtype=DTYPE)
    cdef int[:, :] result_view = result

    cdef int tmp
    cdef Py_ssize_t x, y

    for x in range(x_max):
        for y in range(y_max):

            tmp = clip(array_1[x, y], 2, 10)
            tmp = tmp * a + array_2[x, y] * b
            result_view[x, y] = tmp + c

    return result

print(compute_cy_i(array_1, array_2, a, b, c))

compute_cy_i_time = timeit.timeit(lambda: compute_cy_i(array_1, array_2, a, b, c), number=10)/10

print("compute_cy_i execution time:", compute_cy_i_time)
```

    [[187 259  94 ...  79 190 142]
     [154 178 175 ... 304 217 307]
     [106 331 346 ... 313 103 130]
     ...
     [ 85 280 313 ... 208  73  85]
     [298 256  85 ... 274 260 247]
     [ 82 322 244 ... 241 157 181]]
    compute_cy_i execution time: 0.014358759999959147
    

## 5.3 Declaring the NumPy arrays as contiguous

For extra speed gains, if you know that the NumPy arrays you are providing are contiguous in memory, you can declare the memoryview as contiguous.

We give an example on an array that has 3 dimensions. If you want to give Cython the information that the data is C-contiguous you have to declare the memoryview like this:
```
cdef int [:,:,::1] a
```
If you want to give Cython the information that the data is Fortran-contiguous you have to declare the memoryview like this:
```
cdef int [::1, :, :] a
```
If all this makes no sense to you, you can skip this part, declaring arrays as contiguous constrains the usage of your functions as it rejects array slices as input. 

Here we named it **compute_cy_c**.


```cython
%%cython 

import numpy as np
import timeit


array_1 = np.random.uniform(0, 100, size=(3000, 2000)).astype(np.intc)
array_2 = np.random.uniform(0, 100, size=(3000, 2000)).astype(np.intc)
a = 4
b = 3
c = 9

DTYPE = np.intc   # numpy.intc ---- int. Otherwise,they are implicitly typed as Python objects

cdef int clip(int a, int min_value, int max_value):
    return min(max(a, min_value), max_value)

cimport cython
@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.

def compute_cy_c(int[:, ::1] array_1, int[:, ::1] array_2, int a, int b, int c):
     
    cdef Py_ssize_t x_max = array_1.shape[0]
    cdef Py_ssize_t y_max = array_1.shape[1]

    assert tuple(array_1.shape) == tuple(array_2.shape)

    result = np.zeros((x_max, y_max), dtype=DTYPE)
    cdef int[:, ::1] result_view = result

    cdef int tmp
    cdef Py_ssize_t x, y

    for x in range(x_max):
        for y in range(y_max):

            tmp = clip(array_1[x, y], 2, 10)
            tmp = tmp * a + array_2[x, y] * b
            result_view[x, y] = tmp + c

    return result

print(compute_cy_c(array_1, array_2, a, b, c))

compute_cy_c_time = timeit.timeit(lambda: compute_cy_c(array_1, array_2, a, b, c), number=10)/10

print("compute_cy_c execution time:", compute_cy_c_time)
```

    [[283 109 244 ...  58  61 328]
     [ 85 142 331 ... 145 346 259]
     [202 235 109 ... 169 280  52]
     ...
     [325 232 271 ... 151  87 202]
     [ 52  73 295 ... 247 235 286]
     [232 268 127 ... 196 136 211]]
    compute_cy_c execution time: 0.014614970000002359
    

Obviously, this optimization doesn't work here.

## 5.4 Making the function cleaner & Use of multiple data types

Declaring types can make your code quite verbose. If you don’t mind Cython inferring the C types of your variables, you can use the following compiler directive at the top of the file. It will save you quite a bit of typing.
```
infer_types=True
```
Note that since type declarations must happen at the top indentation level, Cython won’t infer the type of variables declared for the first time in other indentation levels. It would change too much the meaning of our code. This is why, we must still declare manually the type of the tmp, x and y variable.

And actually, manually giving the type of the tmp variable will be useful when using fused types.

All those speed gains are nice, but adding types constrains our code. At the moment, it would mean that our function can only work with NumPy arrays with the np.intc type.

So we can use fused types to make our code work with multiple NumPy data types. The code is as follows:
```
ctypedef fused my_type:
    int
    double
    long long
```
It is similar to a C++ template. It generates multiple function declarations at compile time and then selects the correct function at runtime based on the types of arguments provided. By comparing the types in the if-conditions, it is also possible to execute a completely different code path depending on the specific data type.

In our example, since we can no longer access the dtype of NumPy's input array, we use these if-else statements to know what NumPy data type should be used for our output array.

In this case, our function now applies to ints, doubles and floats.Here we named it **compute_cy_mdt**.


```cython
%%cython 

# cython: infer_types=True
import numpy as np
import timeit
cimport cython

array_1 = np.random.uniform(0, 100, size=(3000, 2000)).astype(np.intc)
array_2 = np.random.uniform(0, 100, size=(3000, 2000)).astype(np.intc)
a = 4
b = 3
c = 9

ctypedef fused my_type:
    int
    double
    long long

cdef my_type clip(my_type a, my_type min_value, my_type max_value):
    return min(max(a, min_value), max_value)

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_cy_mdt(my_type[:, ::1] array_1, my_type[:, ::1] array_2, my_type a, my_type b, my_type c):
     
    x_max = array_1.shape[0]
    y_max = array_1.shape[1]
    
    assert tuple(array_1.shape) == tuple(array_2.shape)

    if my_type is int:
        dtype = np.intc
    elif my_type is double:
        dtype = np.double
    elif my_type is cython.longlong:
        dtype = np.longlong

    result = np.zeros((x_max, y_max), dtype=dtype)
    cdef my_type[:, ::1] result_view = result

    cdef my_type tmp
    cdef Py_ssize_t x, y

    for x in range(x_max):
        for y in range(y_max):

            tmp = clip(array_1[x, y], 2, 10)
            tmp = tmp * a + array_2[x, y] * b
            result_view[x, y] = tmp + c

    return result

print(compute_cy_mdt(array_1, array_2, a, b, c))

compute_cy_mdt_time = timeit.timeit(lambda: compute_cy_mdt(array_1, array_2, a, b, c), number=10)/10

print("compute_cy_mdt execution time:", compute_cy_mdt_time)
```

    [[109 337 268 ...  55 316 184]
     [226 157 310 ... 274 273 258]
     [256 328  61 ...  82 241 259]
     ...
     [154 274 328 ... 340 262 217]
     [304 100  55 ... 256 334 148]
     [ 58 170 289 ... 226 274  91]]
    compute_cy_mdt execution time: 0.014951370000017051
    

## 5.5 Using multiple threads

Cython has support for OpenMP. It also has some nice wrappers around it, like the function prange(). 

The GIL must be released, so this is why we declare our clip() function nogil.

Here we named it **compute_cy_mt.**

**Note:**
The important thing to note here is that we need to open openmp, this may be different for different operating systems, I'm running this on windows so here is: -c=/openmp


```cython
%%cython --force -c=/openmp
# tag: openmp
# You can ignore the previous line.
# It's for internal testing of the cython documentation.

# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

import numpy as np
import timeit
cimport cython
from cython.parallel import prange

array_1 = np.random.uniform(0, 100, size=(3000, 2000)).astype(np.intc)
array_2 = np.random.uniform(0, 100, size=(3000, 2000)).astype(np.intc)
a = 4
b = 3
c = 9

ctypedef fused my_type:
    int
    double
    long long


# We declare our plain c function nogil
cdef my_type clip(my_type a, my_type min_value, my_type max_value) nogil:
    return min(max(a, min_value), max_value)


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_cy_mt(my_type[:, ::1] array_1, my_type[:, ::1] array_2, my_type a, my_type b, my_type c):

    cdef Py_ssize_t x_max = array_1.shape[0]
    cdef Py_ssize_t y_max = array_1.shape[1]

    assert tuple(array_1.shape) == tuple(array_2.shape)

    if my_type is int:
        dtype = np.intc
    elif my_type is double:
        dtype = np.double
    elif my_type is cython.longlong:
        dtype = np.longlong

    result = np.zeros((x_max, y_max), dtype=dtype)
    cdef my_type[:, ::1] result_view = result

    cdef my_type tmp
    cdef Py_ssize_t x, y

    # We use prange here.
    # for x in prange(x_max, nogil=True):
    #     for y in range(y_max):
    for x in range(x_max):
        for y in prange(y_max, num_threads=4,nogil=True):
            tmp = clip(array_1[x, y], 2, 10)
            tmp = tmp * a + array_2[x, y] * b
            result_view[x, y] = tmp + c

    return result

print(compute_cy_mt(array_1, array_2, a, b, c))

compute_cy_mt_time = timeit.timeit(lambda: compute_cy_mt(array_1, array_2, a, b, c), number=10)/10

print("compute_cy_mt execution time:", compute_cy_mt_time)
```

    [[274 322 337 ... 280 192 262]
     [130 200 277 ... 121 307 304]
     [ 76 124  59 ... 184 256 265]
     ...
     [169  38 340 ... 175 247 328]
     [325 337 319 ... 180 328 283]
     [252  97 285 ... 193  69 169]]
    compute_cy_mt execution time: 0.04007001999998465
    

Parallel computing does not necessarily lead to an eventual speed increase, and the reasons for this can be varied. For example, Parallelizing operations incurs additional overhead, such as thread creation and synchronization, that may outweigh the benefits of parallelism for small or simple operations.

Here, instead, I am running slower, which could be that this example is not applicable to parallel computing or is not adapted to my hardware.

# 6 Summarize

 ## 6.1 html file
 
 The final html file is as follows. Most of the lines have become lighter in colour, even white.


```cython
%%cython -a

import numpy as np
cimport cython
from cython.parallel import prange

ctypedef fused my_type:
    int
    double
    long long

cdef my_type clip(my_type a, my_type min_value, my_type max_value) nogil:
    return min(max(a, min_value), max_value)

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_cy_mt(my_type[:, ::1] array_1, my_type[:, ::1] array_2, my_type a, my_type b, my_type c):

    cdef Py_ssize_t x_max = array_1.shape[0]
    cdef Py_ssize_t y_max = array_1.shape[1]

    assert tuple(array_1.shape) == tuple(array_2.shape)

    if my_type is int:
        dtype = np.intc
    elif my_type is double:
        dtype = np.double
    elif my_type is cython.longlong:
        dtype = np.longlong

    result = np.zeros((x_max, y_max), dtype=dtype)
    cdef my_type[:, ::1] result_view = result

    cdef my_type tmp
    cdef Py_ssize_t x, y

    for x in range(x_max):
        for y in prange(y_max, num_threads=4,nogil=True):
            tmp = clip(array_1[x, y], 2, 10)
            tmp = tmp * a + array_2[x, y] * b
            result_view[x, y] = tmp + c

    return result
```




<!DOCTYPE html>
<!-- Generated by Cython 0.29.33 -->
<html>
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <title>Cython: _cython_magic_cd184590f55726778fe97a651b5f5dd0.pyx</title>
    <style type="text/css">

body.cython { font-family: courier; font-size: 12; }

.cython.tag  {  }
.cython.line { margin: 0em }
.cython.code { font-size: 9; color: #444444; display: none; margin: 0px 0px 0px 8px; border-left: 8px none; }

.cython.line .run { background-color: #B0FFB0; }
.cython.line .mis { background-color: #FFB0B0; }
.cython.code.run  { border-left: 8px solid #B0FFB0; }
.cython.code.mis  { border-left: 8px solid #FFB0B0; }

.cython.code .py_c_api  { color: red; }
.cython.code .py_macro_api  { color: #FF7000; }
.cython.code .pyx_c_api  { color: #FF3000; }
.cython.code .pyx_macro_api  { color: #FF7000; }
.cython.code .refnanny  { color: #FFA000; }
.cython.code .trace  { color: #FFA000; }
.cython.code .error_goto  { color: #FFA000; }

.cython.code .coerce  { color: #008000; border: 1px dotted #008000 }
.cython.code .py_attr { color: #FF0000; font-weight: bold; }
.cython.code .c_attr  { color: #0000FF; }
.cython.code .py_call { color: #FF0000; font-weight: bold; }
.cython.code .c_call  { color: #0000FF; }

.cython.score-0 {background-color: #FFFFff;}
.cython.score-1 {background-color: #FFFFe7;}
.cython.score-2 {background-color: #FFFFd4;}
.cython.score-3 {background-color: #FFFFc4;}
.cython.score-4 {background-color: #FFFFb6;}
.cython.score-5 {background-color: #FFFFaa;}
.cython.score-6 {background-color: #FFFF9f;}
.cython.score-7 {background-color: #FFFF96;}
.cython.score-8 {background-color: #FFFF8d;}
.cython.score-9 {background-color: #FFFF86;}
.cython.score-10 {background-color: #FFFF7f;}
.cython.score-11 {background-color: #FFFF79;}
.cython.score-12 {background-color: #FFFF73;}
.cython.score-13 {background-color: #FFFF6e;}
.cython.score-14 {background-color: #FFFF6a;}
.cython.score-15 {background-color: #FFFF66;}
.cython.score-16 {background-color: #FFFF62;}
.cython.score-17 {background-color: #FFFF5e;}
.cython.score-18 {background-color: #FFFF5b;}
.cython.score-19 {background-color: #FFFF57;}
.cython.score-20 {background-color: #FFFF55;}
.cython.score-21 {background-color: #FFFF52;}
.cython.score-22 {background-color: #FFFF4f;}
.cython.score-23 {background-color: #FFFF4d;}
.cython.score-24 {background-color: #FFFF4b;}
.cython.score-25 {background-color: #FFFF48;}
.cython.score-26 {background-color: #FFFF46;}
.cython.score-27 {background-color: #FFFF44;}
.cython.score-28 {background-color: #FFFF43;}
.cython.score-29 {background-color: #FFFF41;}
.cython.score-30 {background-color: #FFFF3f;}
.cython.score-31 {background-color: #FFFF3e;}
.cython.score-32 {background-color: #FFFF3c;}
.cython.score-33 {background-color: #FFFF3b;}
.cython.score-34 {background-color: #FFFF39;}
.cython.score-35 {background-color: #FFFF38;}
.cython.score-36 {background-color: #FFFF37;}
.cython.score-37 {background-color: #FFFF36;}
.cython.score-38 {background-color: #FFFF35;}
.cython.score-39 {background-color: #FFFF34;}
.cython.score-40 {background-color: #FFFF33;}
.cython.score-41 {background-color: #FFFF32;}
.cython.score-42 {background-color: #FFFF31;}
.cython.score-43 {background-color: #FFFF30;}
.cython.score-44 {background-color: #FFFF2f;}
.cython.score-45 {background-color: #FFFF2e;}
.cython.score-46 {background-color: #FFFF2d;}
.cython.score-47 {background-color: #FFFF2c;}
.cython.score-48 {background-color: #FFFF2b;}
.cython.score-49 {background-color: #FFFF2b;}
.cython.score-50 {background-color: #FFFF2a;}
.cython.score-51 {background-color: #FFFF29;}
.cython.score-52 {background-color: #FFFF29;}
.cython.score-53 {background-color: #FFFF28;}
.cython.score-54 {background-color: #FFFF27;}
.cython.score-55 {background-color: #FFFF27;}
.cython.score-56 {background-color: #FFFF26;}
.cython.score-57 {background-color: #FFFF26;}
.cython.score-58 {background-color: #FFFF25;}
.cython.score-59 {background-color: #FFFF24;}
.cython.score-60 {background-color: #FFFF24;}
.cython.score-61 {background-color: #FFFF23;}
.cython.score-62 {background-color: #FFFF23;}
.cython.score-63 {background-color: #FFFF22;}
.cython.score-64 {background-color: #FFFF22;}
.cython.score-65 {background-color: #FFFF22;}
.cython.score-66 {background-color: #FFFF21;}
.cython.score-67 {background-color: #FFFF21;}
.cython.score-68 {background-color: #FFFF20;}
.cython.score-69 {background-color: #FFFF20;}
.cython.score-70 {background-color: #FFFF1f;}
.cython.score-71 {background-color: #FFFF1f;}
.cython.score-72 {background-color: #FFFF1f;}
.cython.score-73 {background-color: #FFFF1e;}
.cython.score-74 {background-color: #FFFF1e;}
.cython.score-75 {background-color: #FFFF1e;}
.cython.score-76 {background-color: #FFFF1d;}
.cython.score-77 {background-color: #FFFF1d;}
.cython.score-78 {background-color: #FFFF1c;}
.cython.score-79 {background-color: #FFFF1c;}
.cython.score-80 {background-color: #FFFF1c;}
.cython.score-81 {background-color: #FFFF1c;}
.cython.score-82 {background-color: #FFFF1b;}
.cython.score-83 {background-color: #FFFF1b;}
.cython.score-84 {background-color: #FFFF1b;}
.cython.score-85 {background-color: #FFFF1a;}
.cython.score-86 {background-color: #FFFF1a;}
.cython.score-87 {background-color: #FFFF1a;}
.cython.score-88 {background-color: #FFFF1a;}
.cython.score-89 {background-color: #FFFF19;}
.cython.score-90 {background-color: #FFFF19;}
.cython.score-91 {background-color: #FFFF19;}
.cython.score-92 {background-color: #FFFF19;}
.cython.score-93 {background-color: #FFFF18;}
.cython.score-94 {background-color: #FFFF18;}
.cython.score-95 {background-color: #FFFF18;}
.cython.score-96 {background-color: #FFFF18;}
.cython.score-97 {background-color: #FFFF17;}
.cython.score-98 {background-color: #FFFF17;}
.cython.score-99 {background-color: #FFFF17;}
.cython.score-100 {background-color: #FFFF17;}
.cython.score-101 {background-color: #FFFF16;}
.cython.score-102 {background-color: #FFFF16;}
.cython.score-103 {background-color: #FFFF16;}
.cython.score-104 {background-color: #FFFF16;}
.cython.score-105 {background-color: #FFFF16;}
.cython.score-106 {background-color: #FFFF15;}
.cython.score-107 {background-color: #FFFF15;}
.cython.score-108 {background-color: #FFFF15;}
.cython.score-109 {background-color: #FFFF15;}
.cython.score-110 {background-color: #FFFF15;}
.cython.score-111 {background-color: #FFFF15;}
.cython.score-112 {background-color: #FFFF14;}
.cython.score-113 {background-color: #FFFF14;}
.cython.score-114 {background-color: #FFFF14;}
.cython.score-115 {background-color: #FFFF14;}
.cython.score-116 {background-color: #FFFF14;}
.cython.score-117 {background-color: #FFFF14;}
.cython.score-118 {background-color: #FFFF13;}
.cython.score-119 {background-color: #FFFF13;}
.cython.score-120 {background-color: #FFFF13;}
.cython.score-121 {background-color: #FFFF13;}
.cython.score-122 {background-color: #FFFF13;}
.cython.score-123 {background-color: #FFFF13;}
.cython.score-124 {background-color: #FFFF13;}
.cython.score-125 {background-color: #FFFF12;}
.cython.score-126 {background-color: #FFFF12;}
.cython.score-127 {background-color: #FFFF12;}
.cython.score-128 {background-color: #FFFF12;}
.cython.score-129 {background-color: #FFFF12;}
.cython.score-130 {background-color: #FFFF12;}
.cython.score-131 {background-color: #FFFF12;}
.cython.score-132 {background-color: #FFFF11;}
.cython.score-133 {background-color: #FFFF11;}
.cython.score-134 {background-color: #FFFF11;}
.cython.score-135 {background-color: #FFFF11;}
.cython.score-136 {background-color: #FFFF11;}
.cython.score-137 {background-color: #FFFF11;}
.cython.score-138 {background-color: #FFFF11;}
.cython.score-139 {background-color: #FFFF11;}
.cython.score-140 {background-color: #FFFF11;}
.cython.score-141 {background-color: #FFFF10;}
.cython.score-142 {background-color: #FFFF10;}
.cython.score-143 {background-color: #FFFF10;}
.cython.score-144 {background-color: #FFFF10;}
.cython.score-145 {background-color: #FFFF10;}
.cython.score-146 {background-color: #FFFF10;}
.cython.score-147 {background-color: #FFFF10;}
.cython.score-148 {background-color: #FFFF10;}
.cython.score-149 {background-color: #FFFF10;}
.cython.score-150 {background-color: #FFFF0f;}
.cython.score-151 {background-color: #FFFF0f;}
.cython.score-152 {background-color: #FFFF0f;}
.cython.score-153 {background-color: #FFFF0f;}
.cython.score-154 {background-color: #FFFF0f;}
.cython.score-155 {background-color: #FFFF0f;}
.cython.score-156 {background-color: #FFFF0f;}
.cython.score-157 {background-color: #FFFF0f;}
.cython.score-158 {background-color: #FFFF0f;}
.cython.score-159 {background-color: #FFFF0f;}
.cython.score-160 {background-color: #FFFF0f;}
.cython.score-161 {background-color: #FFFF0e;}
.cython.score-162 {background-color: #FFFF0e;}
.cython.score-163 {background-color: #FFFF0e;}
.cython.score-164 {background-color: #FFFF0e;}
.cython.score-165 {background-color: #FFFF0e;}
.cython.score-166 {background-color: #FFFF0e;}
.cython.score-167 {background-color: #FFFF0e;}
.cython.score-168 {background-color: #FFFF0e;}
.cython.score-169 {background-color: #FFFF0e;}
.cython.score-170 {background-color: #FFFF0e;}
.cython.score-171 {background-color: #FFFF0e;}
.cython.score-172 {background-color: #FFFF0e;}
.cython.score-173 {background-color: #FFFF0d;}
.cython.score-174 {background-color: #FFFF0d;}
.cython.score-175 {background-color: #FFFF0d;}
.cython.score-176 {background-color: #FFFF0d;}
.cython.score-177 {background-color: #FFFF0d;}
.cython.score-178 {background-color: #FFFF0d;}
.cython.score-179 {background-color: #FFFF0d;}
.cython.score-180 {background-color: #FFFF0d;}
.cython.score-181 {background-color: #FFFF0d;}
.cython.score-182 {background-color: #FFFF0d;}
.cython.score-183 {background-color: #FFFF0d;}
.cython.score-184 {background-color: #FFFF0d;}
.cython.score-185 {background-color: #FFFF0d;}
.cython.score-186 {background-color: #FFFF0d;}
.cython.score-187 {background-color: #FFFF0c;}
.cython.score-188 {background-color: #FFFF0c;}
.cython.score-189 {background-color: #FFFF0c;}
.cython.score-190 {background-color: #FFFF0c;}
.cython.score-191 {background-color: #FFFF0c;}
.cython.score-192 {background-color: #FFFF0c;}
.cython.score-193 {background-color: #FFFF0c;}
.cython.score-194 {background-color: #FFFF0c;}
.cython.score-195 {background-color: #FFFF0c;}
.cython.score-196 {background-color: #FFFF0c;}
.cython.score-197 {background-color: #FFFF0c;}
.cython.score-198 {background-color: #FFFF0c;}
.cython.score-199 {background-color: #FFFF0c;}
.cython.score-200 {background-color: #FFFF0c;}
.cython.score-201 {background-color: #FFFF0c;}
.cython.score-202 {background-color: #FFFF0c;}
.cython.score-203 {background-color: #FFFF0b;}
.cython.score-204 {background-color: #FFFF0b;}
.cython.score-205 {background-color: #FFFF0b;}
.cython.score-206 {background-color: #FFFF0b;}
.cython.score-207 {background-color: #FFFF0b;}
.cython.score-208 {background-color: #FFFF0b;}
.cython.score-209 {background-color: #FFFF0b;}
.cython.score-210 {background-color: #FFFF0b;}
.cython.score-211 {background-color: #FFFF0b;}
.cython.score-212 {background-color: #FFFF0b;}
.cython.score-213 {background-color: #FFFF0b;}
.cython.score-214 {background-color: #FFFF0b;}
.cython.score-215 {background-color: #FFFF0b;}
.cython.score-216 {background-color: #FFFF0b;}
.cython.score-217 {background-color: #FFFF0b;}
.cython.score-218 {background-color: #FFFF0b;}
.cython.score-219 {background-color: #FFFF0b;}
.cython.score-220 {background-color: #FFFF0b;}
.cython.score-221 {background-color: #FFFF0b;}
.cython.score-222 {background-color: #FFFF0a;}
.cython.score-223 {background-color: #FFFF0a;}
.cython.score-224 {background-color: #FFFF0a;}
.cython.score-225 {background-color: #FFFF0a;}
.cython.score-226 {background-color: #FFFF0a;}
.cython.score-227 {background-color: #FFFF0a;}
.cython.score-228 {background-color: #FFFF0a;}
.cython.score-229 {background-color: #FFFF0a;}
.cython.score-230 {background-color: #FFFF0a;}
.cython.score-231 {background-color: #FFFF0a;}
.cython.score-232 {background-color: #FFFF0a;}
.cython.score-233 {background-color: #FFFF0a;}
.cython.score-234 {background-color: #FFFF0a;}
.cython.score-235 {background-color: #FFFF0a;}
.cython.score-236 {background-color: #FFFF0a;}
.cython.score-237 {background-color: #FFFF0a;}
.cython.score-238 {background-color: #FFFF0a;}
.cython.score-239 {background-color: #FFFF0a;}
.cython.score-240 {background-color: #FFFF0a;}
.cython.score-241 {background-color: #FFFF0a;}
.cython.score-242 {background-color: #FFFF0a;}
.cython.score-243 {background-color: #FFFF0a;}
.cython.score-244 {background-color: #FFFF0a;}
.cython.score-245 {background-color: #FFFF0a;}
.cython.score-246 {background-color: #FFFF09;}
.cython.score-247 {background-color: #FFFF09;}
.cython.score-248 {background-color: #FFFF09;}
.cython.score-249 {background-color: #FFFF09;}
.cython.score-250 {background-color: #FFFF09;}
.cython.score-251 {background-color: #FFFF09;}
.cython.score-252 {background-color: #FFFF09;}
.cython.score-253 {background-color: #FFFF09;}
.cython.score-254 {background-color: #FFFF09;}
pre { line-height: 125%; }
td.linenos .normal { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
span.linenos { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
td.linenos .special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
span.linenos.special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
.cython .hll { background-color: #ffffcc }
.cython { background: #f8f8f8; }
.cython .c { color: #3D7B7B; font-style: italic } /* Comment */
.cython .err { border: 1px solid #FF0000 } /* Error */
.cython .k { color: #008000; font-weight: bold } /* Keyword */
.cython .o { color: #666666 } /* Operator */
.cython .ch { color: #3D7B7B; font-style: italic } /* Comment.Hashbang */
.cython .cm { color: #3D7B7B; font-style: italic } /* Comment.Multiline */
.cython .cp { color: #9C6500 } /* Comment.Preproc */
.cython .cpf { color: #3D7B7B; font-style: italic } /* Comment.PreprocFile */
.cython .c1 { color: #3D7B7B; font-style: italic } /* Comment.Single */
.cython .cs { color: #3D7B7B; font-style: italic } /* Comment.Special */
.cython .gd { color: #A00000 } /* Generic.Deleted */
.cython .ge { font-style: italic } /* Generic.Emph */
.cython .gr { color: #E40000 } /* Generic.Error */
.cython .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.cython .gi { color: #008400 } /* Generic.Inserted */
.cython .go { color: #717171 } /* Generic.Output */
.cython .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.cython .gs { font-weight: bold } /* Generic.Strong */
.cython .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.cython .gt { color: #0044DD } /* Generic.Traceback */
.cython .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.cython .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.cython .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.cython .kp { color: #008000 } /* Keyword.Pseudo */
.cython .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.cython .kt { color: #B00040 } /* Keyword.Type */
.cython .m { color: #666666 } /* Literal.Number */
.cython .s { color: #BA2121 } /* Literal.String */
.cython .na { color: #687822 } /* Name.Attribute */
.cython .nb { color: #008000 } /* Name.Builtin */
.cython .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.cython .no { color: #880000 } /* Name.Constant */
.cython .nd { color: #AA22FF } /* Name.Decorator */
.cython .ni { color: #717171; font-weight: bold } /* Name.Entity */
.cython .ne { color: #CB3F38; font-weight: bold } /* Name.Exception */
.cython .nf { color: #0000FF } /* Name.Function */
.cython .nl { color: #767600 } /* Name.Label */
.cython .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.cython .nt { color: #008000; font-weight: bold } /* Name.Tag */
.cython .nv { color: #19177C } /* Name.Variable */
.cython .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.cython .w { color: #bbbbbb } /* Text.Whitespace */
.cython .mb { color: #666666 } /* Literal.Number.Bin */
.cython .mf { color: #666666 } /* Literal.Number.Float */
.cython .mh { color: #666666 } /* Literal.Number.Hex */
.cython .mi { color: #666666 } /* Literal.Number.Integer */
.cython .mo { color: #666666 } /* Literal.Number.Oct */
.cython .sa { color: #BA2121 } /* Literal.String.Affix */
.cython .sb { color: #BA2121 } /* Literal.String.Backtick */
.cython .sc { color: #BA2121 } /* Literal.String.Char */
.cython .dl { color: #BA2121 } /* Literal.String.Delimiter */
.cython .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.cython .s2 { color: #BA2121 } /* Literal.String.Double */
.cython .se { color: #AA5D1F; font-weight: bold } /* Literal.String.Escape */
.cython .sh { color: #BA2121 } /* Literal.String.Heredoc */
.cython .si { color: #A45A77; font-weight: bold } /* Literal.String.Interpol */
.cython .sx { color: #008000 } /* Literal.String.Other */
.cython .sr { color: #A45A77 } /* Literal.String.Regex */
.cython .s1 { color: #BA2121 } /* Literal.String.Single */
.cython .ss { color: #19177C } /* Literal.String.Symbol */
.cython .bp { color: #008000 } /* Name.Builtin.Pseudo */
.cython .fm { color: #0000FF } /* Name.Function.Magic */
.cython .vc { color: #19177C } /* Name.Variable.Class */
.cython .vg { color: #19177C } /* Name.Variable.Global */
.cython .vi { color: #19177C } /* Name.Variable.Instance */
.cython .vm { color: #19177C } /* Name.Variable.Magic */
.cython .il { color: #666666 } /* Literal.Number.Integer.Long */
    </style>
</head>
<body class="cython">
<p><span style="border-bottom: solid 1px grey;">Generated by Cython 0.29.33</span></p>
<p>
    <span style="background-color: #FFFF00">Yellow lines</span> hint at Python interaction.<br />
    Click on a line that starts with a "<code>+</code>" to see the C code that Cython generated for it.
</p>
<div class="cython"><pre class="cython line score-0">&#xA0;<span class="">01</span>: </pre>
<pre class="cython line score-16" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">02</span>: <span class="k">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span></pre>
<pre class='cython code score-16 '>  __pyx_t_1 = <span class='pyx_c_api'>__Pyx_Import</span>(__pyx_n_s_numpy, 0, 0);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 2, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_np, __pyx_t_1) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 2, __pyx_L1_error)</span>
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
/* … */
  __pyx_t_2 = <span class='pyx_c_api'>__Pyx_PyDict_NewPresized</span>(0);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 2, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_test, __pyx_t_2) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 2, __pyx_L1_error)</span>
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
</pre><pre class="cython line score-0">&#xA0;<span class="">03</span>: <span class="k">cimport</span> <span class="nn">cython</span></pre>
<pre class="cython line score-0">&#xA0;<span class="">04</span>: <span class="k">from</span> <span class="nn">cython.parallel</span> <span class="k">import</span> <span class="n">prange</span></pre>
<pre class="cython line score-0">&#xA0;<span class="">05</span>: </pre>
<pre class="cython line score-0">&#xA0;<span class="">06</span>: <span class="k">ctypedef</span> <span class="k">fused</span> <span class="n">my_type</span><span class="p">:</span></pre>
<pre class="cython line score-0">&#xA0;<span class="">07</span>:     <span class="nb">int</span></pre>
<pre class="cython line score-0">&#xA0;<span class="">08</span>:     <span class="n">double</span></pre>
<pre class="cython line score-0">&#xA0;<span class="">09</span>:     <span class="nb">long</span> <span class="nb">long</span></pre>
<pre class="cython line score-0">&#xA0;<span class="">10</span>: </pre>
<pre class="cython line score-0" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">11</span>: <span class="k">cdef</span> <span class="kt">my_type</span> <span class="nf">clip</span><span class="p">(</span><span class="n">my_type</span> <span class="n">a</span><span class="p">,</span> <span class="n">my_type</span> <span class="n">min_value</span><span class="p">,</span> <span class="n">my_type</span> <span class="n">max_value</span><span class="p">)</span> <span class="k">nogil</span><span class="p">:</span></pre>
<pre class='cython code score-0 '>static int __pyx_fuse_0__pyx_f_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_clip(int __pyx_v_a, int __pyx_v_min_value, int __pyx_v_max_value) {
  int __pyx_r;
/* … */
  /* function exit code */
  __pyx_L0:;
  return __pyx_r;
}

static double __pyx_fuse_1__pyx_f_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_clip(double __pyx_v_a, double __pyx_v_min_value, double __pyx_v_max_value) {
  double __pyx_r;
/* … */
  /* function exit code */
  __pyx_L0:;
  return __pyx_r;
}

static PY_LONG_LONG __pyx_fuse_2__pyx_f_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_clip(PY_LONG_LONG __pyx_v_a, PY_LONG_LONG __pyx_v_min_value, PY_LONG_LONG __pyx_v_max_value) {
  PY_LONG_LONG __pyx_r;
/* … */
  /* function exit code */
  __pyx_L0:;
  return __pyx_r;
}
</pre><pre class="cython line score-0" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">12</span>:     <span class="k">return</span> <span class="nb">min</span><span class="p">(</span><span class="nb">max</span><span class="p">(</span><span class="n">a</span><span class="p">,</span> <span class="n">min_value</span><span class="p">),</span> <span class="n">max_value</span><span class="p">)</span></pre>
<pre class='cython code score-0 '>  __pyx_t_1 = __pyx_v_max_value;
  __pyx_t_2 = __pyx_v_min_value;
  __pyx_t_3 = __pyx_v_a;
  if (((__pyx_t_2 &gt; __pyx_t_3) != 0)) {
    __pyx_t_4 = __pyx_t_2;
  } else {
    __pyx_t_4 = __pyx_t_3;
  }
  __pyx_t_2 = __pyx_t_4;
  if (((__pyx_t_1 &lt; __pyx_t_2) != 0)) {
    __pyx_t_4 = __pyx_t_1;
  } else {
    __pyx_t_4 = __pyx_t_2;
  }
  __pyx_r = __pyx_t_4;
  goto __pyx_L0;
/* … */
  __pyx_t_1 = __pyx_v_max_value;
  __pyx_t_2 = __pyx_v_min_value;
  __pyx_t_3 = __pyx_v_a;
  if (((__pyx_t_2 &gt; __pyx_t_3) != 0)) {
    __pyx_t_4 = __pyx_t_2;
  } else {
    __pyx_t_4 = __pyx_t_3;
  }
  __pyx_t_2 = __pyx_t_4;
  if (((__pyx_t_1 &lt; __pyx_t_2) != 0)) {
    __pyx_t_4 = __pyx_t_1;
  } else {
    __pyx_t_4 = __pyx_t_2;
  }
  __pyx_r = __pyx_t_4;
  goto __pyx_L0;
/* … */
  __pyx_t_1 = __pyx_v_max_value;
  __pyx_t_2 = __pyx_v_min_value;
  __pyx_t_3 = __pyx_v_a;
  if (((__pyx_t_2 &gt; __pyx_t_3) != 0)) {
    __pyx_t_4 = __pyx_t_2;
  } else {
    __pyx_t_4 = __pyx_t_3;
  }
  __pyx_t_2 = __pyx_t_4;
  if (((__pyx_t_1 &lt; __pyx_t_2) != 0)) {
    __pyx_t_4 = __pyx_t_1;
  } else {
    __pyx_t_4 = __pyx_t_2;
  }
  __pyx_r = __pyx_t_4;
  goto __pyx_L0;
</pre><pre class="cython line score-0">&#xA0;<span class="">13</span>: </pre>
<pre class="cython line score-0">&#xA0;<span class="">14</span>: <span class="nd">@cython</span><span class="o">.</span><span class="n">boundscheck</span><span class="p">(</span><span class="bp">False</span><span class="p">)</span></pre>
<pre class="cython line score-0">&#xA0;<span class="">15</span>: <span class="nd">@cython</span><span class="o">.</span><span class="n">wraparound</span><span class="p">(</span><span class="bp">False</span><span class="p">)</span></pre>
<pre class="cython line score-586" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">16</span>: <span class="k">def</span> <span class="nf">compute_cy_mt</span><span class="p">(</span><span class="n">my_type</span><span class="p">[:,</span> <span class="p">::</span><span class="mf">1</span><span class="p">]</span> <span class="n">array_1</span><span class="p">,</span> <span class="n">my_type</span><span class="p">[:,</span> <span class="p">::</span><span class="mf">1</span><span class="p">]</span> <span class="n">array_2</span><span class="p">,</span> <span class="n">my_type</span> <span class="n">a</span><span class="p">,</span> <span class="n">my_type</span> <span class="n">b</span><span class="p">,</span> <span class="n">my_type</span> <span class="n">c</span><span class="p">):</span></pre>
<pre class='cython code score-586 '>/* Python wrapper */
static PyObject *__pyx_pw_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_1compute_cy_mt(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyMethodDef __pyx_mdef_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_1compute_cy_mt = {"compute_cy_mt", (PyCFunction)(void*)(PyCFunctionWithKeywords)__pyx_pw_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_1compute_cy_mt, METH_VARARGS|METH_KEYWORDS, 0};
static PyObject *__pyx_pw_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_1compute_cy_mt(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_signatures = 0;
  PyObject *__pyx_v_args = 0;
  PyObject *__pyx_v_kwargs = 0;
  CYTHON_UNUSED PyObject *__pyx_v_defaults = 0;
  PyObject *__pyx_r = 0;
  <span class='refnanny'>__Pyx_RefNannyDeclarations</span>
  <span class='refnanny'>__Pyx_RefNannySetupContext</span>("__pyx_fused_cpdef (wrapper)", 0);
  {
    static PyObject **__pyx_pyargnames[] = {&amp;__pyx_n_s_signatures,&amp;__pyx_n_s_args,&amp;__pyx_n_s_kwargs,&amp;__pyx_n_s_defaults,0};
    PyObject* values[4] = {0,0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      const Py_ssize_t pos_args = <span class='py_macro_api'>PyTuple_GET_SIZE</span>(__pyx_args);
      switch (pos_args) {
        case  4: values[3] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 3);
        CYTHON_FALLTHROUGH;
        case  3: values[2] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 2);
        CYTHON_FALLTHROUGH;
        case  2: values[1] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 1);
        CYTHON_FALLTHROUGH;
        case  1: values[0] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 0);
        CYTHON_FALLTHROUGH;
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = <span class='py_c_api'>PyDict_Size</span>(__pyx_kwds);
      switch (pos_args) {
        case  0:
        if (likely((values[0] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_signatures)) != 0)) kw_args--;
        else goto __pyx_L5_argtuple_error;
        CYTHON_FALLTHROUGH;
        case  1:
        if (likely((values[1] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_args)) != 0)) kw_args--;
        else {
          <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("__pyx_fused_cpdef", 1, 4, 4, 1); <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
        }
        CYTHON_FALLTHROUGH;
        case  2:
        if (likely((values[2] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_kwargs)) != 0)) kw_args--;
        else {
          <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("__pyx_fused_cpdef", 1, 4, 4, 2); <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
        }
        CYTHON_FALLTHROUGH;
        case  3:
        if (likely((values[3] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_defaults)) != 0)) kw_args--;
        else {
          <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("__pyx_fused_cpdef", 1, 4, 4, 3); <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
        }
      }
      if (unlikely(kw_args &gt; 0)) {
        if (unlikely(<span class='pyx_c_api'>__Pyx_ParseOptionalKeywords</span>(__pyx_kwds, __pyx_pyargnames, 0, values, pos_args, "__pyx_fused_cpdef") &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
      }
    } else if (<span class='py_macro_api'>PyTuple_GET_SIZE</span>(__pyx_args) != 4) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 0);
      values[1] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 1);
      values[2] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 2);
      values[3] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 3);
    }
    __pyx_v_signatures = values[0];
    __pyx_v_args = values[1];
    __pyx_v_kwargs = values[2];
    __pyx_v_defaults = values[3];
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("__pyx_fused_cpdef", 1, 4, 4, <span class='py_macro_api'>PyTuple_GET_SIZE</span>(__pyx_args)); <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
  __pyx_L3_error:;
  <span class='pyx_c_api'>__Pyx_AddTraceback</span>("_cython_magic_cd184590f55726778fe97a651b5f5dd0.__pyx_fused_cpdef", __pyx_clineno, __pyx_lineno, __pyx_filename);
  <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __pyx_r = __pyx_pf_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_compute_cy_mt(__pyx_self, __pyx_v_signatures, __pyx_v_args, __pyx_v_kwargs, __pyx_v_defaults);
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;

  /* function exit code */
  <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
  return __pyx_r;
}

static PyObject *__pyx_pf_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_compute_cy_mt(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_signatures, PyObject *__pyx_v_args, PyObject *__pyx_v_kwargs, CYTHON_UNUSED PyObject *__pyx_v_defaults) {
  PyObject *__pyx_v_dest_sig = NULL;
  Py_ssize_t __pyx_v_i;
  PyTypeObject *__pyx_v_ndarray = 0;
  __Pyx_memviewslice __pyx_v_memslice;
  Py_ssize_t __pyx_v_itemsize;
  int __pyx_v_dtype_signed;
  char __pyx_v_kind;
  int __pyx_v_int_is_signed;
  int __pyx_v_long_long_is_signed;
  PyObject *__pyx_v_arg = NULL;
  PyObject *__pyx_v_dtype = NULL;
  PyObject *__pyx_v_arg_base = NULL;
  PyObject *__pyx_v_candidates = NULL;
  PyObject *__pyx_v_sig = NULL;
  int __pyx_v_match_found;
  PyObject *__pyx_v_src_sig = NULL;
  PyObject *__pyx_v_dst_type = NULL;
  PyObject *__pyx_r = NULL;
  <span class='refnanny'>__Pyx_RefNannyDeclarations</span>
  <span class='refnanny'>__Pyx_RefNannySetupContext</span>("compute_cy_mt", 0);
  <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_v_kwargs);
  __pyx_t_1 = <span class='py_c_api'>PyList_New</span>(1);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  <span class='pyx_macro_api'>__Pyx_INCREF</span>(Py_None);
  <span class='refnanny'>__Pyx_GIVEREF</span>(Py_None);
  <span class='py_macro_api'>PyList_SET_ITEM</span>(__pyx_t_1, 0, Py_None);
  __pyx_v_dest_sig = ((PyObject*)__pyx_t_1);
  __pyx_t_1 = 0;
  __pyx_t_3 = (__pyx_v_kwargs != Py_None);
  __pyx_t_4 = (__pyx_t_3 != 0);
  if (__pyx_t_4) {
  } else {
    __pyx_t_2 = __pyx_t_4;
    goto __pyx_L4_bool_binop_done;
  }
  __pyx_t_4 = <span class='pyx_c_api'>__Pyx_PyObject_IsTrue</span>(__pyx_v_kwargs); if (unlikely(__pyx_t_4 &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
  __pyx_t_3 = ((!__pyx_t_4) != 0);
  __pyx_t_2 = __pyx_t_3;
  __pyx_L4_bool_binop_done:;
  if (__pyx_t_2) {
    <span class='pyx_macro_api'>__Pyx_INCREF</span>(Py_None);
    <span class='pyx_macro_api'>__Pyx_DECREF_SET</span>(__pyx_v_kwargs, Py_None);
  }
  __pyx_t_1 = ((PyObject *)<span class='pyx_c_api'>__Pyx_ImportNumPyArrayTypeIfAvailable</span>());<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  __pyx_v_ndarray = ((PyTypeObject*)__pyx_t_1);
  __pyx_t_1 = 0;
  __pyx_v_itemsize = -1L;
  __pyx_v_int_is_signed = (!((((int)-1L) &gt; 0) != 0));
  __pyx_v_long_long_is_signed = (!((((PY_LONG_LONG)-1L) &gt; 0) != 0));
  if (unlikely(__pyx_v_args == Py_None)) {
    <span class='py_c_api'>PyErr_SetString</span>(PyExc_TypeError, "object of type 'NoneType' has no len()");
    <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
  }
  __pyx_t_5 = <span class='py_macro_api'>PyTuple_GET_SIZE</span>(((PyObject*)__pyx_v_args));<span class='error_goto'> if (unlikely(__pyx_t_5 == ((Py_ssize_t)-1))) __PYX_ERR(0, 16, __pyx_L1_error)</span>
  __pyx_t_2 = ((0 &lt; __pyx_t_5) != 0);
  if (__pyx_t_2) {
    if (unlikely(__pyx_v_args == Py_None)) {
      <span class='py_c_api'>PyErr_SetString</span>(PyExc_TypeError, "'NoneType' object is not subscriptable");
      <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
    }
    __pyx_t_1 = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(((PyObject*)__pyx_v_args), 0);
    <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_t_1);
    __pyx_v_arg = __pyx_t_1;
    __pyx_t_1 = 0;
    goto __pyx_L6;
  }
  __pyx_t_3 = (__pyx_v_kwargs != Py_None);
  __pyx_t_4 = (__pyx_t_3 != 0);
  if (__pyx_t_4) {
  } else {
    __pyx_t_2 = __pyx_t_4;
    goto __pyx_L7_bool_binop_done;
  }
  if (unlikely(__pyx_v_kwargs == Py_None)) {
    <span class='py_c_api'>PyErr_SetString</span>(PyExc_TypeError, "'NoneType' object is not iterable");
    <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
  }
  __pyx_t_4 = (<span class='pyx_c_api'>__Pyx_PyDict_ContainsTF</span>(__pyx_n_s_array_1, ((PyObject*)__pyx_v_kwargs), Py_EQ)); if (unlikely(__pyx_t_4 &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
  __pyx_t_3 = (__pyx_t_4 != 0);
  __pyx_t_2 = __pyx_t_3;
  __pyx_L7_bool_binop_done:;
  if (__pyx_t_2) {
    if (unlikely(__pyx_v_kwargs == Py_None)) {
      <span class='py_c_api'>PyErr_SetString</span>(PyExc_TypeError, "'NoneType' object is not subscriptable");
      <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
    }
    __pyx_t_1 = <span class='pyx_c_api'>__Pyx_PyDict_GetItem</span>(((PyObject*)__pyx_v_kwargs), __pyx_n_s_array_1);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
    __pyx_v_arg = __pyx_t_1;
    __pyx_t_1 = 0;
    goto __pyx_L6;
  }
  /*else*/ {
    if (unlikely(__pyx_v_args == Py_None)) {
      <span class='py_c_api'>PyErr_SetString</span>(PyExc_TypeError, "object of type 'NoneType' has no len()");
      <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
    }
    __pyx_t_5 = <span class='py_macro_api'>PyTuple_GET_SIZE</span>(((PyObject*)__pyx_v_args));<span class='error_goto'> if (unlikely(__pyx_t_5 == ((Py_ssize_t)-1))) __PYX_ERR(0, 16, __pyx_L1_error)</span>
    __pyx_t_1 = <span class='py_c_api'>PyInt_FromSsize_t</span>(__pyx_t_5);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
    __pyx_t_6 = <span class='py_c_api'>PyTuple_New</span>(3);<span class='error_goto'> if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_6);
    <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_int_5);
    <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_int_5);
    <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_6, 0, __pyx_int_5);
    <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_n_s_s);
    <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_n_s_s);
    <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_6, 1, __pyx_n_s_s);
    <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_t_1);
    <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_6, 2, __pyx_t_1);
    __pyx_t_1 = 0;
    __pyx_t_1 = <span class='pyx_c_api'>__Pyx_PyString_Format</span>(__pyx_kp_s_Expected_at_least_d_argument_s_g, __pyx_t_6);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_6); __pyx_t_6 = 0;
    __pyx_t_6 = <span class='pyx_c_api'>__Pyx_PyObject_CallOneArg</span>(__pyx_builtin_TypeError, __pyx_t_1);<span class='error_goto'> if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_6);
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
    <span class='pyx_c_api'>__Pyx_Raise</span>(__pyx_t_6, 0, 0, 0);
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_6); __pyx_t_6 = 0;
    <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
  }
  __pyx_L6:;
  while (1) {
    __pyx_t_2 = (__pyx_v_ndarray != ((PyTypeObject*)Py_None));
    __pyx_t_3 = (__pyx_t_2 != 0);
    if (__pyx_t_3) {
      __pyx_t_3 = <span class='pyx_c_api'>__Pyx_TypeCheck</span>(__pyx_v_arg, __pyx_v_ndarray); 
      __pyx_t_2 = (__pyx_t_3 != 0);
      if (__pyx_t_2) {
        __pyx_t_6 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_v_arg, __pyx_n_s_dtype);<span class='error_goto'> if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
        <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_6);
        __pyx_v_dtype = __pyx_t_6;
        __pyx_t_6 = 0;
        goto __pyx_L12;
      }
      __pyx_t_2 = __pyx_memoryview_check(__pyx_v_arg); 
      __pyx_t_3 = (__pyx_t_2 != 0);
      if (__pyx_t_3) {
        __pyx_t_6 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_v_arg, __pyx_n_s_base);<span class='error_goto'> if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
        <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_6);
        __pyx_v_arg_base = __pyx_t_6;
        __pyx_t_6 = 0;
        __pyx_t_3 = <span class='pyx_c_api'>__Pyx_TypeCheck</span>(__pyx_v_arg_base, __pyx_v_ndarray); 
        __pyx_t_2 = (__pyx_t_3 != 0);
        if (__pyx_t_2) {
          __pyx_t_6 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_v_arg_base, __pyx_n_s_dtype);<span class='error_goto'> if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
          <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_6);
          __pyx_v_dtype = __pyx_t_6;
          __pyx_t_6 = 0;
          goto __pyx_L13;
        }
        /*else*/ {
          <span class='pyx_macro_api'>__Pyx_INCREF</span>(Py_None);
          __pyx_v_dtype = Py_None;
        }
        __pyx_L13:;
        goto __pyx_L12;
      }
      /*else*/ {
        <span class='pyx_macro_api'>__Pyx_INCREF</span>(Py_None);
        __pyx_v_dtype = Py_None;
      }
      __pyx_L12:;
      __pyx_v_itemsize = -1L;
      __pyx_t_2 = (__pyx_v_dtype != Py_None);
      __pyx_t_3 = (__pyx_t_2 != 0);
      if (__pyx_t_3) {
        __pyx_t_6 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_v_dtype, __pyx_n_s_itemsize);<span class='error_goto'> if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
        <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_6);
        __pyx_t_5 = <span class='pyx_c_api'>__Pyx_PyIndex_AsSsize_t</span>(__pyx_t_6); if (unlikely((__pyx_t_5 == (Py_ssize_t)-1) &amp;&amp; <span class='py_c_api'>PyErr_Occurred</span>())) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
        <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_6); __pyx_t_6 = 0;
        __pyx_v_itemsize = __pyx_t_5;
        __pyx_t_6 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_v_dtype, __pyx_n_s_kind);<span class='error_goto'> if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
        <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_6);
        __pyx_t_7 = <span class='pyx_c_api'>__Pyx_PyObject_Ord</span>(__pyx_t_6);<span class='error_goto'> if (unlikely(__pyx_t_7 == ((long)(long)(Py_UCS4)-1))) __PYX_ERR(0, 16, __pyx_L1_error)</span>
        <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_6); __pyx_t_6 = 0;
        __pyx_v_kind = __pyx_t_7;
        __pyx_v_dtype_signed = (__pyx_v_kind == 'i');
        switch (__pyx_v_kind) {
          case 'i':
          case 'u':
          __pyx_t_2 = (((sizeof(int)) == __pyx_v_itemsize) != 0);
          if (__pyx_t_2) {
          } else {
            __pyx_t_3 = __pyx_t_2;
            goto __pyx_L16_bool_binop_done;
          }
          __pyx_t_6 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_v_arg, __pyx_n_s_ndim);<span class='error_goto'> if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
          <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_6);
          __pyx_t_5 = <span class='pyx_c_api'>__Pyx_PyIndex_AsSsize_t</span>(__pyx_t_6); if (unlikely((__pyx_t_5 == (Py_ssize_t)-1) &amp;&amp; <span class='py_c_api'>PyErr_Occurred</span>())) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
          <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_6); __pyx_t_6 = 0;
          __pyx_t_2 = ((((Py_ssize_t)__pyx_t_5) == 2) != 0);
          if (__pyx_t_2) {
          } else {
            __pyx_t_3 = __pyx_t_2;
            goto __pyx_L16_bool_binop_done;
          }
          __pyx_t_2 = ((!((__pyx_v_int_is_signed ^ __pyx_v_dtype_signed) != 0)) != 0);
          __pyx_t_3 = __pyx_t_2;
          __pyx_L16_bool_binop_done:;
          if (__pyx_t_3) {
            if (unlikely(<span class='pyx_c_api'>__Pyx_SetItemInt</span>(__pyx_v_dest_sig, 0, __pyx_n_s_int, long, 1, __Pyx_PyInt_From_long, 1, 0, 0) &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
            goto __pyx_L10_break;
          }
          __pyx_t_2 = (((sizeof(PY_LONG_LONG)) == __pyx_v_itemsize) != 0);
          if (__pyx_t_2) {
          } else {
            __pyx_t_3 = __pyx_t_2;
            goto __pyx_L20_bool_binop_done;
          }
          __pyx_t_6 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_v_arg, __pyx_n_s_ndim);<span class='error_goto'> if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
          <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_6);
          __pyx_t_5 = <span class='pyx_c_api'>__Pyx_PyIndex_AsSsize_t</span>(__pyx_t_6); if (unlikely((__pyx_t_5 == (Py_ssize_t)-1) &amp;&amp; <span class='py_c_api'>PyErr_Occurred</span>())) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
          <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_6); __pyx_t_6 = 0;
          __pyx_t_2 = ((((Py_ssize_t)__pyx_t_5) == 2) != 0);
          if (__pyx_t_2) {
          } else {
            __pyx_t_3 = __pyx_t_2;
            goto __pyx_L20_bool_binop_done;
          }
          __pyx_t_2 = ((!((__pyx_v_long_long_is_signed ^ __pyx_v_dtype_signed) != 0)) != 0);
          __pyx_t_3 = __pyx_t_2;
          __pyx_L20_bool_binop_done:;
          if (__pyx_t_3) {
            if (unlikely(<span class='pyx_c_api'>__Pyx_SetItemInt</span>(__pyx_v_dest_sig, 0, __pyx_kp_s_long_long, long, 1, __Pyx_PyInt_From_long, 1, 0, 0) &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
            goto __pyx_L10_break;
          }
          break;
          case 'f':
          __pyx_t_2 = (((sizeof(double)) == __pyx_v_itemsize) != 0);
          if (__pyx_t_2) {
          } else {
            __pyx_t_3 = __pyx_t_2;
            goto __pyx_L24_bool_binop_done;
          }
          __pyx_t_6 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_v_arg, __pyx_n_s_ndim);<span class='error_goto'> if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
          <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_6);
          __pyx_t_5 = <span class='pyx_c_api'>__Pyx_PyIndex_AsSsize_t</span>(__pyx_t_6); if (unlikely((__pyx_t_5 == (Py_ssize_t)-1) &amp;&amp; <span class='py_c_api'>PyErr_Occurred</span>())) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
          <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_6); __pyx_t_6 = 0;
          __pyx_t_2 = ((((Py_ssize_t)__pyx_t_5) == 2) != 0);
          __pyx_t_3 = __pyx_t_2;
          __pyx_L24_bool_binop_done:;
          if (__pyx_t_3) {
            if (unlikely(<span class='pyx_c_api'>__Pyx_SetItemInt</span>(__pyx_v_dest_sig, 0, __pyx_n_s_double, long, 1, __Pyx_PyInt_From_long, 1, 0, 0) &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
            goto __pyx_L10_break;
          }
          break;
          case 'c':
          break;
          case 'O':
          break;
          default: break;
        }
      }
    }
    __pyx_t_2 = ((__pyx_v_itemsize == -1L) != 0);
    if (!__pyx_t_2) {
    } else {
      __pyx_t_3 = __pyx_t_2;
      goto __pyx_L27_bool_binop_done;
    }
    __pyx_t_2 = ((__pyx_v_itemsize == (sizeof(int))) != 0);
    __pyx_t_3 = __pyx_t_2;
    __pyx_L27_bool_binop_done:;
    if (__pyx_t_3) {
      __pyx_t_8 = <span class='pyx_c_api'>__Pyx_PyObject_to_MemoryviewSlice_d_dc_int</span>(__pyx_v_arg, 0); 
      __pyx_v_memslice = __pyx_t_8;
      __pyx_t_3 = (__pyx_v_memslice.memview != 0);
      if (__pyx_t_3) {
        __PYX_XDEC_MEMVIEW((&amp;__pyx_v_memslice), 1); 
        if (unlikely(<span class='pyx_c_api'>__Pyx_SetItemInt</span>(__pyx_v_dest_sig, 0, __pyx_n_s_int, long, 1, __Pyx_PyInt_From_long, 1, 0, 0) &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
        goto __pyx_L10_break;
      }
      /*else*/ {
        <span class='py_c_api'>PyErr_Clear</span>(); 
      }
    }
    __pyx_t_2 = ((__pyx_v_itemsize == -1L) != 0);
    if (!__pyx_t_2) {
    } else {
      __pyx_t_3 = __pyx_t_2;
      goto __pyx_L31_bool_binop_done;
    }
    __pyx_t_2 = ((__pyx_v_itemsize == (sizeof(double))) != 0);
    __pyx_t_3 = __pyx_t_2;
    __pyx_L31_bool_binop_done:;
    if (__pyx_t_3) {
      __pyx_t_8 = <span class='pyx_c_api'>__Pyx_PyObject_to_MemoryviewSlice_d_dc_double</span>(__pyx_v_arg, 0); 
      __pyx_v_memslice = __pyx_t_8;
      __pyx_t_3 = (__pyx_v_memslice.memview != 0);
      if (__pyx_t_3) {
        __PYX_XDEC_MEMVIEW((&amp;__pyx_v_memslice), 1); 
        if (unlikely(<span class='pyx_c_api'>__Pyx_SetItemInt</span>(__pyx_v_dest_sig, 0, __pyx_n_s_double, long, 1, __Pyx_PyInt_From_long, 1, 0, 0) &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
        goto __pyx_L10_break;
      }
      /*else*/ {
        <span class='py_c_api'>PyErr_Clear</span>(); 
      }
    }
    __pyx_t_2 = ((__pyx_v_itemsize == -1L) != 0);
    if (!__pyx_t_2) {
    } else {
      __pyx_t_3 = __pyx_t_2;
      goto __pyx_L35_bool_binop_done;
    }
    __pyx_t_2 = ((__pyx_v_itemsize == (sizeof(PY_LONG_LONG))) != 0);
    __pyx_t_3 = __pyx_t_2;
    __pyx_L35_bool_binop_done:;
    if (__pyx_t_3) {
      __pyx_t_8 = <span class='pyx_c_api'>__Pyx_PyObject_to_MemoryviewSlice_d_dc_PY_LONG_LONG</span>(__pyx_v_arg, 0); 
      __pyx_v_memslice = __pyx_t_8;
      __pyx_t_3 = (__pyx_v_memslice.memview != 0);
      if (__pyx_t_3) {
        __PYX_XDEC_MEMVIEW((&amp;__pyx_v_memslice), 1); 
        if (unlikely(<span class='pyx_c_api'>__Pyx_SetItemInt</span>(__pyx_v_dest_sig, 0, __pyx_kp_s_long_long, long, 1, __Pyx_PyInt_From_long, 1, 0, 0) &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
        goto __pyx_L10_break;
      }
      /*else*/ {
        <span class='py_c_api'>PyErr_Clear</span>(); 
      }
    }
    if (unlikely(<span class='pyx_c_api'>__Pyx_SetItemInt</span>(__pyx_v_dest_sig, 0, Py_None, long, 1, __Pyx_PyInt_From_long, 1, 0, 0) &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
    goto __pyx_L10_break;
  }
  __pyx_L10_break:;
  __pyx_t_6 = <span class='py_c_api'>PyList_New</span>(0);<span class='error_goto'> if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_6);
  __pyx_v_candidates = ((PyObject*)__pyx_t_6);
  __pyx_t_6 = 0;
  __pyx_t_5 = 0;
  if (unlikely(__pyx_v_signatures == Py_None)) {
    <span class='py_c_api'>PyErr_SetString</span>(PyExc_TypeError, "'NoneType' object is not iterable");
    <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
  }
  __pyx_t_1 = __Pyx_dict_iterator(((PyObject*)__pyx_v_signatures), 1, ((PyObject *)NULL), (&amp;__pyx_t_9), (&amp;__pyx_t_10));<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_6);
  __pyx_t_6 = __pyx_t_1;
  __pyx_t_1 = 0;
  while (1) {
    __pyx_t_11 = __Pyx_dict_iter_next(__pyx_t_6, __pyx_t_9, &amp;__pyx_t_5, &amp;__pyx_t_1, NULL, NULL, __pyx_t_10);
    if (unlikely(__pyx_t_11 == 0)) break;
    if (unlikely(__pyx_t_11 == -1)) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
    <span class='pyx_macro_api'>__Pyx_XDECREF_SET</span>(__pyx_v_sig, __pyx_t_1);
    __pyx_t_1 = 0;
    __pyx_v_match_found = 0;
    __pyx_t_13 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_v_sig, __pyx_n_s_strip);<span class='error_goto'> if (unlikely(!__pyx_t_13)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_13);
    __pyx_t_14 = NULL;
    if (CYTHON_UNPACK_METHODS &amp;&amp; likely(<span class='py_c_api'>PyMethod_Check</span>(__pyx_t_13))) {
      __pyx_t_14 = <span class='py_macro_api'>PyMethod_GET_SELF</span>(__pyx_t_13);
      if (likely(__pyx_t_14)) {
        PyObject* function = <span class='py_macro_api'>PyMethod_GET_FUNCTION</span>(__pyx_t_13);
        <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_t_14);
        <span class='pyx_macro_api'>__Pyx_INCREF</span>(function);
        <span class='pyx_macro_api'>__Pyx_DECREF_SET</span>(__pyx_t_13, function);
      }
    }
    __pyx_t_12 = (__pyx_t_14) ? __Pyx_PyObject_Call2Args(__pyx_t_13, __pyx_t_14, __pyx_kp_s_) : <span class='pyx_c_api'>__Pyx_PyObject_CallOneArg</span>(__pyx_t_13, __pyx_kp_s_);
    <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_14); __pyx_t_14 = 0;
    if (unlikely(!__pyx_t_12)) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_12);
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_13); __pyx_t_13 = 0;
    __pyx_t_13 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_t_12, __pyx_n_s_split);<span class='error_goto'> if (unlikely(!__pyx_t_13)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_13);
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_12); __pyx_t_12 = 0;
    __pyx_t_12 = NULL;
    if (CYTHON_UNPACK_METHODS &amp;&amp; likely(<span class='py_c_api'>PyMethod_Check</span>(__pyx_t_13))) {
      __pyx_t_12 = <span class='py_macro_api'>PyMethod_GET_SELF</span>(__pyx_t_13);
      if (likely(__pyx_t_12)) {
        PyObject* function = <span class='py_macro_api'>PyMethod_GET_FUNCTION</span>(__pyx_t_13);
        <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_t_12);
        <span class='pyx_macro_api'>__Pyx_INCREF</span>(function);
        <span class='pyx_macro_api'>__Pyx_DECREF_SET</span>(__pyx_t_13, function);
      }
    }
    __pyx_t_1 = (__pyx_t_12) ? __Pyx_PyObject_Call2Args(__pyx_t_13, __pyx_t_12, __pyx_kp_s__2) : <span class='pyx_c_api'>__Pyx_PyObject_CallOneArg</span>(__pyx_t_13, __pyx_kp_s__2);
    <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_12); __pyx_t_12 = 0;
    if (unlikely(!__pyx_t_1)) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_13); __pyx_t_13 = 0;
    <span class='pyx_macro_api'>__Pyx_XDECREF_SET</span>(__pyx_v_src_sig, __pyx_t_1);
    __pyx_t_1 = 0;
    __pyx_t_15 = <span class='py_macro_api'>PyList_GET_SIZE</span>(__pyx_v_dest_sig);<span class='error_goto'> if (unlikely(__pyx_t_15 == ((Py_ssize_t)-1))) __PYX_ERR(0, 16, __pyx_L1_error)</span>
    __pyx_t_16 = __pyx_t_15;
    for (__pyx_t_17 = 0; __pyx_t_17 &lt; __pyx_t_16; __pyx_t_17+=1) {
      __pyx_v_i = __pyx_t_17;
      __pyx_t_1 = <span class='py_macro_api'>PyList_GET_ITEM</span>(__pyx_v_dest_sig, __pyx_v_i);
      <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_t_1);
      <span class='pyx_macro_api'>__Pyx_XDECREF_SET</span>(__pyx_v_dst_type, __pyx_t_1);
      __pyx_t_1 = 0;
      __pyx_t_3 = (__pyx_v_dst_type != Py_None);
      __pyx_t_2 = (__pyx_t_3 != 0);
      if (__pyx_t_2) {
        __pyx_t_1 = <span class='pyx_c_api'>__Pyx_GetItemInt</span>(__pyx_v_src_sig, __pyx_v_i, Py_ssize_t, 1, PyInt_FromSsize_t, 0, 0, 0);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
        <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
        __pyx_t_13 = <span class='py_c_api'>PyObject_RichCompare</span>(__pyx_t_1, __pyx_v_dst_type, Py_EQ); <span class='refnanny'>__Pyx_XGOTREF</span>(__pyx_t_13);<span class='error_goto'> if (unlikely(!__pyx_t_13)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
        <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
        __pyx_t_2 = <span class='pyx_c_api'>__Pyx_PyObject_IsTrue</span>(__pyx_t_13); if (unlikely(__pyx_t_2 &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
        <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_13); __pyx_t_13 = 0;
        if (__pyx_t_2) {
          __pyx_v_match_found = 1;
          goto __pyx_L43;
        }
        /*else*/ {
          __pyx_v_match_found = 0;
          goto __pyx_L41_break;
        }
        __pyx_L43:;
      }
    }
    __pyx_L41_break:;
    __pyx_t_2 = (__pyx_v_match_found != 0);
    if (__pyx_t_2) {
      __pyx_t_18 = <span class='pyx_c_api'>__Pyx_PyList_Append</span>(__pyx_v_candidates, __pyx_v_sig);<span class='error_goto'> if (unlikely(__pyx_t_18 == ((int)-1))) __PYX_ERR(0, 16, __pyx_L1_error)</span>
    }
  }
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_6); __pyx_t_6 = 0;
  __pyx_t_2 = (<span class='py_macro_api'>PyList_GET_SIZE</span>(__pyx_v_candidates) != 0);
  __pyx_t_3 = ((!__pyx_t_2) != 0);
  if (__pyx_t_3) {
    __pyx_t_6 = <span class='pyx_c_api'>__Pyx_PyObject_Call</span>(__pyx_builtin_TypeError, __pyx_tuple__3, NULL);<span class='error_goto'> if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_6);
    <span class='pyx_c_api'>__Pyx_Raise</span>(__pyx_t_6, 0, 0, 0);
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_6); __pyx_t_6 = 0;
    <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
  }
  __pyx_t_9 = <span class='py_macro_api'>PyList_GET_SIZE</span>(__pyx_v_candidates);<span class='error_goto'> if (unlikely(__pyx_t_9 == ((Py_ssize_t)-1))) __PYX_ERR(0, 16, __pyx_L1_error)</span>
  __pyx_t_3 = ((__pyx_t_9 &gt; 1) != 0);
  if (__pyx_t_3) {
/* … */
  __pyx_tuple__3 = <span class='py_c_api'>PyTuple_Pack</span>(1, __pyx_kp_s_No_matching_signature_found);<span class='error_goto'> if (unlikely(!__pyx_tuple__3)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_tuple__3);
  <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_tuple__3);
    __pyx_t_6 = <span class='pyx_c_api'>__Pyx_PyObject_Call</span>(__pyx_builtin_TypeError, __pyx_tuple__4, NULL);<span class='error_goto'> if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_6);
    <span class='pyx_c_api'>__Pyx_Raise</span>(__pyx_t_6, 0, 0, 0);
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_6); __pyx_t_6 = 0;
    <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
  }
  /*else*/ {
    <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_r);
    if (unlikely(__pyx_v_signatures == Py_None)) {
      <span class='py_c_api'>PyErr_SetString</span>(PyExc_TypeError, "'NoneType' object is not subscriptable");
      <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
    }
    __pyx_t_6 = <span class='pyx_c_api'>__Pyx_PyDict_GetItem</span>(((PyObject*)__pyx_v_signatures), <span class='py_macro_api'>PyList_GET_ITEM</span>(__pyx_v_candidates, 0));<span class='error_goto'> if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_6);
    __pyx_r = __pyx_t_6;
    __pyx_t_6 = 0;
    goto __pyx_L0;
  }

  /* function exit code */
  __pyx_L1_error:;
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_1);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_6);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_12);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_13);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_14);
  <span class='pyx_c_api'>__Pyx_AddTraceback</span>("_cython_magic_cd184590f55726778fe97a651b5f5dd0.__pyx_fused_cpdef", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_v_dest_sig);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_v_ndarray);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_v_arg);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_v_dtype);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_v_arg_base);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_v_candidates);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_v_sig);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_v_src_sig);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_v_dst_type);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_v_kwargs);
  <span class='refnanny'>__Pyx_XGIVEREF</span>(__pyx_r);
  <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
  return __pyx_r;
}

/* Python wrapper */
static PyObject *__pyx_fuse_0__pyx_pw_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_3compute_cy_mt(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyMethodDef __pyx_fuse_0__pyx_mdef_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_3compute_cy_mt = {"__pyx_fuse_0compute_cy_mt", (PyCFunction)(void*)(PyCFunctionWithKeywords)__pyx_fuse_0__pyx_pw_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_3compute_cy_mt, METH_VARARGS|METH_KEYWORDS, 0};
static PyObject *__pyx_fuse_0__pyx_pw_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_3compute_cy_mt(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  __Pyx_memviewslice __pyx_v_array_1 = { 0, 0, { 0 }, { 0 }, { 0 } };
  __Pyx_memviewslice __pyx_v_array_2 = { 0, 0, { 0 }, { 0 }, { 0 } };
  int __pyx_v_a;
  int __pyx_v_b;
  int __pyx_v_c;
  PyObject *__pyx_r = 0;
  <span class='refnanny'>__Pyx_RefNannyDeclarations</span>
  <span class='refnanny'>__Pyx_RefNannySetupContext</span>("compute_cy_mt (wrapper)", 0);
  {
    static PyObject **__pyx_pyargnames[] = {&amp;__pyx_n_s_array_1,&amp;__pyx_n_s_array_2,&amp;__pyx_n_s_a,&amp;__pyx_n_s_b,&amp;__pyx_n_s_c,0};
    PyObject* values[5] = {0,0,0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      const Py_ssize_t pos_args = <span class='py_macro_api'>PyTuple_GET_SIZE</span>(__pyx_args);
      switch (pos_args) {
        case  5: values[4] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 4);
        CYTHON_FALLTHROUGH;
        case  4: values[3] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 3);
        CYTHON_FALLTHROUGH;
        case  3: values[2] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 2);
        CYTHON_FALLTHROUGH;
        case  2: values[1] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 1);
        CYTHON_FALLTHROUGH;
        case  1: values[0] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 0);
        CYTHON_FALLTHROUGH;
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = <span class='py_c_api'>PyDict_Size</span>(__pyx_kwds);
      switch (pos_args) {
        case  0:
        if (likely((values[0] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_array_1)) != 0)) kw_args--;
        else goto __pyx_L5_argtuple_error;
        CYTHON_FALLTHROUGH;
        case  1:
        if (likely((values[1] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_array_2)) != 0)) kw_args--;
        else {
          <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("compute_cy_mt", 1, 5, 5, 1); <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
        }
        CYTHON_FALLTHROUGH;
        case  2:
        if (likely((values[2] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_a)) != 0)) kw_args--;
        else {
          <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("compute_cy_mt", 1, 5, 5, 2); <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
        }
        CYTHON_FALLTHROUGH;
        case  3:
        if (likely((values[3] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_b)) != 0)) kw_args--;
        else {
          <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("compute_cy_mt", 1, 5, 5, 3); <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
        }
        CYTHON_FALLTHROUGH;
        case  4:
        if (likely((values[4] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_c)) != 0)) kw_args--;
        else {
          <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("compute_cy_mt", 1, 5, 5, 4); <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
        }
      }
      if (unlikely(kw_args &gt; 0)) {
        if (unlikely(<span class='pyx_c_api'>__Pyx_ParseOptionalKeywords</span>(__pyx_kwds, __pyx_pyargnames, 0, values, pos_args, "compute_cy_mt") &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
      }
    } else if (<span class='py_macro_api'>PyTuple_GET_SIZE</span>(__pyx_args) != 5) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 0);
      values[1] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 1);
      values[2] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 2);
      values[3] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 3);
      values[4] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 4);
    }
    __pyx_v_array_1 = <span class='pyx_c_api'>__Pyx_PyObject_to_MemoryviewSlice_d_dc_int</span>(values[0], PyBUF_WRITABLE);<span class='error_goto'> if (unlikely(!__pyx_v_array_1.memview)) __PYX_ERR(0, 16, __pyx_L3_error)</span>
    __pyx_v_array_2 = <span class='pyx_c_api'>__Pyx_PyObject_to_MemoryviewSlice_d_dc_int</span>(values[1], PyBUF_WRITABLE);<span class='error_goto'> if (unlikely(!__pyx_v_array_2.memview)) __PYX_ERR(0, 16, __pyx_L3_error)</span>
    __pyx_v_a = <span class='pyx_c_api'>__Pyx_PyInt_As_int</span>(values[2]); if (unlikely((__pyx_v_a == (int)-1) &amp;&amp; <span class='py_c_api'>PyErr_Occurred</span>())) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
    __pyx_v_b = <span class='pyx_c_api'>__Pyx_PyInt_As_int</span>(values[3]); if (unlikely((__pyx_v_b == (int)-1) &amp;&amp; <span class='py_c_api'>PyErr_Occurred</span>())) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
    __pyx_v_c = <span class='pyx_c_api'>__Pyx_PyInt_As_int</span>(values[4]); if (unlikely((__pyx_v_c == (int)-1) &amp;&amp; <span class='py_c_api'>PyErr_Occurred</span>())) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("compute_cy_mt", 1, 5, 5, <span class='py_macro_api'>PyTuple_GET_SIZE</span>(__pyx_args)); <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
  __pyx_L3_error:;
  <span class='pyx_c_api'>__Pyx_AddTraceback</span>("_cython_magic_cd184590f55726778fe97a651b5f5dd0.compute_cy_mt", __pyx_clineno, __pyx_lineno, __pyx_filename);
  <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __pyx_r = __pyx_pf_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_2compute_cy_mt(__pyx_self, __pyx_v_array_1, __pyx_v_array_2, __pyx_v_a, __pyx_v_b, __pyx_v_c);
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;

  /* function exit code */
  <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
  return __pyx_r;
}

static PyObject *__pyx_pf_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_2compute_cy_mt(CYTHON_UNUSED PyObject *__pyx_self, __Pyx_memviewslice __pyx_v_array_1, __Pyx_memviewslice __pyx_v_array_2, int __pyx_v_a, int __pyx_v_b, int __pyx_v_c) {
  Py_ssize_t __pyx_v_x_max;
  Py_ssize_t __pyx_v_y_max;
  PyObject *__pyx_v_dtype = NULL;
  PyObject *__pyx_v_result = NULL;
  __Pyx_memviewslice __pyx_v_result_view = { 0, 0, { 0 }, { 0 }, { 0 } };
  int __pyx_v_tmp;
  Py_ssize_t __pyx_v_x;
  Py_ssize_t __pyx_v_y;
  PyObject *__pyx_r = NULL;
  <span class='refnanny'>__Pyx_RefNannyDeclarations</span>
  <span class='refnanny'>__Pyx_RefNannySetupContext</span>("__pyx_fuse_0compute_cy_mt", 0);
/* … */
  /* function exit code */
  __pyx_L1_error:;
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_1);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_2);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_3);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_5);
  __PYX_XDEC_MEMVIEW(&amp;__pyx_t_6, 1);
  <span class='pyx_c_api'>__Pyx_AddTraceback</span>("_cython_magic_cd184590f55726778fe97a651b5f5dd0.compute_cy_mt", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_v_dtype);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_v_result);
  __PYX_XDEC_MEMVIEW(&amp;__pyx_v_result_view, 1);
  __PYX_XDEC_MEMVIEW(&amp;__pyx_v_array_1, 1);
  __PYX_XDEC_MEMVIEW(&amp;__pyx_v_array_2, 1);
  <span class='refnanny'>__Pyx_XGIVEREF</span>(__pyx_r);
  <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
  return __pyx_r;
}

/* Python wrapper */
static PyObject *__pyx_fuse_1__pyx_pw_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_5compute_cy_mt(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyMethodDef __pyx_fuse_1__pyx_mdef_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_5compute_cy_mt = {"__pyx_fuse_1compute_cy_mt", (PyCFunction)(void*)(PyCFunctionWithKeywords)__pyx_fuse_1__pyx_pw_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_5compute_cy_mt, METH_VARARGS|METH_KEYWORDS, 0};
static PyObject *__pyx_fuse_1__pyx_pw_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_5compute_cy_mt(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  __Pyx_memviewslice __pyx_v_array_1 = { 0, 0, { 0 }, { 0 }, { 0 } };
  __Pyx_memviewslice __pyx_v_array_2 = { 0, 0, { 0 }, { 0 }, { 0 } };
  double __pyx_v_a;
  double __pyx_v_b;
  double __pyx_v_c;
  PyObject *__pyx_r = 0;
  <span class='refnanny'>__Pyx_RefNannyDeclarations</span>
  <span class='refnanny'>__Pyx_RefNannySetupContext</span>("compute_cy_mt (wrapper)", 0);
  {
    static PyObject **__pyx_pyargnames[] = {&amp;__pyx_n_s_array_1,&amp;__pyx_n_s_array_2,&amp;__pyx_n_s_a,&amp;__pyx_n_s_b,&amp;__pyx_n_s_c,0};
    PyObject* values[5] = {0,0,0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      const Py_ssize_t pos_args = <span class='py_macro_api'>PyTuple_GET_SIZE</span>(__pyx_args);
      switch (pos_args) {
        case  5: values[4] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 4);
        CYTHON_FALLTHROUGH;
        case  4: values[3] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 3);
        CYTHON_FALLTHROUGH;
        case  3: values[2] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 2);
        CYTHON_FALLTHROUGH;
        case  2: values[1] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 1);
        CYTHON_FALLTHROUGH;
        case  1: values[0] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 0);
        CYTHON_FALLTHROUGH;
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = <span class='py_c_api'>PyDict_Size</span>(__pyx_kwds);
      switch (pos_args) {
        case  0:
        if (likely((values[0] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_array_1)) != 0)) kw_args--;
        else goto __pyx_L5_argtuple_error;
        CYTHON_FALLTHROUGH;
        case  1:
        if (likely((values[1] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_array_2)) != 0)) kw_args--;
        else {
          <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("compute_cy_mt", 1, 5, 5, 1); <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
        }
        CYTHON_FALLTHROUGH;
        case  2:
        if (likely((values[2] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_a)) != 0)) kw_args--;
        else {
          <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("compute_cy_mt", 1, 5, 5, 2); <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
        }
        CYTHON_FALLTHROUGH;
        case  3:
        if (likely((values[3] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_b)) != 0)) kw_args--;
        else {
          <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("compute_cy_mt", 1, 5, 5, 3); <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
        }
        CYTHON_FALLTHROUGH;
        case  4:
        if (likely((values[4] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_c)) != 0)) kw_args--;
        else {
          <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("compute_cy_mt", 1, 5, 5, 4); <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
        }
      }
      if (unlikely(kw_args &gt; 0)) {
        if (unlikely(<span class='pyx_c_api'>__Pyx_ParseOptionalKeywords</span>(__pyx_kwds, __pyx_pyargnames, 0, values, pos_args, "compute_cy_mt") &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
      }
    } else if (<span class='py_macro_api'>PyTuple_GET_SIZE</span>(__pyx_args) != 5) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 0);
      values[1] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 1);
      values[2] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 2);
      values[3] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 3);
      values[4] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 4);
    }
    __pyx_v_array_1 = <span class='pyx_c_api'>__Pyx_PyObject_to_MemoryviewSlice_d_dc_double</span>(values[0], PyBUF_WRITABLE);<span class='error_goto'> if (unlikely(!__pyx_v_array_1.memview)) __PYX_ERR(0, 16, __pyx_L3_error)</span>
    __pyx_v_array_2 = <span class='pyx_c_api'>__Pyx_PyObject_to_MemoryviewSlice_d_dc_double</span>(values[1], PyBUF_WRITABLE);<span class='error_goto'> if (unlikely(!__pyx_v_array_2.memview)) __PYX_ERR(0, 16, __pyx_L3_error)</span>
    __pyx_v_a = __pyx_<span class='py_c_api'>PyFloat_AsDouble</span>(values[2]); if (unlikely((__pyx_v_a == (double)-1) &amp;&amp; <span class='py_c_api'>PyErr_Occurred</span>())) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
    __pyx_v_b = __pyx_<span class='py_c_api'>PyFloat_AsDouble</span>(values[3]); if (unlikely((__pyx_v_b == (double)-1) &amp;&amp; <span class='py_c_api'>PyErr_Occurred</span>())) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
    __pyx_v_c = __pyx_<span class='py_c_api'>PyFloat_AsDouble</span>(values[4]); if (unlikely((__pyx_v_c == (double)-1) &amp;&amp; <span class='py_c_api'>PyErr_Occurred</span>())) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("compute_cy_mt", 1, 5, 5, <span class='py_macro_api'>PyTuple_GET_SIZE</span>(__pyx_args)); <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
  __pyx_L3_error:;
  <span class='pyx_c_api'>__Pyx_AddTraceback</span>("_cython_magic_cd184590f55726778fe97a651b5f5dd0.compute_cy_mt", __pyx_clineno, __pyx_lineno, __pyx_filename);
  <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __pyx_r = __pyx_pf_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_4compute_cy_mt(__pyx_self, __pyx_v_array_1, __pyx_v_array_2, __pyx_v_a, __pyx_v_b, __pyx_v_c);
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;

  /* function exit code */
  <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
  return __pyx_r;
}

static PyObject *__pyx_pf_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_4compute_cy_mt(CYTHON_UNUSED PyObject *__pyx_self, __Pyx_memviewslice __pyx_v_array_1, __Pyx_memviewslice __pyx_v_array_2, double __pyx_v_a, double __pyx_v_b, double __pyx_v_c) {
  Py_ssize_t __pyx_v_x_max;
  Py_ssize_t __pyx_v_y_max;
  PyObject *__pyx_v_dtype = NULL;
  PyObject *__pyx_v_result = NULL;
  __Pyx_memviewslice __pyx_v_result_view = { 0, 0, { 0 }, { 0 }, { 0 } };
  double __pyx_v_tmp;
  Py_ssize_t __pyx_v_x;
  Py_ssize_t __pyx_v_y;
  PyObject *__pyx_r = NULL;
  <span class='refnanny'>__Pyx_RefNannyDeclarations</span>
  <span class='refnanny'>__Pyx_RefNannySetupContext</span>("__pyx_fuse_1compute_cy_mt", 0);
/* … */
  /* function exit code */
  __pyx_L1_error:;
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_1);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_2);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_3);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_5);
  __PYX_XDEC_MEMVIEW(&amp;__pyx_t_6, 1);
  <span class='pyx_c_api'>__Pyx_AddTraceback</span>("_cython_magic_cd184590f55726778fe97a651b5f5dd0.compute_cy_mt", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_v_dtype);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_v_result);
  __PYX_XDEC_MEMVIEW(&amp;__pyx_v_result_view, 1);
  __PYX_XDEC_MEMVIEW(&amp;__pyx_v_array_1, 1);
  __PYX_XDEC_MEMVIEW(&amp;__pyx_v_array_2, 1);
  <span class='refnanny'>__Pyx_XGIVEREF</span>(__pyx_r);
  <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
  return __pyx_r;
}

/* Python wrapper */
static PyObject *__pyx_fuse_2__pyx_pw_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_7compute_cy_mt(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyMethodDef __pyx_fuse_2__pyx_mdef_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_7compute_cy_mt = {"__pyx_fuse_2compute_cy_mt", (PyCFunction)(void*)(PyCFunctionWithKeywords)__pyx_fuse_2__pyx_pw_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_7compute_cy_mt, METH_VARARGS|METH_KEYWORDS, 0};
static PyObject *__pyx_fuse_2__pyx_pw_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_7compute_cy_mt(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  __Pyx_memviewslice __pyx_v_array_1 = { 0, 0, { 0 }, { 0 }, { 0 } };
  __Pyx_memviewslice __pyx_v_array_2 = { 0, 0, { 0 }, { 0 }, { 0 } };
  PY_LONG_LONG __pyx_v_a;
  PY_LONG_LONG __pyx_v_b;
  PY_LONG_LONG __pyx_v_c;
  PyObject *__pyx_r = 0;
  <span class='refnanny'>__Pyx_RefNannyDeclarations</span>
  <span class='refnanny'>__Pyx_RefNannySetupContext</span>("compute_cy_mt (wrapper)", 0);
  {
    static PyObject **__pyx_pyargnames[] = {&amp;__pyx_n_s_array_1,&amp;__pyx_n_s_array_2,&amp;__pyx_n_s_a,&amp;__pyx_n_s_b,&amp;__pyx_n_s_c,0};
    PyObject* values[5] = {0,0,0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      const Py_ssize_t pos_args = <span class='py_macro_api'>PyTuple_GET_SIZE</span>(__pyx_args);
      switch (pos_args) {
        case  5: values[4] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 4);
        CYTHON_FALLTHROUGH;
        case  4: values[3] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 3);
        CYTHON_FALLTHROUGH;
        case  3: values[2] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 2);
        CYTHON_FALLTHROUGH;
        case  2: values[1] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 1);
        CYTHON_FALLTHROUGH;
        case  1: values[0] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 0);
        CYTHON_FALLTHROUGH;
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = <span class='py_c_api'>PyDict_Size</span>(__pyx_kwds);
      switch (pos_args) {
        case  0:
        if (likely((values[0] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_array_1)) != 0)) kw_args--;
        else goto __pyx_L5_argtuple_error;
        CYTHON_FALLTHROUGH;
        case  1:
        if (likely((values[1] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_array_2)) != 0)) kw_args--;
        else {
          <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("compute_cy_mt", 1, 5, 5, 1); <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
        }
        CYTHON_FALLTHROUGH;
        case  2:
        if (likely((values[2] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_a)) != 0)) kw_args--;
        else {
          <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("compute_cy_mt", 1, 5, 5, 2); <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
        }
        CYTHON_FALLTHROUGH;
        case  3:
        if (likely((values[3] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_b)) != 0)) kw_args--;
        else {
          <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("compute_cy_mt", 1, 5, 5, 3); <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
        }
        CYTHON_FALLTHROUGH;
        case  4:
        if (likely((values[4] = <span class='pyx_c_api'>__Pyx_PyDict_GetItemStr</span>(__pyx_kwds, __pyx_n_s_c)) != 0)) kw_args--;
        else {
          <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("compute_cy_mt", 1, 5, 5, 4); <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
        }
      }
      if (unlikely(kw_args &gt; 0)) {
        if (unlikely(<span class='pyx_c_api'>__Pyx_ParseOptionalKeywords</span>(__pyx_kwds, __pyx_pyargnames, 0, values, pos_args, "compute_cy_mt") &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
      }
    } else if (<span class='py_macro_api'>PyTuple_GET_SIZE</span>(__pyx_args) != 5) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 0);
      values[1] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 1);
      values[2] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 2);
      values[3] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 3);
      values[4] = <span class='py_macro_api'>PyTuple_GET_ITEM</span>(__pyx_args, 4);
    }
    __pyx_v_array_1 = <span class='pyx_c_api'>__Pyx_PyObject_to_MemoryviewSlice_d_dc_PY_LONG_LONG</span>(values[0], PyBUF_WRITABLE);<span class='error_goto'> if (unlikely(!__pyx_v_array_1.memview)) __PYX_ERR(0, 16, __pyx_L3_error)</span>
    __pyx_v_array_2 = <span class='pyx_c_api'>__Pyx_PyObject_to_MemoryviewSlice_d_dc_PY_LONG_LONG</span>(values[1], PyBUF_WRITABLE);<span class='error_goto'> if (unlikely(!__pyx_v_array_2.memview)) __PYX_ERR(0, 16, __pyx_L3_error)</span>
    __pyx_v_a = <span class='pyx_c_api'>__Pyx_PyInt_As_PY_LONG_LONG</span>(values[2]); if (unlikely((__pyx_v_a == (PY_LONG_LONG)-1) &amp;&amp; <span class='py_c_api'>PyErr_Occurred</span>())) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
    __pyx_v_b = <span class='pyx_c_api'>__Pyx_PyInt_As_PY_LONG_LONG</span>(values[3]); if (unlikely((__pyx_v_b == (PY_LONG_LONG)-1) &amp;&amp; <span class='py_c_api'>PyErr_Occurred</span>())) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
    __pyx_v_c = <span class='pyx_c_api'>__Pyx_PyInt_As_PY_LONG_LONG</span>(values[4]); if (unlikely((__pyx_v_c == (PY_LONG_LONG)-1) &amp;&amp; <span class='py_c_api'>PyErr_Occurred</span>())) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  <span class='pyx_c_api'>__Pyx_RaiseArgtupleInvalid</span>("compute_cy_mt", 1, 5, 5, <span class='py_macro_api'>PyTuple_GET_SIZE</span>(__pyx_args)); <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L3_error)</span>
  __pyx_L3_error:;
  <span class='pyx_c_api'>__Pyx_AddTraceback</span>("_cython_magic_cd184590f55726778fe97a651b5f5dd0.compute_cy_mt", __pyx_clineno, __pyx_lineno, __pyx_filename);
  <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __pyx_r = __pyx_pf_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_6compute_cy_mt(__pyx_self, __pyx_v_array_1, __pyx_v_array_2, __pyx_v_a, __pyx_v_b, __pyx_v_c);
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;

  /* function exit code */
  <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
  return __pyx_r;
}

static PyObject *__pyx_pf_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_6compute_cy_mt(CYTHON_UNUSED PyObject *__pyx_self, __Pyx_memviewslice __pyx_v_array_1, __Pyx_memviewslice __pyx_v_array_2, PY_LONG_LONG __pyx_v_a, PY_LONG_LONG __pyx_v_b, PY_LONG_LONG __pyx_v_c) {
  Py_ssize_t __pyx_v_x_max;
  Py_ssize_t __pyx_v_y_max;
  PyObject *__pyx_v_dtype = NULL;
  PyObject *__pyx_v_result = NULL;
  __Pyx_memviewslice __pyx_v_result_view = { 0, 0, { 0 }, { 0 }, { 0 } };
  PY_LONG_LONG __pyx_v_tmp;
  Py_ssize_t __pyx_v_x;
  Py_ssize_t __pyx_v_y;
  PyObject *__pyx_r = NULL;
  <span class='refnanny'>__Pyx_RefNannyDeclarations</span>
  <span class='refnanny'>__Pyx_RefNannySetupContext</span>("__pyx_fuse_2compute_cy_mt", 0);
/* … */
  /* function exit code */
  __pyx_L1_error:;
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_1);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_2);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_3);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_5);
  __PYX_XDEC_MEMVIEW(&amp;__pyx_t_6, 1);
  <span class='pyx_c_api'>__Pyx_AddTraceback</span>("_cython_magic_cd184590f55726778fe97a651b5f5dd0.compute_cy_mt", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_v_dtype);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_v_result);
  __PYX_XDEC_MEMVIEW(&amp;__pyx_v_result_view, 1);
  __PYX_XDEC_MEMVIEW(&amp;__pyx_v_array_1, 1);
  __PYX_XDEC_MEMVIEW(&amp;__pyx_v_array_2, 1);
  <span class='refnanny'>__Pyx_XGIVEREF</span>(__pyx_r);
  <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
  return __pyx_r;
}
  __pyx_tuple__4 = <span class='py_c_api'>PyTuple_Pack</span>(1, __pyx_kp_s_Function_call_with_ambiguous_arg);<span class='error_goto'> if (unlikely(!__pyx_tuple__4)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_tuple__4);
  <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_tuple__4);
/* … */
  __pyx_tuple__24 = <span class='py_c_api'>PyTuple_Pack</span>(13, __pyx_n_s_array_1, __pyx_n_s_array_2, __pyx_n_s_a, __pyx_n_s_b, __pyx_n_s_c, __pyx_n_s_x_max, __pyx_n_s_y_max, __pyx_n_s_dtype, __pyx_n_s_result, __pyx_n_s_result_view, __pyx_n_s_tmp, __pyx_n_s_x, __pyx_n_s_y);<span class='error_goto'> if (unlikely(!__pyx_tuple__24)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_tuple__24);
  <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_tuple__24);
/* … */
  __pyx_t_1 = <span class='pyx_c_api'>__Pyx_PyDict_NewPresized</span>(3);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  __pyx_t_2 = __pyx_FusedFunction_New(&amp;__pyx_fuse_0__pyx_mdef_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_3compute_cy_mt, 0, __pyx_n_s_compute_cy_mt, NULL, __pyx_n_s_cython_magic_cd184590f55726778f, __pyx_d, ((PyObject *)__pyx_codeobj__25));<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
  <span class='pyx_c_api'>__Pyx_CyFunction_SetDefaultsTuple</span>(__pyx_t_2, __pyx_empty_tuple);
  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_t_1, __pyx_n_s_int, __pyx_t_2) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
  __pyx_t_2 = __pyx_FusedFunction_New(&amp;__pyx_fuse_1__pyx_mdef_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_5compute_cy_mt, 0, __pyx_n_s_compute_cy_mt, NULL, __pyx_n_s_cython_magic_cd184590f55726778f, __pyx_d, ((PyObject *)__pyx_codeobj__25));<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
  <span class='pyx_c_api'>__Pyx_CyFunction_SetDefaultsTuple</span>(__pyx_t_2, __pyx_empty_tuple);
  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_t_1, __pyx_n_s_double, __pyx_t_2) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
  __pyx_t_2 = __pyx_FusedFunction_New(&amp;__pyx_fuse_2__pyx_mdef_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_7compute_cy_mt, 0, __pyx_n_s_compute_cy_mt, NULL, __pyx_n_s_cython_magic_cd184590f55726778f, __pyx_d, ((PyObject *)__pyx_codeobj__25));<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
  <span class='pyx_c_api'>__Pyx_CyFunction_SetDefaultsTuple</span>(__pyx_t_2, __pyx_empty_tuple);
  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_t_1, __pyx_kp_s_long_long, __pyx_t_2) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
  __pyx_t_2 = __pyx_FusedFunction_New(&amp;__pyx_mdef_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_1compute_cy_mt, 0, __pyx_n_s_compute_cy_mt, NULL, __pyx_n_s_cython_magic_cd184590f55726778f, __pyx_d, ((PyObject *)__pyx_codeobj__25));<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
  <span class='pyx_c_api'>__Pyx_CyFunction_SetDefaultsTuple</span>(__pyx_t_2, __pyx_empty_tuple);
  ((__pyx_FusedFunctionObject *) __pyx_t_2)-&gt;__signatures__ = __pyx_t_1;
  <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_t_1);
  __pyx_t_1 = 0;
  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_compute_cy_mt, __pyx_t_2) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 16, __pyx_L1_error)</span>
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
  __pyx_codeobj__25 = (PyObject*)<span class='pyx_c_api'>__Pyx_PyCode_New</span>(5, 0, 13, 0, CO_OPTIMIZED|CO_NEWLOCALS, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple__24, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_C_Users_DELL_ipython_cython__cyt, __pyx_n_s_compute_cy_mt, 16, __pyx_empty_bytes);<span class='error_goto'> if (unlikely(!__pyx_codeobj__25)) __PYX_ERR(0, 16, __pyx_L1_error)</span>
</pre><pre class="cython line score-0">&#xA0;<span class="">17</span>: </pre>
<pre class="cython line score-0" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">18</span>:     <span class="k">cdef</span> <span class="kt">Py_ssize_t</span> <span class="nf">x_max</span> <span class="o">=</span> <span class="n">array_1</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mf">0</span><span class="p">]</span></pre>
<pre class='cython code score-0 '>  __pyx_v_x_max = (__pyx_v_array_1.shape[0]);
/* … */
  __pyx_v_x_max = (__pyx_v_array_1.shape[0]);
/* … */
  __pyx_v_x_max = (__pyx_v_array_1.shape[0]);
</pre><pre class="cython line score-0" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">19</span>:     <span class="k">cdef</span> <span class="kt">Py_ssize_t</span> <span class="nf">y_max</span> <span class="o">=</span> <span class="n">array_1</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mf">1</span><span class="p">]</span></pre>
<pre class='cython code score-0 '>  __pyx_v_y_max = (__pyx_v_array_1.shape[1]);
/* … */
  __pyx_v_y_max = (__pyx_v_array_1.shape[1]);
/* … */
  __pyx_v_y_max = (__pyx_v_array_1.shape[1]);
</pre><pre class="cython line score-0">&#xA0;<span class="">20</span>: </pre>
<pre class="cython line score-63" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">21</span>:     <span class="k">assert</span> <span class="nb">tuple</span><span class="p">(</span><span class="n">array_1</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span> <span class="o">==</span> <span class="nb">tuple</span><span class="p">(</span><span class="n">array_2</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span></pre>
<pre class='cython code score-63 '>  #ifndef CYTHON_WITHOUT_ASSERTIONS
  if (unlikely(!Py_OptimizeFlag)) {
    __pyx_t_1 = __Pyx_carray_to_py_Py_ssize_t(__pyx_v_array_1.shape, 8);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 21, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
    __pyx_t_2 = <span class='pyx_c_api'>__Pyx_PySequence_Tuple</span>(__pyx_t_1);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 21, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_t_1 = __Pyx_carray_to_py_Py_ssize_t(__pyx_v_array_2.shape, 8);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 21, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
    __pyx_t_3 = <span class='pyx_c_api'>__Pyx_PySequence_Tuple</span>(__pyx_t_1);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 21, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_t_1 = <span class='py_c_api'>PyObject_RichCompare</span>(__pyx_t_2, __pyx_t_3, Py_EQ); <span class='refnanny'>__Pyx_XGOTREF</span>(__pyx_t_1);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 21, __pyx_L1_error)</span>
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
    __pyx_t_4 = <span class='pyx_c_api'>__Pyx_PyObject_IsTrue</span>(__pyx_t_1); if (unlikely(__pyx_t_4 &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 21, __pyx_L1_error)</span>
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
    if (unlikely(!__pyx_t_4)) {
      <span class='py_c_api'>PyErr_SetNone</span>(PyExc_AssertionError);
      <span class='error_goto'>__PYX_ERR(0, 21, __pyx_L1_error)</span>
    }
  }
  #endif
/* … */
  #ifndef CYTHON_WITHOUT_ASSERTIONS
  if (unlikely(!Py_OptimizeFlag)) {
    __pyx_t_1 = __Pyx_carray_to_py_Py_ssize_t(__pyx_v_array_1.shape, 8);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 21, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
    __pyx_t_2 = <span class='pyx_c_api'>__Pyx_PySequence_Tuple</span>(__pyx_t_1);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 21, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_t_1 = __Pyx_carray_to_py_Py_ssize_t(__pyx_v_array_2.shape, 8);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 21, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
    __pyx_t_3 = <span class='pyx_c_api'>__Pyx_PySequence_Tuple</span>(__pyx_t_1);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 21, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_t_1 = <span class='py_c_api'>PyObject_RichCompare</span>(__pyx_t_2, __pyx_t_3, Py_EQ); <span class='refnanny'>__Pyx_XGOTREF</span>(__pyx_t_1);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 21, __pyx_L1_error)</span>
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
    __pyx_t_4 = <span class='pyx_c_api'>__Pyx_PyObject_IsTrue</span>(__pyx_t_1); if (unlikely(__pyx_t_4 &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 21, __pyx_L1_error)</span>
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
    if (unlikely(!__pyx_t_4)) {
      <span class='py_c_api'>PyErr_SetNone</span>(PyExc_AssertionError);
      <span class='error_goto'>__PYX_ERR(0, 21, __pyx_L1_error)</span>
    }
  }
  #endif
/* … */
  #ifndef CYTHON_WITHOUT_ASSERTIONS
  if (unlikely(!Py_OptimizeFlag)) {
    __pyx_t_1 = __Pyx_carray_to_py_Py_ssize_t(__pyx_v_array_1.shape, 8);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 21, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
    __pyx_t_2 = <span class='pyx_c_api'>__Pyx_PySequence_Tuple</span>(__pyx_t_1);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 21, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_t_1 = __Pyx_carray_to_py_Py_ssize_t(__pyx_v_array_2.shape, 8);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 21, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
    __pyx_t_3 = <span class='pyx_c_api'>__Pyx_PySequence_Tuple</span>(__pyx_t_1);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 21, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_t_1 = <span class='py_c_api'>PyObject_RichCompare</span>(__pyx_t_2, __pyx_t_3, Py_EQ); <span class='refnanny'>__Pyx_XGOTREF</span>(__pyx_t_1);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 21, __pyx_L1_error)</span>
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
    __pyx_t_4 = <span class='pyx_c_api'>__Pyx_PyObject_IsTrue</span>(__pyx_t_1); if (unlikely(__pyx_t_4 &lt; 0)) <span class='error_goto'>__PYX_ERR(0, 21, __pyx_L1_error)</span>
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
    if (unlikely(!__pyx_t_4)) {
      <span class='py_c_api'>PyErr_SetNone</span>(PyExc_AssertionError);
      <span class='error_goto'>__PYX_ERR(0, 21, __pyx_L1_error)</span>
    }
  }
  #endif
</pre><pre class="cython line score-0">&#xA0;<span class="">22</span>: </pre>
<pre class="cython line score-0">&#xA0;<span class="">23</span>:     <span class="k">if</span> <span class="n">my_type</span> <span class="ow">is</span> <span class="nb">int</span><span class="p">:</span></pre>
<pre class="cython line score-5" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">24</span>:         <span class="n">dtype</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">intc</span></pre>
<pre class='cython code score-5 '>  <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_t_1, __pyx_n_s_np);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 24, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  __pyx_t_3 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_t_1, __pyx_n_s_intc);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 24, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_v_dtype = __pyx_t_3;
  __pyx_t_3 = 0;
</pre><pre class="cython line score-0">&#xA0;<span class="">25</span>:     <span class="k">elif</span> <span class="n">my_type</span> <span class="ow">is</span> <span class="n">double</span><span class="p">:</span></pre>
<pre class="cython line score-5" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">26</span>:         <span class="n">dtype</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">double</span></pre>
<pre class='cython code score-5 '>  <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_t_1, __pyx_n_s_np);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 26, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  __pyx_t_3 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_t_1, __pyx_n_s_double);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 26, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_v_dtype = __pyx_t_3;
  __pyx_t_3 = 0;
</pre><pre class="cython line score-0">&#xA0;<span class="">27</span>:     <span class="k">elif</span> <span class="n">my_type</span> <span class="ow">is</span> <span class="n">cython</span><span class="o">.</span><span class="n">longlong</span><span class="p">:</span></pre>
<pre class="cython line score-5" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">28</span>:         <span class="n">dtype</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">longlong</span></pre>
<pre class='cython code score-5 '>  <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_t_1, __pyx_n_s_np);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 28, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  __pyx_t_3 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_t_1, __pyx_n_s_longlong);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 28, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_v_dtype = __pyx_t_3;
  __pyx_t_3 = 0;
</pre><pre class="cython line score-0">&#xA0;<span class="">29</span>: </pre>
<pre class="cython line score-120" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">30</span>:     <span class="n">result</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">((</span><span class="n">x_max</span><span class="p">,</span> <span class="n">y_max</span><span class="p">),</span> <span class="n">dtype</span><span class="o">=</span><span class="n">dtype</span><span class="p">)</span></pre>
<pre class='cython code score-120 '>  <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_t_3, __pyx_n_s_np);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  __pyx_t_1 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_t_3, __pyx_n_s_zeros);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
  __pyx_t_3 = <span class='py_c_api'>PyInt_FromSsize_t</span>(__pyx_v_x_max);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  __pyx_t_2 = <span class='py_c_api'>PyInt_FromSsize_t</span>(__pyx_v_y_max);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
  __pyx_t_5 = <span class='py_c_api'>PyTuple_New</span>(2);<span class='error_goto'> if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
  <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_t_3);
  <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_5, 0, __pyx_t_3);
  <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_t_2);
  <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_5, 1, __pyx_t_2);
  __pyx_t_3 = 0;
  __pyx_t_2 = 0;
  __pyx_t_2 = <span class='py_c_api'>PyTuple_New</span>(1);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
  <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_t_5);
  <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_2, 0, __pyx_t_5);
  __pyx_t_5 = 0;
  __pyx_t_5 = <span class='pyx_c_api'>__Pyx_PyDict_NewPresized</span>(1);<span class='error_goto'> if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_t_5, __pyx_n_s_dtype, __pyx_v_dtype) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 30, __pyx_L1_error)</span>
  __pyx_t_3 = <span class='pyx_c_api'>__Pyx_PyObject_Call</span>(__pyx_t_1, __pyx_t_2, __pyx_t_5);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_5); __pyx_t_5 = 0;
  __pyx_v_result = __pyx_t_3;
  __pyx_t_3 = 0;
/* … */
  <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_t_3, __pyx_n_s_np);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  __pyx_t_1 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_t_3, __pyx_n_s_zeros);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
  __pyx_t_3 = <span class='py_c_api'>PyInt_FromSsize_t</span>(__pyx_v_x_max);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  __pyx_t_2 = <span class='py_c_api'>PyInt_FromSsize_t</span>(__pyx_v_y_max);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
  __pyx_t_5 = <span class='py_c_api'>PyTuple_New</span>(2);<span class='error_goto'> if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
  <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_t_3);
  <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_5, 0, __pyx_t_3);
  <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_t_2);
  <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_5, 1, __pyx_t_2);
  __pyx_t_3 = 0;
  __pyx_t_2 = 0;
  __pyx_t_2 = <span class='py_c_api'>PyTuple_New</span>(1);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
  <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_t_5);
  <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_2, 0, __pyx_t_5);
  __pyx_t_5 = 0;
  __pyx_t_5 = <span class='pyx_c_api'>__Pyx_PyDict_NewPresized</span>(1);<span class='error_goto'> if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_t_5, __pyx_n_s_dtype, __pyx_v_dtype) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 30, __pyx_L1_error)</span>
  __pyx_t_3 = <span class='pyx_c_api'>__Pyx_PyObject_Call</span>(__pyx_t_1, __pyx_t_2, __pyx_t_5);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_5); __pyx_t_5 = 0;
  __pyx_v_result = __pyx_t_3;
  __pyx_t_3 = 0;
/* … */
  <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_t_3, __pyx_n_s_np);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  __pyx_t_1 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_t_3, __pyx_n_s_zeros);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
  __pyx_t_3 = <span class='py_c_api'>PyInt_FromSsize_t</span>(__pyx_v_x_max);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  __pyx_t_2 = <span class='py_c_api'>PyInt_FromSsize_t</span>(__pyx_v_y_max);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
  __pyx_t_5 = <span class='py_c_api'>PyTuple_New</span>(2);<span class='error_goto'> if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
  <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_t_3);
  <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_5, 0, __pyx_t_3);
  <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_t_2);
  <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_5, 1, __pyx_t_2);
  __pyx_t_3 = 0;
  __pyx_t_2 = 0;
  __pyx_t_2 = <span class='py_c_api'>PyTuple_New</span>(1);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
  <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_t_5);
  <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_2, 0, __pyx_t_5);
  __pyx_t_5 = 0;
  __pyx_t_5 = <span class='pyx_c_api'>__Pyx_PyDict_NewPresized</span>(1);<span class='error_goto'> if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_t_5, __pyx_n_s_dtype, __pyx_v_dtype) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 30, __pyx_L1_error)</span>
  __pyx_t_3 = <span class='pyx_c_api'>__Pyx_PyObject_Call</span>(__pyx_t_1, __pyx_t_2, __pyx_t_5);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 30, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_5); __pyx_t_5 = 0;
  __pyx_v_result = __pyx_t_3;
  __pyx_t_3 = 0;
</pre><pre class="cython line score-6" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">31</span>:     <span class="k">cdef</span> <span class="kt">my_type</span>[<span class="p">:,</span> <span class="p">::</span><span class="mf">1</span><span class="p">]</span> <span class="n">result_view</span> <span class="o">=</span> <span class="n">result</span></pre>
<pre class='cython code score-6 '>  __pyx_t_6 = <span class='pyx_c_api'>__Pyx_PyObject_to_MemoryviewSlice_d_dc_int</span>(__pyx_v_result, PyBUF_WRITABLE);<span class='error_goto'> if (unlikely(!__pyx_t_6.memview)) __PYX_ERR(0, 31, __pyx_L1_error)</span>
  __pyx_v_result_view = __pyx_t_6;
  __pyx_t_6.memview = NULL;
  __pyx_t_6.data = NULL;
/* … */
  __pyx_t_6 = <span class='pyx_c_api'>__Pyx_PyObject_to_MemoryviewSlice_d_dc_double</span>(__pyx_v_result, PyBUF_WRITABLE);<span class='error_goto'> if (unlikely(!__pyx_t_6.memview)) __PYX_ERR(0, 31, __pyx_L1_error)</span>
  __pyx_v_result_view = __pyx_t_6;
  __pyx_t_6.memview = NULL;
  __pyx_t_6.data = NULL;
/* … */
  __pyx_t_6 = <span class='pyx_c_api'>__Pyx_PyObject_to_MemoryviewSlice_d_dc_PY_LONG_LONG</span>(__pyx_v_result, PyBUF_WRITABLE);<span class='error_goto'> if (unlikely(!__pyx_t_6.memview)) __PYX_ERR(0, 31, __pyx_L1_error)</span>
  __pyx_v_result_view = __pyx_t_6;
  __pyx_t_6.memview = NULL;
  __pyx_t_6.data = NULL;
</pre><pre class="cython line score-0">&#xA0;<span class="">32</span>: </pre>
<pre class="cython line score-0">&#xA0;<span class="">33</span>:     <span class="k">cdef</span> <span class="kt">my_type</span> <span class="nf">tmp</span></pre>
<pre class="cython line score-0">&#xA0;<span class="">34</span>:     <span class="k">cdef</span> <span class="kt">Py_ssize_t</span> <span class="nf">x</span><span class="p">,</span> <span class="nf">y</span></pre>
<pre class="cython line score-0">&#xA0;<span class="">35</span>: </pre>
<pre class="cython line score-0" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">36</span>:     <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">x_max</span><span class="p">):</span></pre>
<pre class='cython code score-0 '>  __pyx_t_7 = __pyx_v_x_max;
  __pyx_t_8 = __pyx_t_7;
  for (__pyx_t_9 = 0; __pyx_t_9 &lt; __pyx_t_8; __pyx_t_9+=1) {
    __pyx_v_x = __pyx_t_9;
/* … */
  __pyx_t_7 = __pyx_v_x_max;
  __pyx_t_8 = __pyx_t_7;
  for (__pyx_t_9 = 0; __pyx_t_9 &lt; __pyx_t_8; __pyx_t_9+=1) {
    __pyx_v_x = __pyx_t_9;
/* … */
  __pyx_t_7 = __pyx_v_x_max;
  __pyx_t_8 = __pyx_t_7;
  for (__pyx_t_9 = 0; __pyx_t_9 &lt; __pyx_t_8; __pyx_t_9+=1) {
    __pyx_v_x = __pyx_t_9;
</pre><pre class="cython line score-12" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">37</span>:         <span class="k">for</span> <span class="n">y</span> <span class="ow">in</span> <span class="n">prange</span><span class="p">(</span><span class="n">y_max</span><span class="p">,</span> <span class="n">num_threads</span><span class="o">=</span><span class="mf">4</span><span class="p">,</span><span class="k">nogil</span><span class="o">=</span><span class="bp">True</span><span class="p">):</span></pre>
<pre class='cython code score-12 '>    {
        #ifdef WITH_THREAD
        PyThreadState *_save;
        Py_UNBLOCK_THREADS
        <span class='pyx_c_api'>__Pyx_FastGIL_Remember</span>();
        #endif
        /*try:*/ {
          __pyx_t_10 = __pyx_v_y_max;
          if ((1 == 0)) abort();
          {
              #if ((defined(__APPLE__) || defined(__OSX__)) &amp;&amp; (defined(__GNUC__) &amp;&amp; (__GNUC__ &gt; 2 || (__GNUC__ == 2 &amp;&amp; (__GNUC_MINOR__ &gt; 95)))))
                  #undef likely
                  #undef unlikely
                  #define likely(x)   (x)
                  #define unlikely(x) (x)
              #endif
              __pyx_t_12 = (__pyx_t_10 - 0 + 1 - 1/abs(1)) / 1;
              if (__pyx_t_12 &gt; 0)
              {
                  #ifdef _OPENMP
                  #pragma omp parallel
                  #endif /* _OPENMP */
                  {
                      #ifdef _OPENMP
                      #pragma omp for lastprivate(__pyx_v_tmp) firstprivate(__pyx_v_y) lastprivate(__pyx_v_y) num_threads(4)
                      #endif /* _OPENMP */
                      for (__pyx_t_11 = 0; __pyx_t_11 &lt; __pyx_t_12; __pyx_t_11++){
                          {
                              __pyx_v_y = (Py_ssize_t)(0 + 1 * __pyx_t_11);
                              /* Initialize private variables to invalid values */
                              __pyx_v_tmp = ((int)0xbad0bad0);
/* … */
        /*finally:*/ {
          /*normal exit:*/{
            #ifdef WITH_THREAD
            <span class='pyx_c_api'>__Pyx_FastGIL_Forget</span>();
            Py_BLOCK_THREADS
            #endif
            goto __pyx_L9;
          }
          __pyx_L9:;
        }
    }
  }
/* … */
    {
        #ifdef WITH_THREAD
        PyThreadState *_save;
        Py_UNBLOCK_THREADS
        <span class='pyx_c_api'>__Pyx_FastGIL_Remember</span>();
        #endif
        /*try:*/ {
          __pyx_t_10 = __pyx_v_y_max;
          if ((1 == 0)) abort();
          {
              #if ((defined(__APPLE__) || defined(__OSX__)) &amp;&amp; (defined(__GNUC__) &amp;&amp; (__GNUC__ &gt; 2 || (__GNUC__ == 2 &amp;&amp; (__GNUC_MINOR__ &gt; 95)))))
                  #undef likely
                  #undef unlikely
                  #define likely(x)   (x)
                  #define unlikely(x) (x)
              #endif
              __pyx_t_12 = (__pyx_t_10 - 0 + 1 - 1/abs(1)) / 1;
              if (__pyx_t_12 &gt; 0)
              {
                  #ifdef _OPENMP
                  #pragma omp parallel
                  #endif /* _OPENMP */
                  {
                      #ifdef _OPENMP
                      #pragma omp for lastprivate(__pyx_v_tmp) firstprivate(__pyx_v_y) lastprivate(__pyx_v_y) num_threads(4)
                      #endif /* _OPENMP */
                      for (__pyx_t_11 = 0; __pyx_t_11 &lt; __pyx_t_12; __pyx_t_11++){
                          {
                              __pyx_v_y = (Py_ssize_t)(0 + 1 * __pyx_t_11);
                              /* Initialize private variables to invalid values */
                              __pyx_v_tmp = ((double)__PYX_NAN());
/* … */
        /*finally:*/ {
          /*normal exit:*/{
            #ifdef WITH_THREAD
            <span class='pyx_c_api'>__Pyx_FastGIL_Forget</span>();
            Py_BLOCK_THREADS
            #endif
            goto __pyx_L9;
          }
          __pyx_L9:;
        }
    }
  }
/* … */
    {
        #ifdef WITH_THREAD
        PyThreadState *_save;
        Py_UNBLOCK_THREADS
        <span class='pyx_c_api'>__Pyx_FastGIL_Remember</span>();
        #endif
        /*try:*/ {
          __pyx_t_10 = __pyx_v_y_max;
          if ((1 == 0)) abort();
          {
              #if ((defined(__APPLE__) || defined(__OSX__)) &amp;&amp; (defined(__GNUC__) &amp;&amp; (__GNUC__ &gt; 2 || (__GNUC__ == 2 &amp;&amp; (__GNUC_MINOR__ &gt; 95)))))
                  #undef likely
                  #undef unlikely
                  #define likely(x)   (x)
                  #define unlikely(x) (x)
              #endif
              __pyx_t_12 = (__pyx_t_10 - 0 + 1 - 1/abs(1)) / 1;
              if (__pyx_t_12 &gt; 0)
              {
                  #ifdef _OPENMP
                  #pragma omp parallel
                  #endif /* _OPENMP */
                  {
                      #ifdef _OPENMP
                      #pragma omp for lastprivate(__pyx_v_tmp) firstprivate(__pyx_v_y) lastprivate(__pyx_v_y) num_threads(4)
                      #endif /* _OPENMP */
                      for (__pyx_t_11 = 0; __pyx_t_11 &lt; __pyx_t_12; __pyx_t_11++){
                          {
                              __pyx_v_y = (Py_ssize_t)(0 + 1 * __pyx_t_11);
                              /* Initialize private variables to invalid values */
                              __pyx_v_tmp = ((PY_LONG_LONG)0xbad0bad0);
/* … */
        /*finally:*/ {
          /*normal exit:*/{
            #ifdef WITH_THREAD
            <span class='pyx_c_api'>__Pyx_FastGIL_Forget</span>();
            Py_BLOCK_THREADS
            #endif
            goto __pyx_L9;
          }
          __pyx_L9:;
        }
    }
  }
</pre><pre class="cython line score-0" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">38</span>:             <span class="n">tmp</span> <span class="o">=</span> <span class="n">clip</span><span class="p">(</span><span class="n">array_1</span><span class="p">[</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">],</span> <span class="mf">2</span><span class="p">,</span> <span class="mf">10</span><span class="p">)</span></pre>
<pre class='cython code score-0 '>                              __pyx_t_13 = __pyx_v_x;
                              __pyx_t_14 = __pyx_v_y;
                              __pyx_v_tmp = __pyx_fuse_0__pyx_f_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_clip((*((int *) ( /* dim=1 */ ((char *) (((int *) ( /* dim=0 */ (__pyx_v_array_1.data + __pyx_t_13 * __pyx_v_array_1.strides[0]) )) + __pyx_t_14)) ))), 2, 10);
/* … */
                              __pyx_t_13 = __pyx_v_x;
                              __pyx_t_14 = __pyx_v_y;
                              __pyx_v_tmp = __pyx_fuse_1__pyx_f_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_clip((*((double *) ( /* dim=1 */ ((char *) (((double *) ( /* dim=0 */ (__pyx_v_array_1.data + __pyx_t_13 * __pyx_v_array_1.strides[0]) )) + __pyx_t_14)) ))), 2.0, 10.0);
/* … */
                              __pyx_t_13 = __pyx_v_x;
                              __pyx_t_14 = __pyx_v_y;
                              __pyx_v_tmp = __pyx_fuse_2__pyx_f_46_cython_magic_cd184590f55726778fe97a651b5f5dd0_clip((*((PY_LONG_LONG *) ( /* dim=1 */ ((char *) (((PY_LONG_LONG *) ( /* dim=0 */ (__pyx_v_array_1.data + __pyx_t_13 * __pyx_v_array_1.strides[0]) )) + __pyx_t_14)) ))), 2, 10);
</pre><pre class="cython line score-0" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">39</span>:             <span class="n">tmp</span> <span class="o">=</span> <span class="n">tmp</span> <span class="o">*</span> <span class="n">a</span> <span class="o">+</span> <span class="n">array_2</span><span class="p">[</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">]</span> <span class="o">*</span> <span class="n">b</span></pre>
<pre class='cython code score-0 '>                              __pyx_t_14 = __pyx_v_x;
                              __pyx_t_13 = __pyx_v_y;
                              __pyx_v_tmp = ((__pyx_v_tmp * __pyx_v_a) + ((*((int *) ( /* dim=1 */ ((char *) (((int *) ( /* dim=0 */ (__pyx_v_array_2.data + __pyx_t_14 * __pyx_v_array_2.strides[0]) )) + __pyx_t_13)) ))) * __pyx_v_b));
/* … */
                              __pyx_t_14 = __pyx_v_x;
                              __pyx_t_13 = __pyx_v_y;
                              __pyx_v_tmp = ((__pyx_v_tmp * __pyx_v_a) + ((*((double *) ( /* dim=1 */ ((char *) (((double *) ( /* dim=0 */ (__pyx_v_array_2.data + __pyx_t_14 * __pyx_v_array_2.strides[0]) )) + __pyx_t_13)) ))) * __pyx_v_b));
/* … */
                              __pyx_t_14 = __pyx_v_x;
                              __pyx_t_13 = __pyx_v_y;
                              __pyx_v_tmp = ((__pyx_v_tmp * __pyx_v_a) + ((*((PY_LONG_LONG *) ( /* dim=1 */ ((char *) (((PY_LONG_LONG *) ( /* dim=0 */ (__pyx_v_array_2.data + __pyx_t_14 * __pyx_v_array_2.strides[0]) )) + __pyx_t_13)) ))) * __pyx_v_b));
</pre><pre class="cython line score-0" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">40</span>:             <span class="n">result_view</span><span class="p">[</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">]</span> <span class="o">=</span> <span class="n">tmp</span> <span class="o">+</span> <span class="n">c</span></pre>
<pre class='cython code score-0 '>                              __pyx_t_13 = __pyx_v_x;
                              __pyx_t_14 = __pyx_v_y;
                              *((int *) ( /* dim=1 */ ((char *) (((int *) ( /* dim=0 */ (__pyx_v_result_view.data + __pyx_t_13 * __pyx_v_result_view.strides[0]) )) + __pyx_t_14)) )) = (__pyx_v_tmp + __pyx_v_c);
                          }
                      }
                  }
              }
          }
          #if ((defined(__APPLE__) || defined(__OSX__)) &amp;&amp; (defined(__GNUC__) &amp;&amp; (__GNUC__ &gt; 2 || (__GNUC__ == 2 &amp;&amp; (__GNUC_MINOR__ &gt; 95)))))
              #undef likely
              #undef unlikely
              #define likely(x)   __builtin_expect(!!(x), 1)
              #define unlikely(x) __builtin_expect(!!(x), 0)
          #endif
        }
/* … */
                              __pyx_t_13 = __pyx_v_x;
                              __pyx_t_14 = __pyx_v_y;
                              *((double *) ( /* dim=1 */ ((char *) (((double *) ( /* dim=0 */ (__pyx_v_result_view.data + __pyx_t_13 * __pyx_v_result_view.strides[0]) )) + __pyx_t_14)) )) = (__pyx_v_tmp + __pyx_v_c);
                          }
                      }
                  }
              }
          }
          #if ((defined(__APPLE__) || defined(__OSX__)) &amp;&amp; (defined(__GNUC__) &amp;&amp; (__GNUC__ &gt; 2 || (__GNUC__ == 2 &amp;&amp; (__GNUC_MINOR__ &gt; 95)))))
              #undef likely
              #undef unlikely
              #define likely(x)   __builtin_expect(!!(x), 1)
              #define unlikely(x) __builtin_expect(!!(x), 0)
          #endif
        }
/* … */
                              __pyx_t_13 = __pyx_v_x;
                              __pyx_t_14 = __pyx_v_y;
                              *((PY_LONG_LONG *) ( /* dim=1 */ ((char *) (((PY_LONG_LONG *) ( /* dim=0 */ (__pyx_v_result_view.data + __pyx_t_13 * __pyx_v_result_view.strides[0]) )) + __pyx_t_14)) )) = (__pyx_v_tmp + __pyx_v_c);
                          }
                      }
                  }
              }
          }
          #if ((defined(__APPLE__) || defined(__OSX__)) &amp;&amp; (defined(__GNUC__) &amp;&amp; (__GNUC__ &gt; 2 || (__GNUC__ == 2 &amp;&amp; (__GNUC_MINOR__ &gt; 95)))))
              #undef likely
              #undef unlikely
              #define likely(x)   __builtin_expect(!!(x), 1)
              #define unlikely(x) __builtin_expect(!!(x), 0)
          #endif
        }
</pre><pre class="cython line score-0">&#xA0;<span class="">41</span>: </pre>
<pre class="cython line score-6" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">42</span>:     <span class="k">return</span> <span class="n">result</span></pre>
<pre class='cython code score-6 '>  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_r);
  <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_v_result);
  __pyx_r = __pyx_v_result;
  goto __pyx_L0;
/* … */
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_r);
  <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_v_result);
  __pyx_r = __pyx_v_result;
  goto __pyx_L0;
/* … */
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_r);
  <span class='pyx_macro_api'>__Pyx_INCREF</span>(__pyx_v_result);
  __pyx_r = __pyx_v_result;
  goto __pyx_L0;
</pre></div></body></html>



## 6.2 Compare

Finally let's compare their speeds. If we look at Numpy's speed as 100%, then the last column is their comparison with the very first Numpy speed.


```python
import pandas as pd
from IPython.display import HTML

data = {
    'Methods': ['Numpy', 'Pure Python','Original Cython','Cython_add types','Cython_memoryviews','Cython_tuning indexing','Cyton_contiguous','Cython_multiple data types','Cython_multiple threads'],
    'Speed(s)': [compute_np_time, compute_py_time,compute_cy_time,compute_cy_t_time,compute_cy_m_time,compute_cy_i_time,compute_cy_c_time,compute_cy_mdt_time,compute_cy_mt_time],
    'Percentage(%)': [100, compute_np_time/compute_py_time*100,compute_np_time/compute_cy_time*100,compute_np_time/compute_cy_t_time*100,compute_np_time/compute_cy_m_time*100,compute_np_time/compute_cy_i_time*100,compute_np_time/compute_cy_c_time*100,compute_np_time/compute_cy_mdt_time*100,compute_np_time/compute_cy_mt_time*100]
}
df = pd.DataFrame(data)

# Creating style functions
def add_border(val):
    return 'border: 1px solid black'

# Applying style functions to data boxes
styled_df = df.style.applymap(add_border)

# Defining CSS styles
table_style = [
    {'selector': 'table', 'props': [('border-collapse', 'collapse')]},
    {'selector': 'th, td', 'props': [('border', '1px solid black')]}
]

# Adding styles to stylised data boxes
styled_df.set_table_styles(table_style)

# Displaying stylised data boxes in Jupyter Notebook
HTML(styled_df.to_html())
```




<style type="text/css">
#T_0de2c table {
  border-collapse: collapse;
}
#T_0de2c th {
  border: 1px solid black;
}
#T_0de2c  td {
  border: 1px solid black;
}
#T_0de2c_row0_col0, #T_0de2c_row0_col1, #T_0de2c_row0_col2, #T_0de2c_row1_col0, #T_0de2c_row1_col1, #T_0de2c_row1_col2, #T_0de2c_row2_col0, #T_0de2c_row2_col1, #T_0de2c_row2_col2, #T_0de2c_row3_col0, #T_0de2c_row3_col1, #T_0de2c_row3_col2, #T_0de2c_row4_col0, #T_0de2c_row4_col1, #T_0de2c_row4_col2, #T_0de2c_row5_col0, #T_0de2c_row5_col1, #T_0de2c_row5_col2, #T_0de2c_row6_col0, #T_0de2c_row6_col1, #T_0de2c_row6_col2, #T_0de2c_row7_col0, #T_0de2c_row7_col1, #T_0de2c_row7_col2, #T_0de2c_row8_col0, #T_0de2c_row8_col1, #T_0de2c_row8_col2 {
  border: 1px solid black;
}
</style>
<table id="T_0de2c">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_0de2c_level0_col0" class="col_heading level0 col0" >Methods</th>
      <th id="T_0de2c_level0_col1" class="col_heading level0 col1" >Speed(s)</th>
      <th id="T_0de2c_level0_col2" class="col_heading level0 col2" >Percentage(%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_0de2c_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_0de2c_row0_col0" class="data row0 col0" >Numpy</td>
      <td id="T_0de2c_row0_col1" class="data row0 col1" >0.058665</td>
      <td id="T_0de2c_row0_col2" class="data row0 col2" >100.000000</td>
    </tr>
    <tr>
      <th id="T_0de2c_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_0de2c_row1_col0" class="data row1 col0" >Pure Python</td>
      <td id="T_0de2c_row1_col1" class="data row1 col1" >10.784625</td>
      <td id="T_0de2c_row1_col2" class="data row1 col2" >0.543966</td>
    </tr>
    <tr>
      <th id="T_0de2c_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_0de2c_row2_col0" class="data row2 col0" >Original Cython</td>
      <td id="T_0de2c_row2_col1" class="data row2 col1" >8.321112</td>
      <td id="T_0de2c_row2_col2" class="data row2 col2" >0.705010</td>
    </tr>
    <tr>
      <th id="T_0de2c_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_0de2c_row3_col0" class="data row3 col0" >Cython_add types</td>
      <td id="T_0de2c_row3_col1" class="data row3 col1" >5.123431</td>
      <td id="T_0de2c_row3_col2" class="data row3 col2" >1.145028</td>
    </tr>
    <tr>
      <th id="T_0de2c_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_0de2c_row4_col0" class="data row4 col0" >Cython_memoryviews</td>
      <td id="T_0de2c_row4_col1" class="data row4 col1" >0.037777</td>
      <td id="T_0de2c_row4_col2" class="data row4 col2" >155.291081</td>
    </tr>
    <tr>
      <th id="T_0de2c_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_0de2c_row5_col0" class="data row5 col0" >Cython_tuning indexing</td>
      <td id="T_0de2c_row5_col1" class="data row5 col1" >0.014359</td>
      <td id="T_0de2c_row5_col2" class="data row5 col2" >408.563831</td>
    </tr>
    <tr>
      <th id="T_0de2c_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_0de2c_row6_col0" class="data row6 col0" >Cyton_contiguous</td>
      <td id="T_0de2c_row6_col1" class="data row6 col1" >0.014615</td>
      <td id="T_0de2c_row6_col2" class="data row6 col2" >401.401440</td>
    </tr>
    <tr>
      <th id="T_0de2c_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_0de2c_row7_col0" class="data row7 col0" >Cython_multiple data types</td>
      <td id="T_0de2c_row7_col1" class="data row7 col1" >0.014951</td>
      <td id="T_0de2c_row7_col2" class="data row7 col2" >392.370064</td>
    </tr>
    <tr>
      <th id="T_0de2c_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_0de2c_row8_col0" class="data row8 col0" >Cython_multiple threads</td>
      <td id="T_0de2c_row8_col1" class="data row8 col1" >0.040070</td>
      <td id="T_0de2c_row8_col2" class="data row8 col2" >146.405467</td>
    </tr>
  </tbody>
</table>




We can see that the final optimisation is already more than three times faster than the Numpy.
