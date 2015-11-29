import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s=pd.Series([1,3,6,np.nan, 9,0])
s1=pd.Series([1,"c",6,np.nan, 9,0])
s2=pd.Series(["ab","bu","da"],[1,5,3])
print(s2)
a=[[1,2,3],[4,5,6],[7,8,9]]
ind=['f','d','k']
df = pd.DataFrame(a, index=ind, columns=list('ABC'))
print(df)

df2 = pd.DataFrame({ "id": [1,2,3],
                     "name":["Ted", "Tom", "Jane"],
                     "score":np.array([5.,3.4,4.6]),
                     1.:s1[1:4],
                     'BD':pd.Timestamp('20130102'),
                     'sex':pd.Categorical(["m","m","f"])

    })

##print(df2)
##print(df2.dtypes)
##
##print(df2.head(1))
##print(df2.tail(1))
##
##print(s2.index)
##print(df2.columns)
##print(df2.values[0])
##df2.sort_index(axis=1, ascending=False)
print(df2.sort_values(by="score"))

df['A'] a single column
df[0:3] 1-3 строки
df['20130102':'20130104'] по значениям строк

Selection by Label
df.loc[dates[0]] первая строка(index=dates[0])
df.loc[:,['A','B']] столбцы а б
df.loc['20130102':'20130104',['A','B']] both endpoints are included по обим направлениям
f.loc['20130102',['A','B']] два атрибута объекта
df.loc[dates[0],'A'] a scalar value
df.at[dates[0],'A'] the same

Selection by Position
df.iloc[3] 4 строка
df.iloc[3:5,0:2] как в нумпи (4-5строки, 1-2 столбец)
df.iloc[[1,2,4],[0,2]] ряд условий на строки\столбцы
df.iloc[1:3,:] строки 2-3
df.iloc[:,1:3] столбцы 2-3
df.iloc[1,1] одно значение
df.iat[1,1] the same

Boolean Indexing
df[df.A > 0] Using a single column’s values to select data
df[df > 0]
df2[df2['E'].isin(['two','four'])] выбираем строки, у кот E=2,4
                   A         B         C         D     E
2013-01-03 -0.861849 -2.104569 -0.494929  1.071804   two
2013-01-05 -0.424972  0.567020  0.276232 -1.087401  four

Setting
df['F'] = s1 новый столбец серия s1
df.at[dates[0],'A'] = 0 первый элемент станет нулевым
df.iat[0,1] = 0 (1,2)=0
df.loc[:,'D'] = np.array([5] * len(df)) столбец D заполняем nympy array
df2[df2 > 0] = -df2
                   A         B         C  D   F
2013-01-01  0.000000  0.000000 -1.509059 -5 NaN
2013-01-02 -1.212112 -0.173215 -0.119209 -5  -1
2013-01-03 -0.861849 -2.104569 -0.494929 -5  -2
2013-01-04 -0.721555 -0.706771 -1.039575 -5  -3
2013-01-05 -0.424972 -0.567020 -0.276232 -5  -4
2013-01-06 -0.673690 -0.113648 -1.478427 -5  -5

Missing Data
pandas primarily uses the value np.nan to represent missing data. It is by default not included in computations. See the Missing Data section
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E']) формируем
                    новую структуры, выбираем строки и столбцы
df1.loc[dates[0]:dates[1],'E'] = 1 
df1.dropna(how='any') удаляем строки с любыми пропусками информации
df1.fillna(value=5) заполняем пропуски 5
pd.isnull(df1) ответ таблица True\False пропущ. зн.\не пропущ.

Operations
Stats
df.mean() среднее по столбцам (а=9,б=8...)
df.mean(1) среднее по строкам (2013 = 7, 2014 =4...)
s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2) сдвигаем [1,3,5...]
                            на 2 позиции вниз. кол. строк по dates
df.sub(s, axis='index') ? вычитаем из каждого столбца df s

Apply
df.apply(np.cumsum) Applying functions to the data
df.apply(lambda x: x.max() - x.min())
A    2.073961
B    2.671590
C    1.785291
D    0.000000
F    4.000000

Histogramming
s = pd.Series(np.random.randint(0, 7, size=10)) 10 чисел из 0-7, int
s.value_counts() ?

String Methods
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str.lower() все буквы маленькие

Merge
Concat

Concatenating pandas objects together with concat():
df = pd.DataFrame(np.random.randn(10, 4)) 10 строк и 4 столбца
pieces = [df[:3], df[3:7], df[7:]] разделяем на куски по строкам
pd.concat(pieces) сцепляем все вместе

Join
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
   key  lval
0  foo     1
1  foo     2
   key  rval
0  foo     4
1  foo     5
pd.merge(left, right, on='key') объединение
   key  lval  rval
0  foo     1     4
1  foo     1     5
2  foo     2     4
3  foo     2     5

Append
df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
s = df.iloc[3] выделяем 4 строку
df.append(s, ignore_index=True) добавляем снизу к df

Grouping
By “group by” we are referring to a process involving one or more of the following
steps

Splitting the data into groups based on some criteria
Applying a function to each group independently
Combining the results into a data structure
See the Grouping section

 df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
                           'foo', 'bar', 'foo', 'foo'],
                    'B' : ['one', 'one', 'two', 'three',
                           'two', 'two', 'one', 'three'],
                    'C' : np.random.randn(8),
                    'D' : np.random.randn(8)}) 
     A      B         C         D
0  foo    one -1.202872 -0.055224
1  bar    one -1.814470  2.395985
2  foo    two  1.018601  1.552825
3  bar  three -0.595447  0.166599
4  foo    two  1.395433  0.047609
5  bar    two -0.392670 -0.136473
6  foo    one  0.007207 -0.561757
7  foo  three  1.928123 -1.623033
df.groupby('A').sum() группируем по столбцу А и суммируем числ. значения
df.groupby(['A','B']).sum() группируем по 2 столбцам
                  C         D
A   B                        
bar one   -1.814470  2.395985
    three -0.595447  0.166599
    two   -0.392670 -0.136473
foo one   -1.195665 -0.616981
    three  1.928123 -1.623033
    two    2.414034  1.600434

Reshaping
Stack
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                    'foo', 'foo', 'qux', 'qux'],
                   ['one', 'two', 'one', 'two',
                    'one', 'two', 'one', 'two']]))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df2 = df[:4] 
                     A         B
first second                    
bar   one     0.029399 -0.542108
      two     0.282696 -0.087302
baz   one    -1.575170  1.771208
      two     0.816482  1.100230
stacked = df2.stack() “compresses” a level
first  second   
bar    one     A    0.029399
               B   -0.542108
       two     A    0.282696
               B   -0.087302
baz    one     A   -1.575170
               B    1.771208
       two     A    0.816482
               B    1.100230
stacked.unstack() обратная операция
                     A         B
first second                    
bar   one     0.029399 -0.542108
      two     0.282696 -0.087302
baz   one    -1.575170  1.771208
      two     0.816482  1.100230
stacked.unstack(1) по 2 атрибуту
second        one       two
first                      
bar   A  0.029399  0.282696
      B -0.542108 -0.087302
baz   A -1.575170  0.816482
      B  1.771208  1.100230
stacked.unstack(0) по 1 атрибуту
first          bar       baz
second                      
one    A  0.029399 -1.575170
       B -0.542108  1.771208
two    A  0.282696  0.816482
       B -0.087302  1.100230

Pivot Tables
df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
                    'B' : ['A', 'B', 'C'] * 4,
                    'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                    'D' : np.random.randn(12),
                    'E' : np.random.randn(12)})
        A  B    C         D         E
0     one  A  foo  1.418757 -0.179666
1     one  B  foo -1.879024  1.291836
2     two  C  foo  0.536826 -0.009614
3   three  A  bar  1.006160  0.392149
4     one  B  bar -0.029716  0.264599
5     one  C  bar -1.146178 -0.057409
6     two  A  foo  0.100900 -1.425638
7   three  B  foo -1.035018  1.024098
8     one  C  foo  0.314665 -0.106062
9     one  A  bar -0.773723  1.824375
10    two  B  bar -1.170653  0.595974
11  three  C  bar  0.648740  1.167115
pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C']) структурируем по атр.
    с неб. кол. значений
C             bar       foo
A     B                    
one   A -0.773723  1.418757
      B -0.029716 -1.879024
      C -1.146178  0.314665
three A  1.006160       NaN
      B       NaN -1.035018
      C  0.648740       NaN
two   A       NaN  0.100900
      B -1.170653       NaN
      C       NaN  0.536826
      
Time Series
rng = pd.date_range('1/1/2012', periods=100, freq='S') строка в сто секунд
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng) ? 500 случ.знач. на пред. интервале
ts.resample('5Min', how='sum') типо сумма случ. знач. 
2012-01-01    25083
Freq: 5T, dtype: int32

Time zone representation
rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D') 5 дней
ts = pd.Series(np.random.randn(len(rng)), rng)  5 случ. чисел из странного диапозона
ts.tz_localize('UTC') добавялем время
2012-03-06 00:00:00+00:00    0.464000
2012-03-07 00:00:00+00:00    0.227371
2012-03-08 00:00:00+00:00   -0.496922
2012-03-09 00:00:00+00:00    0.306389
2012-03-10 00:00:00+00:00   -2.290613
ts_utc.tz_convert('US/Eastern') переходим в другой час. пояс
2012-03-05 19:00:00-05:00    0.464000
2012-03-06 19:00:00-05:00    0.227371
2012-03-07 19:00:00-05:00   -0.496922
2012-03-08 19:00:00-05:00    0.306389
2012-03-09 19:00:00-05:00   -2.290613

Converting between time span representations
rng = pd.date_range('1/1/2012', periods=5, freq='M')
ts = pd.Series(np.random.randn(len(rng)), index=rng) 
2012-01-31   -1.134623
2012-02-29   -1.561819
2012-03-31   -0.260838
2012-04-30    0.281957
2012-05-31    1.523962
ps = ts.to_period()
2012-01   -1.134623
2012-02   -1.561819
2012-03   -0.260838
2012-04    0.281957
2012-05    1.523962
ps.to_timestamp()
2012-01-01   -1.134623
2012-02-01   -1.561819
2012-03-01   -0.260838
2012-04-01    0.281957
2012-05-01    1.523962

Converting between period and timestamp enables some convenient arithmetic functions to be used. In the following example, we convert a quarterly frequency with year ending in November to 9am of the end of the month following the quarter end:
prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
ts = pd.Series(np.random.randn(len(prng)), prng)
ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
ts.head()
1990-03-01 09:00   -0.902937
1990-06-01 09:00    0.068159
1990-09-01 09:00   -0.057873
1990-12-01 09:00   -0.368204
1991-03-01 09:00   -1.144073

Categoricals
df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'a', 'e']})
Convert the raw grades to a categorical data type.
df["grade"] = df["raw_grade"].astype("category")
0    a
1    b
2    b
3    a
4    a
5    e
Name: grade, dtype: category
Categories (3, object): [a, b, e]
df["grade"].cat.categories = ["very good", "good", "very bad"] Rename the categories to more meaningful names 
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
    Reorder the categories and simultaneously add the missing categories
0    very good
1         good
2         good
3    very good
4    very good
5     very bad
df.sort_values(by="grade") Sorting is per order in the categories, not lexical order.
   id raw_grade      grade
5   6         e   very bad
1   2         b       good
2   3         b       good
0   1         a  very good
3   4         a  very good
4   5         a  very good
df.groupby("grade").size() сколько в каждой категории

Plotting docs.
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
              columns=['A', 'B', 'C', 'D']) рисует все столбцы
df = df.cumsum()
plt.figure(); df.plot(); plt.legend(loc='best')

Getting Data In/Out
CSV
df.to_csv('foo.csv')
pd.read_csv('foo.csv')

HDF5
df.to_hdf('foo.h5','df')
pd.read_hdf('foo.h5','df')

Excel

df.to_excel('foo.xlsx', sheet_name='Sheet1')
pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
