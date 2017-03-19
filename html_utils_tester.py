from IPython.display import display, Math, Latex, HTML
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl

MAX = 40

nf = mpl.colors.Normalize(vmin=0, vmax=1.5*MAX, clip=True)
cmap = plt.cm.ScalarMappable(cmap=plt.cm.hot_r, norm=nf)

M = 10
N = 20

data = np.random.choice(range(MAX),replace=True, size=(M,N))

htmlTable = Table()
for i in range(M):
    row = []
    for j in range(N):
        col = mpl.colors.rgb2hex(cmap.to_rgba(data[i,j]))
        htmlCell = TableCell(data[i,j],style='background-color:'+col)
        row.append(htmlCell)

    htmlTableRow = TableRow(row)
    htmlTable.rows.append(htmlTableRow)
    
#print(str(htmlTable))
display(HTML(str(htmlTable)) )


#------------------

t = Table()
t.rows.append(TableRow(['A', 'B', 'C'], header=True))
t.rows.append(TableRow(['D', 'E', 'F']))
t.rows.append(('i', 'j', 'k'))
display(HTML(str(t)))

# ----------
t2 =  Table([
        ('1', '2'),
        ['3', '4']
    ], width='100%', header_row=('col1', 'col2'),
    col_width=('', '75%'))

display(HTML(str(t2)))

t2.rows.append(['5', '6'])
t2.rows[1][1] = TableCell('new', bgcolor='red')
t2.rows.append(TableRow(['7', '8'], attribs={'align': 'center'}))
display(HTML(str(t2)))

# ----------

# sample table with column attributes and styles:
table_data = [
        ['Smith',       'John',         30,    4.5],
        ['Carpenter',   'Jack',         47],
        ['Johnson',     'Paul',         62,    10.55],
    ]
htmlcode = make_htmlTable(table_data,
    header_row = ['Last name',   'First name',   'Age', 'Score'],
    col_width=['', '1%', '5%', '5%'],
    col_align=['left', 'center', 'right', 'char'],
    col_styles=['font-size: large', '', 'font-size: small', 'background-color:yellow'])

HTML(str(htmlcode))
