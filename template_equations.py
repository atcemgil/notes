from IPython.display import display, Math, Latex, HTML
import notes_utilities as nut
from importlib import reload
#reload(nut)
display(Latex(r'$\DeclareMathOperator{\trace}{Tr}$'))

#L = nut.pdf2latex_gauss(x=r's', m=r'\mu',v=r'v')
L = nut.pdf2latex_mvnormal(x=r's', m=r'\mu',v=r'\Sigma')
#L = nut.pdf2latex_mvnormal(x=r'x_t', m=r'(Ax_{t-1})',v=r'Q')
#L = nut.pdf2latex_mvnormal(x=r'y_t', m=r'(Cx_{t})',v=r'R')
#L = nut.pdf2latex_mvnormal(x=r's', m=0,v=r'I')
#L = nut.pdf2latex_gamma(x=r'x', a=r'a',b=r'b')
#L = nut.pdf2latex_invgamma(x=r'x', a=r'a',b=r'b')
#L = nut.pdf2latex_beta(x=r'\pi', a=r'\alpha',b=r'\beta')

eq = L[0]+'='+L[1]+'='+L[2]
display(Math(eq))
display(Latex(eq))
display(HTML(nut.eqs2html_table(L)))