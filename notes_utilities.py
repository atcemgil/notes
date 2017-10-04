## Contains utilities

import numpy as np


# Utilities for discrete distributions

def randgen(pr, N=1): 
    L = len(pr)
    return int(np.random.choice(range(L), size=N, replace=True, p=pr))

def log_sum_exp(l, axis=0):
    l_star = np.max(l, axis=axis, keepdims=True)
    return l_star + np.log(np.sum(np.exp(l - l_star),axis=axis,keepdims=True)) 

def normalize_exp(log_P, axis=None):
    a = np.max(log_P, keepdims=True, axis=axis)
    P = normalize(np.exp(log_P - a), axis=axis)
    return P

def normalize(A, axis=None):
    Z = np.sum(A, axis=axis,keepdims=True)
    idx = np.where(Z == 0)
    Z[idx] = 1
    return A/Z

## 

def GivensMat(th):
    c = np.cos(th)
    s = np.sin(th)
    return np.array([[c,s],[-s,c]])

def pnorm_ball_points(A=np.eye(2), mu=np.array([0,0]),p=2, N=128):
    '''
    Creates the points on a p normball y = A x + \mu 
    where x are points on the unit circle.
    '''
    if p is np.infty:
    	X = np.array([[1,1],[1,-1],[-1,-1],[-1,1],[1,1]]).T
    else:
	    th = np.arange(0, 2*np.pi+np.pi/N, np.pi/N)
	    X = np.array([np.cos(th),np.sin(th)])
	    X = X*(1/(np.abs(X[0,:])**p + np.abs(X[1,:])**p ))**(1/p)


    Y = np.dot(A, X)
    data_x = mu[0]+Y[0,:]
    data_y = mu[1]+Y[1,:]
    return data_x, data_y

import matplotlib.pylab as plt

def pnorm_ball_line(A=np.eye(2), mu=np.array([0,0]),p=2, N=128,color='r',linewidth=3):
	'''	Creates line objects. Show them with ax.add_line(ln) '''
	dx,dy = pnorm_ball_points(A, mu)
	ln = plt.Line2D(dx,dy, color=color, linewidth=linewidth)
	return ln

def mat2latex(a,dollar=False):
    """Returns a LaTeX string for typesetting a matrix

    :a: numpy array
    :returns: LaTeX string
    """
    if len(a.shape) > 2:
        raise ValueError('mat2latex can display only matrices or vectors')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\left(\begin{array}{cc}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{array}\right)']
    if dollar:
        return '$'.join(rv)+'$'		
    else:
        return ''.join(rv)

def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'$\left(\begin{array}{cc}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{array}\right)$']
    return ''.join(rv)


## Standard Densities
##
##


def eqs2html_table(L):
    html = ["<table width=100%>"]
    for eq in L:
        html.append("<tr>")
        html.append('<td align="center">{0}</td>'.format(r'\begin{eqnarray}'+eq+r'\end{eqnarray}'))
        html.append("<td>{0}</td>".format(eq))
        html.append("</tr>")
    html.append("</table>")

    html = ''.join(html)
    return html


def pdf2latex_gauss(x=r'x', m=r'\mu', v=r'v', N=r'N'):
    L = [r'\mathcal{{{N}}}({x}; {m}, {v})'.format(x=x, m=m, v=v, N=N) ]
    rv = r'\frac{{1}}{{\sqrt{{2\pi {v}}} }} \exp\left(-\frac12 \frac{{({x} - {m} )^2}}{{{v}}}\right)'.format(x=x, m=m, v=v)
    L.append(rv)
    rv = r'\exp\left(-\frac{{1}}{{2}}\frac{{{x}^2}}{{{v}}} + \frac{{{x} {m} }}{{{v}}} -\frac{{1}}{{2}}\frac{{{m}^2}}{{{v}}} -\frac{{1}}{{2}}\log(2{{\pi}}{v})    \right)'.format(x=x, m=m, v=v)
    L.append(rv)
    rv = r'-\frac{{1}}{{2}}\frac{{{x}^2}}{{{v}}} + \frac{{{x} {m} }}{{{v}}} -\frac{{1}}{{2}}\frac{{{m}^2}}{{{v}}} -\frac{{1}}{{2}}\log {v} -\frac{{1}}{{2}}\log 2\pi'.format(x=x, m=m, v=v)
    L.append(rv)
    return L

def pdf2latex_mvnormal(x=r'x', m=r'\mu', v=r'\Sigma', N=r'N'):

    L = []

    if m==0:
        rv = r'\mathcal{{{N}}}({x}; {m}, {v})'.format(x=x, m=m, v=v, N=N) 
        L.append(rv)
        rv = r'\left|{{ 2\pi {v} }} \right|^{{-1/2}} \exp\left(-\frac12 {{{x}}}^\top {{{v}}}^{{-1}} {{{x}}} \right)'.format(x=x, m=m, v=v)
        L.append(rv)
        logpdf = r'  -\frac{{1}}{{2}}\trace {{{v}}}^{{-1}} {{{x}}}{{{x}}}^\top  -\frac{{1}}{{2}}\log \left|2{{\pi}}{v}\right|'.format(x=x, m=m, v=v)     
    else:
        rv = r'\mathcal{{{N}}}({x}; {m}, {v})'.format(x=x, m=m, v=v, N=N) 
        L.append(rv)
        rv = r'\left|{{ 2\pi {v} }} \right|^{{-1/2}} \exp\left(-\frac12 ({{{x}}} - {{{m}}} )^\top {{{v}}}^{{-1}} ({{{x}}} - {{{m}}} ) \right)'.format(x=x, m=m, v=v)
        L.append(rv)
        logpdf = r'  -\frac{{1}}{{2}}\trace {{{v}}}^{{-1}} {{{x}}}{{{x}}}^\top + \trace {{{v}}}^{{-1}} {{{x}}}{{{m}}}^\top -\frac{{1}}{{2}}\trace {{{v}}}^{{-1}} {{{m}}}{{{m}}}^\top  -\frac{{1}}{{2}}\log \left|2{{\pi}}{v}\right|'.format(x=x, m=m, v=v) 

    rv = r'\exp\left('+logpdf+ r'\right)'
    L.append(rv)
    rv = logpdf
    L.append(rv)
    return L

def pdf2latex_gamma(x=r'x', a=r'a', b=r'b', G=r'G'):
    L = [r'\mathcal{{{G}}}({x}; {a}, {b})'.format(x=x, a=a, b=b, G=G) ]
    rv = r'\frac{{ {b}^{a} {x}^{{{a}-1}}}}{{\Gamma({a})}} \exp\left(-{{{b}}} {{{x}}}\right)'.format(x=x, a=a, b=b)
    L.append(rv)
    rv = r'\exp(({{{a}}} - 1)\log {x} - {{{b}}}{{{x}}} - \log \Gamma({{{a}}}) + {{{a}}} \log {{{b}}})'.format(x=x, a=a, b=b)  
    L.append(rv)
    rv = r'({{{a}}} - 1)\log {x} - {{{b}}}{{{x}}} - \log \Gamma({{{a}}}) + {{{a}}} \log {{{b}}}'.format(x=x, a=a, b=b)  
    L.append(rv)    
    return L

def pdf2latex_invgamma(x=r'v', a=r'a', b=r'b', IG=r'IG'):
    L = [r'\mathcal{{{IG}}}({x}; {a}, {b})'.format(x=x, a=a, b=b, IG=IG) ]
    rv = r'\frac{{ {b}^{a} }}{{\Gamma({a}){x}^{{{a}+1}}}} \exp\left(-\frac{{{{{b}}}}}{{{{{x}}}}}\right)'.format(x=x, a=a, b=b)
    L.append(rv)
    rv = r'\exp(-({{{a}}} + 1)\log {x} - \frac{{{{{b}}}}}{{{{{x}}}}} - \log \Gamma({{{a}}}) + {{{a}}} \log {{{b}}})'.format(x=x, a=a, b=b)  
    L.append(rv)
    rv = r'-({{{a}}} + 1)\log {x} - \frac{{{{{b}}}}}{{{{{x}}}}} - \log \Gamma({{{a}}}) + {{{a}}} \log {{{b}}}'.format(x=x, a=a, b=b)  
    L.append(rv)    
    return L

def pdf2latex_beta(x=r'w', a=r'a', b=r'b', B=r'B'):
    L = [r'\mathcal{{{B}}}({x}; {a}, {b})'.format(x=x, a=a, b=b, B=B) ]
    rv = r'\frac{{ \Gamma({a}+{b}) }}{{ \Gamma({a}) \Gamma({b}) }} {x}^{{{a}-1}} (1-{x})^{{{b}-1}}'.format(x=x, a=a, b=b)
    L.append(rv)
    rv = r'\exp\left(({{{a}}} - 1)\log {x} + ({{{b}}} - 1)\log (1-{x}) + \log\Gamma({{{a}}}+{{{b}}}) - \log \Gamma({{{a}}}) - \log \Gamma({{{b}}})\right)'.format(x=x, a=a, b=b)  
    L.append(rv)
    rv = r'({{{a}}} - 1)\log {x} + ({{{b}}} - 1)\log (1-{x}) + \log\Gamma({{{a}}}+{{{b}}}) - \log \Gamma({{{a}}}) - \log \Gamma({{{b}}})'.format(x=x, a=a, b=b)  
    L.append(rv)    
    return L


def pdf2latex_bernoulli(x=r'c', th=r'theta', BE='BE'):
    L = [r'\mathcal{{{BE}}}({x}; {th})'.format(x=x, th=th) ]
    rv = r'{{{th}}}^{{{x}_0}}_0 {{{th}}}^{{{x}_1}}_1'.format(x=x, th=th)
    L.append(rv)
    rv = r'\exp\left({x}_0 \log {th}_0 + {x}_1 \log {th}_1\right)'.format(x=x, th=th)
    L.append(rv)
    rv = r'{x}_0 \log {th}_0 + {x}_1 \log {th}_1'.format(x=x, th=th)
    L.append(rv)    
    return L

## -----------------------------

def pdf2latex_dirichlet(x=r'w', a=r'a', N=r'N', D=r'D',i=r'u'):
    L = [r'\mathcal{{{D}}}({x}_{{1:{N}}}; {a}_{{1:{N}}} )'.format(x=x, N=N, a=a, D=D, i=i) ]
    rv = r'\frac{{\Gamma(\sum_{{{i}}} {a}_{{{i}}})}}{{\prod_{{{i}}} \Gamma({a}_{{{i}}})}} \prod_{{{{{i}}}=1}}^{{{N}}} {{{x}}}_{{{i}}}^{{{a}_{{{i}}} - 1}} '.format(x=x, a=a, N=N, i=i)
    L.append(rv)
    rv = r'\log{{\Gamma(\sum_{{{i}}} {a}_{{{i}}})}} - {{\sum_{{{i}}} \log \Gamma({a}_{{{i}}})}} + \sum_{{{{{i}}}=1}}^{{{N}}} ({a}_{{{i}}} - 1) \log{{{x}}}_{{{i}}} '.format(x=x, a=a, N=N, i=i)
    L.append(r'\exp\left('+rv+r'\right)')
    L.append(rv)
    return L


## -----------------------------

def pdf2latex_invwishart(X=r'X', nu=r'\nu', S=r'S', k=r'k', IW='IW'):
    L = [r'\mathcal{{{IW}}}_{k}({X}; {nu}, {S} )\;\;\;\Gamma_{k}({nu}/2) = \pi^{{{k}({k}-1)/4}} \prod_{{i=1}}^{{{k}}} \Gamma({nu}/2 - (i-1)/2)'.format(X=X, nu=nu, S=S, IW=IW, k=k)  ]
    rv = r'\frac{{ |{S}/2|^{{ {nu} /2}} }}{{\left|{X}\right|^{{{{( {{ {nu} }} + {k} + 1)}}/{{2}}}} \Gamma_{k}({nu}/2) }} \exp\left( - \trace ({S}/2) {X}^{{-1}}\right)'.format(X=X, nu=nu, S=S, k=k, IW=IW)
    L.append(rv)
    lw = r'-\frac{{ {{ {nu} }} + {k} + 1}}{{2}} \log \left|{X}\right| - \trace ({S}/2) {X}^{{-1}} + \frac{{ {nu} }}{{2}}\log |{S}/2| - \log\Gamma_{k}({nu}/2)'.format(X=X, nu=nu, S=S, k=k, IW=IW)
    rv = r'\exp\left('+lw+r'\right)' 
    L.append(rv)
    L.append(lw)    
    return L

def pdf2latex_wishart(X=r'X', nu=r'\nu', S=r'S', k=r'k', W='W'):
    L = [r'\mathcal{{{W}}}_{k}({X}; {nu}, {S} )\;\;\;\Gamma_{k}({nu}/2) = \pi^{{{k}({k}-1)/4}} \prod_{{i=1}}^{{{k}}} \Gamma({nu}/2 - (i-1)/2)'.format(X=X, nu=nu, S=S, W=W, k=k)  ]
    rv = r'\frac{{ \left|{X}\right|^{{{{( {{ {nu} }} - {k} - 1)}}/{{2}}}} }}{{ |2{S}|^{{{nu} /2}} \Gamma_{k}({nu}/2) }} \exp\left( - \trace (2{S})^{{-1}} {X}\right)'.format(X=X, nu=nu, S=S, k=k, W=W)
    L.append(rv)
    lw = r'\frac{{ {{ {nu} }} - {k} - 1}}{{2}} \log \left|{X}\right| - \trace (2{S})^{{-1}} {X} - \frac{{ {nu} }}{{2}}\log |2{S}| - \log\Gamma_{k}({nu}/2)'.format(X=X, nu=nu, S=S, k=k, W=W)
    rv = r'\exp\left('+lw+r'\right)' 
    L.append(rv)
    L.append(lw)    
    return L





