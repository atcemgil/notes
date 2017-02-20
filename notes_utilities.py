## Contains utilities

import numpy as np

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




def pdf2latex_gauss(x=r'x', m=r'\mu', v=r'v', N=r'N'):
    L = [r'\mathcal{{{N}}}({x}; {m}, {v})'.format(x=x, m=m, v=v, N=N) ]
    rv = r'\frac{{1}}{{\sqrt{{2\pi {v}}} }} \exp\left(-\frac12 \frac{{({x} - {m} )^2}}{{{v}}}\right)'.format(x=x, m=m, v=v)
    L.append(rv)
    rv = r'\exp\left(-\frac{{1}}{{2}}\frac{{{x}^2}}{{{v}}} + \frac{{{x} {m} }}{{{v}}} -\frac{{1}}{{2}}\frac{{{m}^2}}{{{v}}} -\frac{{1}}{{2}}\log(2{{\pi}}{v})    \right)'.format(x=x, m=m, v=v)
    L.append(rv)
    rv = r'-\frac{{1}}{{2}}\frac{{{x}^2}}{{{v}}} + \frac{{{x} {m} }}{{{v}}} -\frac{{1}}{{2}}\frac{{{m}^2}}{{{v}}} -\frac{{1}}{{2}}\log {v} -\frac{{1}}{{2}}\log 2\pi'.format(x=x, m=m, v=v)
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
