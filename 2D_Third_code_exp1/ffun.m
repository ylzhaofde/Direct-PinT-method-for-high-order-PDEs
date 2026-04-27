function f = ffun(x,y,t)
% source term
f = -2*exp(-2*t)*sin(x)*sin(y);