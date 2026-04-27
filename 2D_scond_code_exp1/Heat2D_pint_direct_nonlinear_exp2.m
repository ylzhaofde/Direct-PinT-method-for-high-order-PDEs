%Direct PinT leapfrog scheme for 2D nonlinear heat equation
%u_tt-c1*u_t=Delta_u+u+nlfun(u)+f(x,,y,t),  u=0 on BC; u(x,,y,0)=u0(x,y);
clear
%%(B) Set the problem data: rhs, exact solutions, parameters
T=1; xa=0; xb=2*pi;
y0=@(x,y) sin(x).*sin(y);
y0_t=@(x,y) -sin(x).*sin(y);
% y0_tt=@(x,y) 4*sin(x).*sin(y);
y_sol=@(x,y,t) exp(-t).*sin(x).*sin(y);% require zero boundary condition
nlfun=@(z) z.^2 + 2*z;               %nonlinear functions
% nlfunjac=@(z) (3*z.^2-ones(size(z))); 
%nlfun=@(z) sqrt(z); nlfunjac=@(z) 0.5./sqrt(z);
f=@(x,y,t,c1) -(exp(-t).*sin(x).*sin(y)).^2 + nlfun(y_sol(x,y,t)) ...
    + c1*y_sol(x,y,t);

maxL=9; % increase for larger mesh
 Nx=2.^(3:maxL);
Nt=T*2.^(3:maxL); levmax=length(Nx);
fprintf('(Nx,Nx,Nt) \t Error\t\t Order\t  CPU\t Iter \n');
y_err_0=0;
for s=1:levmax
    tic
    nx=Nx(s);m=nx-1; nt=Nt(s);c1=-1/4;
    dt=T/nt; h=(xb-xa)/nx; xx=xa+h:h:xb-h; %interior nodes in space
    [X,Y] = meshgrid(xx,xx);
    [XX,YY,TT] = meshgrid(xx,xx,dt:dt:T);%not including initial time step
   
    Ix=speye(m^2,m^2); It=speye(nt,nt);
%     Ixx=speye(m,m);Iyy=speye(m,m);
%     e1=ones(Nx-1,1);
%     A= 1/h^2*spdiags([e1,-2*e1,e1],[-1 0 1],Nx - 1,Nx - 1);% central finite difference
%     Ax=kron(A,Iyy)+kron(Ixx,A);
    Ax=-(1/h^2)*gallery('poisson',m);% central finite difference

    e_t=ones(nt,1);
    At= spdiags([-e_t/2 e_t/2],[-1 1],nt,nt)/dt; %time scheme
    At(end,end-2:end)=[1/2 -2 3/2]/dt; %fix last row for backward Euler
    
    F1=f(XX,YY,TT,c1); 
%     F1(:,:,1)=y0(X,Y)/(2*dt); %adjust rhs for first step
    F=zeros(nt,m^2);
     
    b1=[reshape(y0(X,Y)/(2*dt),1,m^2);zeros(nt-1,m^2)];
    b2=[reshape(y0_t(X,Y)/(2*dt),1,m^2);zeros(nt-1,m^2)];
%   b2=[reshape(y0_tt(X,Y)/(2*dt),1,m^2);reshape(-y0_t(X,Y)/(4*dt^2),1,m^2);zeros(nt-2,m^2)];
    b=(At-c1*It)*b1 + b2;

      
    for i=1:nt
        f1=F1(:,:,i);
        f1=reshape(f1,1,m^2);
        F(i,:)=f1;
    end
%     F=reshape(F,m^2,nt)';  
    %A=kron(Ix,At)+kron(Ax,It);%system matrix, no need to construct
    [Vs,Ds] = eig(full(At),'vector'); %factorize At
    
    %Newton iteration
    maxit=50;tol=1e-8;
    y_h=zeros(nt,m^2);%initial guess
    for iter=1:maxit
        %fjac=spdiags(nlfunjac(mean(y_h)'),0,m^2,m^2); %mean of y_h in time
%         fjac=spdiags(mean(nlfunjac(y_h))',0,m^2,m^2); %mean of nlfun'(y_h),bett
        Res=b+F;%residual   
        
        R1=Vs\Res; %step (a)
        parfor j=1:nt %step (b)
            R1(j,:)=((((Ds(j)^2-c1*Ds(j)-1)*Ix-Ax))\R1(j,:).').';  %parallel in time
        end
        y_h_new=Vs*R1;%step (c)
        if(norm(y_h_new(:)-y_h(:),inf)<tol)
            break;
        end
        y_h=y_h_new;
    end
    %measure error
    ysol=y_sol(XX,YY,TT); 
    y_sol1=zeros(nt,m^2);
    for i=1:nt
        y_ext=ysol(:,:,i);
       
        y_sol1(i,:)=reshape(y_ext,1,m^2);
    end                              %exact solution
    ysol=y_sol1;
    y_err=norm(y_h(:)-ysol(:),inf);%maxmum error
    fprintf('(%3d,%3d,%3d)&\t %1.2e& \t%1.2f\t&%1.3f&\t%d \n',...
        nx,nx,nt,y_err,log2(y_err_0/y_err),toc,iter)
    y_err_0=y_err;
end
