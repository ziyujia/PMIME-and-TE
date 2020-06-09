function xM = aleCoupledhenonmaps2(m,c,n,noise,aV,x0M)
% xM = coupledhenonmaps2(m,c,n,aV,x0V)
% COUPLEDHENONMAPS2 generates m time series from identical coupled Henon
% maps with coupling strength c. Each map is coupled to the next apart from
% the first and last map, the first only drives the second and the last
% only drives the second last (first and last are not coupled).
% COUPLEDHENONMAPS2 从具有耦合强度c的相同耦合Henon映射生成m个时间序列。 
%每个地图除了第一个和最后一个地图之外都耦合到下一个地图，第一个地图仅驱动第二个地图，
%而最后一个地图仅驱动第二个地图（第一个和最后一个没有耦合）。
% INPUTS
% - m       : the number of coupled maps (time series)
% - c       : the coupling strength
% - n       : the time series length
% - noise   : noise to add
% - aV      : the two parameters in a vector (if omitted the standard
%         parameters for chaos are chosen)
% - x0M     : initial conditions, a matrix of size 2 x m
% OUTPUTS
% - xM      : the n x m matrix of the generated time series
% Reference
% Antonio Politi and Alessandro Torcini
% Periodic orbits in coupled H?non maps: Lyapunov and multifractal analysis
% Chaos 2, 293 (1992); doi:10.1063/1.165871 (8 pages)
% Xunhe Yin  Yong Rena and Xiuming Shana
% Synchronization of discrete spatiotemporal chaos by using variable structure control
% Chaos, Solitons & Fractals, Volume 14, Issue 7, October 2002, Pages 1077-1082

if nargin==5
    x0M = NaN;
elseif nargin==4
    x0M = NaN;
    aV = [1.4 0.3];
end
if isempty(aV)
    aV = [1.4 0.3];
end  
if length(c)==1
    cV = c*ones(m,1);
else
    cV = c;
end
if isnan(x0M)
    ntrans = 1000;
    x0limV = [-1 1 -0.3 0.3]';
    xM = NaN*ones(n+ntrans,m);
    tmpM = rand(2,m);
    xM(1,:)=(x0limV(4)-tmpM(1,:))/(x0limV(4)-x0limV(3));
    xM(2,:)=(x0limV(2)-tmpM(2,:))/(x0limV(2)-x0limV(1));
else
    ntrans = 2;
    xM(1:2,:)= x0M;
end
for t=3:n+ntrans
    xM(t,1)= aV(1) - xM(t-1,1)^2 + aV(2)*xM(t-2,1) + noise*rand(1);
    for im=2:m-1
        xM(t,im)= aV(1) - (0.5*cV(im)*(xM(t-1,im-1)+xM(t-1,im+1)) + ...
            (1-cV(im))*xM(t-1,im))^2 + aV(2)*xM(t-2,im) + noise*rand(1);
    end
    xM(t,m)= aV(1) - xM(t-1,m)^2 + aV(2)*xM(t-2,m) + noise*rand(1);
end
xM = xM(ntrans+1:n+ntrans,:);
