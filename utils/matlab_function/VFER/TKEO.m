function TKEO_HZ=TKEO(x);
M=size(x,2);
    for q=2:M-1
        TKEO_HZ(q-1)=x(q)*x(q)-x(q-1)*x(q+1);        
    end
    