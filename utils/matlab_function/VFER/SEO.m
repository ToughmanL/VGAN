function SEO_HZ=SEO(x)
M=size(x,2);
for j=1:M
    SEO_HZ(j)=x(j)*x(j);
end