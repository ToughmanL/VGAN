function F_n=DFA(DATA,win_length,order)           
N=length(DATA);      
n=floor(N/win_length);       
N1=n*win_length;      
y=zeros(N1,1);       
Yn=zeros(N1,1);           
fitcoef=zeros(n,order+1);          
mean1=mean(DATA(1:N1));       
for i=1:N1         
    y(i)=sum(DATA(1:i)-mean1);    
end 
y=y';      
for j=1:n 
    fitcoef(j,:)=polyfit(1:win_length,y(((j-1)*win_length+1):j*win_length),order);  
end 
for j=1:n     
    Yn(((j-1)*win_length+1):j*win_length)=polyval(fitcoef(j,:),1:win_length);   
end 
sum1=sum((y'-Yn).^2)/N1;   
sum1=sqrt(sum1);      
F_n=sum1; 
t=1:0.05:3; 
n=zeros(1,length(t)); 
for i=1:length(t) 
    n(i)=10^t(i); 
end 
n=floor(n);n=n'; 
len=length(n); 
F_n=zeros(len,1); 
 
 
for i=1:len    
    F_n(i)=DFA(data,n(i),1); 
end 
p=polyfit(log10(n),log10(F_n),1); 
 
plot(n,F_n(:,1),'o'); 
x=n; 
y=p(2)+p(1)*log10(x); 
for i=1:len   
    y(i)=10^y(i); 
end 
hold onplot(x,y,'black'); 