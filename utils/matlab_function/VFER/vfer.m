function VFER = vfer(x,Fs)

% Downsample to 16kHz
if Fs > 16000
    Fsd = 16000;
    ratio = Fsd/Fs;
    s = resample(x,round(ratio*1000),1000);
    %   if vflag; Tind = ceil(Tind*Fsd/Fs); end
else
    s = x;
    Fsd = 16000;
end

%�Ȱ��²������s�źŸ���DYSPA�����ſ��͹رյ�ʱ����ȡ��ÿһ�������ſ�ʱ��ĵ���
[gci,goi] = dypsa(s,Fsd);
if isempty(gci) || isempty(goi)
    VFER=[];
    return;
end

if goi(1) > gci(1)
    goi_new = [goi(1:size(goi,2)-1)];
    gci_new = [gci(2:size(gci,2))];
else
    goi_new = goi;
    gci_new = gci;
end
openpoint = [goi_new;gci_new];
M = size(openpoint,2);%M�����ж��ٸ������ſ���ʱ���

for n=1:M
    analysis_region = s(openpoint(1,n) : openpoint(2,n));
    %% Find Hilbert envelopes from the frequency domain
    % Find spectrum of voiced excitations
    fftsize = 2^nextpow2(length(analysis_region));
    analysis_region = analysis_region(:)';
    U = fft(analysis_region,fftsize);
    
    % Band-filter using Hanning windows for bands 0-2kHz; 1-3kHz; and 2-4kHz
    % Only positive frequencies considered. Find Hilbert envelope by filtering in
    % frequency --> taking inverse FFT --> finding
    % absolute value of complex-valued signal
    Fnyq = Fsd/2;
    Bnyq = fftsize/2;
    mult = Bnyq/Fnyq;
    U_HILBT = zeros(16,fftsize);
    index1   = [1:500*mult];
    if length(index1) == 0
        VFER(n) = 0;
    else
        U_PART = zeros(16,fftsize);
        U_PART(1,index1)  = U(index1).*hanning(length(index1))';
        U_HILBT(1,:)= [abs(ifft(U_PART(1,:)))];
        for i = 2 : 16
            index = [(500*(i-1)*mult):(500*i*mult)];
            U_PART(i,index) = U(index).*hanning(length(index))';
            U_HILBT(i,:) = [abs(ifft(U_PART(i,:)))];
        end
        %   index1   = [1:500*mult];
        %   U_PART(:,1)  = U(index1).*hanning(length(index1))';
        %   U_HILBT(:,1) = [abs(ifft(U_PART(:,1)))];
        %   for i = 2 : 23;
        %       index = [(500*(i-1)*mult):(500*i*mult)];
        %       U_PART(:,i) = U(index).*hanning(length(index))';
        %       U_HILBT(:,i) = [abs(ifft(U_PART(:,i)))];
        %   end
        % Determine pairwise cross-correlations between each of the
        % Hilbert envelopes for lags between -0.3ms<lag<0.3ms.
        % Normalize the xcorrs by the sqrt of the product of
        % the energy of each signal to obtain a number 0<n<1.
        % The glottal to noise excitation ratio is the max of
        % the max of each cross-correlation
        %   maxlag = ceil(0.3e-3*Fsd);
        %   xxx = 0;
        %   for i = 1:22
        %       for j = 2:23
        %           xxx = xxx + 1;
        %           U_XCORR(:,xxx) = xcorr(U_HILBT(:,i),U_HILBT(:,j),maxlag) / (sqrt(sum(U_HILBT(:,i).^2)*sum(U_HILBT(:,j).^2)));
        %       end
        %   end
        maxlag = ceil(0.3e-3*Fsd);
        xxx = 0;
        for i = 1:15
            for j = i+1:16
                xxx = xxx + 1;
                U_XCORR(xxx,:) = xcorr(U_HILBT(i,:),U_HILBT(j,:),maxlag) / (sqrt(sum(U_HILBT(i,:).^2)*sum(U_HILBT(j,:).^2)));
            end
        end
        VFER(n) = max(max(U_XCORR));
        
    end
    
end
