clc, clear, close all

filename = 'E:\\human\\Retina2_Fovea_SNR_101_1.tif';
data = FastTiff(filename);

%% re-arrange
Nf = 5;
dim = size(data);
x_range = [150,150+512];

% [Nf,Bscan,H,W]
v = zeros(Nf,dim(end)/Nf,x_range(2)-x_range(1),dim(2));

for i = 1:dim(end)
    frame = mod(i,Nf)+1;
    bscan = ceil(i/Nf);
    v(frame,bscan,:,:) = data(x_range(1):x_range(2)-1,:,i);
end

%% sub-pixel registration
v_reg = zeros(size(v));
tic
for slc = 1:dim(end)/Nf
    fix = squeeze(v(1,slc,:,:));
    for idx = 1:Nf
        mov = squeeze(v(idx,slc,:,:));
        [output,Greg] = dftregistration(fft2(fix),fft2(mov),100);
        %shiftsize = [output(3),output(4)];
        %v_reg(idx,slc,:,:) = Imshift(mov,shiftsize);
        v_reg(idx,slc,:,:) = abs(ifft2(Greg));
    end
end
toc
%%
v_var = squeeze(var(v,0,1));
reg_var = squeeze(var(v_reg,0,1));

%% var of raw and registered
slc = 250; 
figure(1)
subplot(121),imshow(squeeze(v_var(slc,:,:)),[]),title('raw')
subplot(122),imshow(squeeze(reg_var(slc,:,:)),[]),title('registered')

%% shift and ifft compare
slc = 330;
fix = squeeze(v(1,slc,:,:));
mov = squeeze(v(5,slc,:,:));

[output,Greg] = dftregistration(fft2(fix),fft2(mov),100);
shiftsize = [output(3),output(4)];
Im1 = Imshift(mov,shiftsize);
Im2 = abs(ifft2(Greg));

figure(1)
subplot(121),imshow(Im1,[])
subplot(122),imshow(Im2,[])

%% tiff reading
function data = FastTiff(filename)
    warning('off','all') % Suppress all the tiff warnings
    tstack  = Tiff(filename);
    [I,J] = size(tstack.read());
    K = length(imfinfo(filename));
    data = zeros(I,J,K);
    data(:,:,1)  = tstack.read();
    for n = 2:K
        tstack.nextDirectory()
        data(:,:,n) = tstack.read();
    end
    warning('on','all')
end