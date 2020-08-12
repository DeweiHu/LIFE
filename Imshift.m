function B = Imshift(A, shiftsize, padval)
%SHIFT Shift the elements of an array.
%
%    SHIFT is an extension of CIRCSHIFT, allowing both fractional SHIFTSIZE 
%    and a padding different from circular. SHIFT is especially useful for 
%    the shift, including sub-pixel shift, of image data.
%
%B = SHIFT(A, SHIFTSIZE) circularly shifts the elements of the array A by 
%    the amount given by the elements of SHIFTSIZE, which must be a vector 
%    of real numbers. A can be a numeric or logical array; B is an array of 
%    the same size as A. The N-th element of SHIFTSIZE specifies the shift 
%    along the N-th dimension of A. If the length of SHIFTSIZE is less than
%    the number of dimensions of A, the other dimensions of A are not shif-
%    ted. If an element of SHIFTSIZE is positive (negative), the elements
%    of A are shifted towards increasing (decreasing) index values. The 
%    absolute value of the N-th element of SHIFTSIZE may not exceed the 
%    size of the N-th dimension of A. If an element of SHIFTSIZE is 
%    fractional, linear interpolation is used to calculate the shifted 
%    array B.
%
%B = SHIFT(A, SHIFTSIZE, PADVAL) shifts the elements of the array A by the 
%    amount given by the elements of SHIFTSIZE, where PADVAL specifies the
%    method for padding the newly emerging elements behind the shifted 
%    elements:      
%      'circular'  - pad with circular repetition of elements (default) 
%      'replicate' - pad by repeating the border elements of A
%      'symmetric' - pad with mirror reflections of the elements of A
%      'nan'       - pad with NaN (not possible for integer data type of A)
%    If a numeric scalar value is assigned to PADVAL, the elements are 
%    padded with that scalar, e.g. for padding with 0). The padding methods 
%    are the same as used with the MATLAB function PADARRAY.  
%
%   Examples: 
%
%   A=(1:6)'*(1:6)
%
%   A =
%
%       1     2     3     4     5     6
%       2     4     6     8    10    12
%       3     6     9    12    15    18
%       4     8    12    16    20    24
%       5    10    15    20    25    30
%       6    12    18    24    30    36
%
%   shift (A,[-2,1],'circular')             alternatively: shift (A,[-2,1])
%
%   ans =
%
%       18     3     6     9    12    15
%       24     4     8    12    16    20
%       30     5    10    15    20    25
%       36     6    12    18    24    30
%        6     1     2     3     4     5
%       12     2     4     6     8    10
%
%   shift (A,[-2,1],'replicate')
%
%   ans =
%
%       3     3     6     9    12    15
%       4     4     8    12    16    20
%       5     5    10    15    20    25
%       6     6    12    18    24    30
%       6     6    12    18    24    30
%       6     6    12    18    24    30
%
%   shift (A,[-2,1],'symmetric')
%
%   ans =
%
%     3     3     6     9    12    15
%     4     4     8    12    16    20
%     5     5    10    15    20    25
%     6     6    12    18    24    30
%     6     6    12    18    24    30
%     5     5    10    15    20    25
%
%   shift (A,[-2,1],'nan')          
%
%   ans =
%
%      NaN     3     6     9    12    15
%      NaN     4     8    12    16    20
%      NaN     5    10    15    20    25
%      NaN     6    12    18    24    30
%      NaN   NaN   NaN   NaN   NaN   NaN
%      NaN   NaN   NaN   NaN   NaN   NaN
%
%   shift (A,[-2,1],0)
%
%   ans =
%
%        0     3     6     9    12    15
%        0     4     8    12    16    20
%        0     5    10    15    20    25
%        0     6    12    18    24    30
%        0     0     0     0     0     0
%        0     0     0     0     0     0
%
%    format bank, shift (A,[1.5,-.5],'circular')
%
%    ans =
%         8.25       13.75       19.25       24.75       30.25       19.25
%         5.25        8.75       12.25       15.75       19.25       12.25
%         2.25        3.75        5.25        6.75        8.25        5.25
%         3.75        6.25        8.75       11.25       13.75        8.75
%         5.25        8.75       12.25       15.75       19.25       12.25
%         6.75       11.25       15.75       20.25       24.75       15.75 
%
%    format bank, shift (A,[1.5,-.5],'nan')
%
%    ans =
%          NaN         NaN         NaN         NaN         NaN         NaN
%          NaN         NaN         NaN         NaN         NaN         NaN
%         2.25        3.75        5.25        6.75        8.25         NaN
%         3.75        6.25        8.75       11.25       13.75         NaN
%         5.25        8.75       12.25       15.75       19.25         NaN
%         6.75       11.25       15.75       20.25       24.75         NaN 
%% Parse and check inputs
if nargin < 2
    error('missing input arguments')
end
if nargin == 2
    padval = 'circular';
end
if nargin > 3
    error('too many input arguments')
end
if ~(isnumeric(A) || islogical(A))
    error('A must be a numeric or logical array')
end
if ~(isvector(shiftsize) && isnumeric(shiftsize) && isreal(shiftsize))
    error('SHIFTSIZE must be a real numeric vector')
end
s = size(A);
dims = length(s);
if length(shiftsize) > dims
    shiftsize = shiftsize(1:dims);
end
if length(shiftsize) < dims
    shiftsize(length(shiftsize)+1:dims) = 0;
end
if any(abs(shiftsize) > s)
    error('Elements of SHIFTSIZE may not exceed the size of A')
end
if any(isnan(shiftsize))
    error('NaN is not allowed as an element of SHIFTSIZE')
end
if strcmp(padval,'nan')
    padval = NaN;
end
if isnumeric(padval)
    pval = padval;
    if ~isscalar(padval)
        error('Numeric values of PADVAL must be scalar.')
    end
    if isnan(pval) && isinteger(A)
        error('Convert A to ''single'' or ''double'' for NaN padding.')
    end
    padval = 'numeric';
end
%% Perform the shift
int = floor(shiftsize);     %vector with integer portions of shiftsize
idx0 = cell(1,dims);        %cell vector for the original index order
for k = 1:dims
    idx0{k} = 1:s(k);
end 
%In the case of integer shiftsize, the algorithm of circshift.m is used
%which allows to perform the whole shift in one go, which is fastest:
if int == shiftsize        
    idx = cell(1,dims);     %cell vector for the index order after shift
    switch padval
        case {'circular','numeric'}
            for k = 1:dims
                m = s(k);
                idx{k} = mod((1:m)-shiftsize(k)-1,m)+1;
            end
        case 'replicate'
            for k = 1:dims
                m = s(k);
                ix = (1:m)-shiftsize(k);
                ix(ix<1) = 1;
                ix(ix>m) = m;
                idx{k} = ix;
            end
        case 'symmetric'
            for k = 1:dims
                m = s(k);
                ix = (1:m)-shiftsize(k);
                ix(ix<1) = 1-ix(ix<1);
                ix(ix>m) = 2*m+1-ix(ix>m);
                idx{k} = ix;
            end
        otherwise
            error(['''',padval,''' is not a valid choice for PADVAL'])
    end
    B = A(idx{:});          %perform shift by indexing into the input array
    
%In the case of a numeric value of padval, replace the wrapped elements:
    if strcmp(padval,'numeric')     
        for k = 1:dims      %replace elements along the k-th dimension
            pidx = idx0;    %cell vector for the indices of the elements 
                            %to be replaced with pval
            m = s(k);
            sh = shiftsize(k);
            if sh > 0       %positive shift -> replace first sh elements
                pidx{k} = 1:sh;
                B(pidx{:}) = pval;
            elseif sh < 0   %negative shift -> replace last sh elements
                pidx{k} = m+sh+1:m;
                B(pidx{:}) = pval;
            else            %no shift -> no replacement
            end           
        end
    end
%In the case of fractional shiftsize, a linear interpolation is required
%which is easier to do one dimension after the other:
else
    fra = shiftsize - int;  %vector with fractional portions of shiftsize
    B = A;
    for d = 1:dims          %shift & interpolate along the d-th dimension:
        shi = int(d);        
        f = fra(d);
        B = (1 - f) * int1Dshift(B,d,shi)...
            + f * int1Dshift(B,d,shi+1);
    end
end
%% Subfunction for integer 1D shift
function outarray = int1Dshift(inarray, dim, sh)
n = s(dim);
ind = idx0;                 %cell vector for the index order after shift
switch padval
    case {'circular','numeric'}
        ind{dim} = mod((1:n)-sh-1,n)+1;
    case 'replicate'
        inx = (1:n)-sh;
        inx(inx<1) = 1;
        inx(inx>n) = n;
        ind{dim} = inx;
    case 'symmetric'
        inx = (1:n)-sh;
        inx(inx<1) = 1-inx(inx<1);
        inx(inx>n) = 2*n+1-inx(inx>n);
        ind{dim} = inx;
    otherwise
        error(['''',padval,''' is not a valid choice for PADVAL'])
end
outarray = inarray(ind{:}); %perform shift by indexing into the input array
%In case of numeric value of padval, replace the wrapped elements:
if strcmp(padval,'numeric')     
    pind = idx0;            %cell vector for the indices of the elements 
                            %to be replaced with pval
    if sh > 0               %positive shift -> replace first sh elements
        pind{dim} = 1:sh;
        outarray(pind{:}) = pval;
    elseif sh < 0           %negative shift -> replace last sh elements
        pind{dim} = n+sh+1:n;
        outarray(pind{:}) = pval;
    else                    %no shift -> no replacement
    end           
end    
end                         %end of the subfunction int1Dshift
end                         %end of the function shift.m