% Perturb quad mesh
% specific to 2D box

% inputs
fileIN = 'box_20x20.gri';
fileOUT = 'box_20x20_perturbed.gri';
q = 2; % hard-code for now
dx = 10/20; dy = dx;
dd = [dx dy];
fac = 0.1;

% read file
datacell = importfile(fileIN);

% # nodes per element
n = (q+1)^2;

% get mesh info
data = str2num(datacell{1});
nn = data(1); % number of nodes
ne = data(2); % number of elements
dim = data(3);

coord = zeros(nn,2);
elems2nodes = zeros(ne,n);

% extract node info
for i = 1:nn
    coord(i,:) = str2num(datacell{i+1});
end

% randomize node Ordering
idxnew = randperm(nn);

% extract element info
idx = length(datacell)-ne+1;
j = 1;
for i = idx:length(datacell)
    elems2nodes(j,:) = str2num(datacell{i});
    j = j+1;
end

% create "perturbed" array
% if pert(i) = 1, then ith node (in coord(i,:)) has been perturbed
pert = zeros(nn,1);

% determine elements adjacent to a boundary
% if bnode(i) = 1, then ith node (in coord(i,:)) is on boundary
bnode = zeros(nn,1);
idx = nn+4; % skip first line, # bgroups, first bgroup info
for i = idx:length(datacell)-ne-1 % up to element info
    data = str2num(datacell{i});
    if isempty(data)
        continue;
    end
    % fill in bnode
    for j = 1:length(data)
        node = data(j);
        bnode(node) = 1;
    end
end

% loop through elements
for i = 1:ne
    nodes = elems2nodes(i,:);
%     % if any nodes are located within Rthresh OR on a boundary, then skip
%     skip = 0;
%     for j = 1:n
%         node = nodes(j);
%         % check if node on boundary
%         if (bnode(node) == 1)
%             skip = 1;
%             break;
%         end
%         % check if located within Rthresh
%         % this assumes cylinder center is located at origin
%         dist = norm(coord(node,:));
%         if (dist < Rthresh) 
%             skip = 1;
%             break;
%         end
%     end
%     if skip
%         continue;
%     end
    % check if on boundary
    if max(bnode(nodes) == 1)
        continue
    end
    % perturb midside nodes
    for k = [2,4,6,8]
        node = nodes(k);
        if pert(node) == 1 || bnode(node) == 1 
            % already perturbed or on boundary
            continue
        else
            pert(node) = 1; % now perturbed
        end
        dpert = fac*(-1+2*rand(1)).*dx;
        if k == 2 || k == 8
            d = 1;
        else 
            d = 2;
        end
        coord(node,d) = coord(node,d) + dpert;
    end
    
    % fill perturbed array
%     pert(nodes) = 1;
        
end

% write back to file
% only need to change node locations
f = fopen(fileOUT,'w');
j = 1;
for i = 1:length(datacell)
    if i == 1
        fprintf(f,'%s\n',datacell{i});
    elseif i >= 2 && i <= nn+1
        % node location
        fprintf(f, '%20.10E %20.10E\n', coord(find(idxnew==j),1), ...
            coord(find(idxnew==j),2));
        j = j+1;
    elseif i >= nn+2 && i <= nn+3
        % BGroup info
        fprintf(f,'%s\n',datacell{i});
    elseif i >= nn+4 && i <= length(datacell)-ne-1
        % BFace nodes
        nodes = str2num(datacell{i});
        if isempty(nodes)
            fprintf(f,'%s\n',datacell{i});
            continue;
        else
            fprintf(f, '%d %d\n', idxnew(nodes(1)), idxnew(nodes(2)));
        end
    elseif i == length(datacell)-ne
        % ElemGroup info
        fprintf(f,'%s\n',datacell{i});
        k = 1;
    else
        fprintf(f, '%d %d %d %d %d %d %d %d %d\n', idxnew(elems2nodes(k,1)), ... 
            idxnew(elems2nodes(k,2)), idxnew(elems2nodes(k,3)), ...
            idxnew(elems2nodes(k,4)), idxnew(elems2nodes(k,5)), ...
            idxnew(elems2nodes(k,6)), idxnew(elems2nodes(k,7)), ...
            idxnew(elems2nodes(k,8)), idxnew(elems2nodes(k,9)));
        k = k+1;
    end
end

fclose(f);