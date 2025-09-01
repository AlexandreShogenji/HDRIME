% 📜 HDRIME Optimization source codes (version 1.0)
% 🌐 Website and codes of HDRIME: https://github.com/AlexandreShogenji/HDRIME
 
% 🔗 https://github.com/AlexandreShogenji/HDRIME

% 👥 Zhengjie Cai, Tianyang Chen, Zhennao Cai, Huiling Chen, Sudan Yu

% 📅 Last update: 20250901

% 📧 E-Mail: alexandre_cai@163.com, ctrelly1234@outlook.com,
% cznao@wzu.edu.cn, chenhuiling.jlu@gmail.com,2023020603@wzpt.edu.cn
  
% 📜 After use of code, please users cite the main paper on HDRIME: 
% HDRIME Optimization: Concepts and Performance
% Zhengjie Cai, Tianyang Chen, Zhennao Cai, Huiling Chen, Sudan Yu
% Journal, 2025

%----------------------------------------------------------------------------------------------------------------------------------------------------%

% 📊 You can use and compare with other optimization methods developed recently:
%     - (MGO) 2024: 🔗 https://aliasgharheidari.com/MGO.html
%     - (PLO) 2024: 🔗 https://aliasgharheidari.com/PLO.html
%     - (FATA) 2024: 🔗 https://aliasgharheidari.com/FATA.html
%     - (ECO) 2024: 🔗 https://aliasgharheidari.com/ECO.html
%     - (AO) 2024: 🔗 https://aliasgharheidari.com/AO.html
%     - (PO) 2024: 🔗 https://aliasgharheidari.com/PO.html
%     - (RIME) 2023: 🔗 https://aliasgharheidari.com/RIME.html
%     - (INFO) 2022: 🔗 https://aliasgharheidari.com/INFO.html
%     - (RUN) 2021: 🔗 https://aliasgharheidari.com/RUN.html
%     - (HGS) 2021: 🔗 https://aliasgharheidari.com/HGS.html
%     - (SMA) 2020: 🔗 https://aliasgharheidari.com/SMA.html
%     - (HHO) 2019: 🔗 https://aliasgharheidari.com/HHO.html
%____________________________________________________________________________________________________________________________________________________%
function [Best_rime_rate,Best_rime, Convergence_curve,Time] = bHDRIME(N, MaxFEs, dim, A, trn, vald, TFid, classifierFhd)

    if (nargin<8)
        str = 'knn';
        classifierFhd = Get_Classifiers(str);
    end
    tic;
    % Initialize position
    Best_rime = zeros(1, dim);
    Best_rime_rate = inf; % Change to -inf for maximization problems
    Rimepop=initialization(N,dim,1,0)>0.5;
    Lb=zeros(1,dim);% lower boundary
    Ub=ones(1,dim);% upper boundary
    FEs = 1;
    Convergence_curve = [];
    Rime_rates=zeros(1,N);%Initialize the fitness value
    newRime_rates=zeros(1,N);
    % Parameters for dynamic adaptation
    W = 5; % Initial soft-rime parameter    
    beta_min = 0.2; % Minimum scaling factor for DE mutation
    beta_max = 0.8; % Maximum scaling factor for DE mutation
    
    % 计算初始种群的适应度
    for i = 1:N
        Rime_rates(1,i)=AccSz2(Rimepop(i,:), A,trn,vald,classifierFhd);% Calculate the fitness value for each search agent
        if Rime_rates(1, i) < Best_rime_rate
            Best_rime_rate = Rime_rates(1, i);
            Best_rime = Rimepop(i, :);
        end
    end

    % 主循环
    while FEs <= MaxFEs
        beta = beta_min + (beta_max - beta_min) * (1 - FEs / MaxFEs); % Scaling factor for DE mutation
        RimeFactor = (rand - 0.5) * 2 * cos((pi * FEs / (MaxFEs / 10))) * (1 - round(FEs * W / MaxFEs) / W);
        E = sqrt(FEs / MaxFEs); % Exploration coefficient
        normalized_rime_rates = normr(Rime_rates);
        % Main update loop for each agent
        for i = 1:N
            idxs = randperm(N, 3);
            a = Rimepop(idxs(1), :);
            b = Rimepop(idxs(2), :);
            c = Rimepop(idxs(3), :);  
            mutant = a + beta * (b - c);            
            r1 = rand();
            if r1 < E
                newRimepop(i, :) = Best_rime + RimeFactor * rand(1, dim);                
            else
                for j = 1:dim
                    if rand() < 0.8 % CR:Crossover probability
                        newRimepop(i, j) = mutant(j);
                    else
                        newRimepop(i, j) = Rimepop(i, j);
                    end
                end
            end
            % Apply modified hard-rime puncture mechanism
            for j = 1:dim
                if rand() < normalized_rime_rates(i)                                     
                    newRimepop(i, j) = Best_rime(1, j) + (mutant(j) - Best_rime(1, j)) * tanh(W * rand() * (1 - FEs / MaxFEs));                    
                end 
            end
        end        
        %% Convert to binary
        for i = 1:N
            for j = 1:dim 
                newRimepop(i, j) = transferFun(Rimepop(i, j), newRimepop(i, j), TFid);
            end
        end 
        % DLH       
        dist_Position = squareform(pdist(Rimepop, 'hamming'));   % Calculate Hamming distance
        r1 = randperm(N, N);
        for t = 1:N
            radius = mean(Rimepop(t,:) ~= newRimepop(t,:));
            neighbor(t, :) = (dist_Position(t, :) <= radius); % Neighborhood
            [~, Idx] = find(neighbor(t, :) == 1); 
            random_Idx_neighbor = randi(length(Idx), 1, dim); % Randomly select some neighbors
            for d = 1:dim                    
                X_DLH(t, d) = Rimepop(t, d) + rand() * (Rimepop(Idx(random_Idx_neighbor(d)), d) - Rimepop(r1(t), d)); 
            end
        end
        %% Convert to binary
        for i = 1:N
            for j = 1:dim                
                tmp = X_DLH(i, j);                    
                X_DLH(i, j) = transferFun(Rimepop(i, j), tmp, TFid);
            end
        end

        % Boundary control and assessment of the fitness of new populations
        for i = 1:N
            newRime_rates(1,i)=AccSz2(newRimepop(i,:), A,trn,vald,classifierFhd);
            Fit_DLH(i) = AccSz2(X_DLH(i, :), A, trn, vald, classifierFhd);
            % Greedy selection update
            if Fit_DLH(i) < newRime_rates(1, i)
                selected_rate = Fit_DLH(i);
                selected_pop = X_DLH(i, :);
            else
                selected_rate = newRime_rates(1, i);
                selected_pop = newRimepop(i, :);
            end            
            if selected_rate < Rime_rates(1, i)
                Rime_rates(1, i) = selected_rate;
                Rimepop(i, :) = selected_pop;

                if selected_rate < Best_rime_rate
                    Best_rime_rate = selected_rate;
                    Best_rime = selected_pop;
                end
            end
        end
        % Storing convergence curves
        Convergence_curve(FEs) = Best_rime_rate;
        FEs = FEs + 1;
    end
Time = toc;
end
