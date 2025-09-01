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
function [Best_rime, Convergence_curve] = HDRIME(N, MaxFEs, lb, ub, dim, fobj)
    % Initialize position
    Best_rime = zeros(1, dim);
    Best_rime_rate = inf; % Change to -inf for maximization problems
    Lb = lb .* ones(1, dim); % Lower boundary
    Ub = ub .* ones(1, dim); % Upper boundary
    FEs = 0;
    Time = 1;
    neighbor = zeros(N,N);
    Convergence_curve = [];
    W = 5; % Initial soft-rime parameter
    beta_min = 0.2; % Minimum scaling factor for DE mutation
    beta_max = 0.8; % Maximum scaling factor for DE mutation    
    
%% initialization
    [Rimepop, Rime_rates,FEs] = initialization(N, dim, ub, lb, fobj,FEs); 
    % Update the global optimal solution of the initialized population
    for i = 1:N
        if Rime_rates(1, i) < Best_rime_rate
            Best_rime_rate = Rime_rates(1, i);
            Best_rime = Rimepop(i, :);
        end
    end

%% Main loop
    while FEs < MaxFEs
        % Update dynamic parameters
        beta = beta_min + (beta_max - beta_min) * (1 - FEs / MaxFEs); % Scaling factor for DE mutation
        % RimeFactor and E coefficients
        RimeFactor = (rand - 0.5) * 2 * cos((pi * FEs / (MaxFEs / 10))) * (1 - round(FEs * W / MaxFEs) / W);
        E = sqrt(FEs / MaxFEs); 
        newRimepop = Rimepop; % New population container
        % Apply normalization to fitness values for hard-rime mechanism
        normalized_rime_rates = normr(Rime_rates);

        % Main update loop for each agent
        for i = 1:N

            % Select three random indices for DE mutation
            idxs = randperm(N, 3);
            a = Rimepop(idxs(1), :);
            b = Rimepop(idxs(2), :);
            c = Rimepop(idxs(3), :);   
            % Apply DE mutation (DE/rand/1)
            mutant = a + beta * (b - c);

            r1 = rand();
            if r1 < E
                newRimepop(i, :) = Best_rime + RimeFactor * ((Ub - Lb) .* rand(1, dim) + Lb); % Soft-rime mechanism
            else
                for j = 1:dim
                    if rand() < 0.8 % Crossover probability
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
    
%% dimension learning-based hunting
        % Calculates a distance list based on Euclidean distance
        delta    = Rimepop - newRimepop;   
        radius_t = sqrt(sum(delta.^2,2));  
        % The distance between individuals in population
        dist_Position = squareform(pdist(Rimepop)); 
        r1 = randperm(N,N);
        for t = 1:N
            % Divide individuals within the neighborhood
            neighbor(t,:) = (dist_Position(t,:) <= radius_t(t));
            [~,Idx] = find(neighbor(t,:) == 1); % 找到邻居
            random_Idx_neighbor = randi(size(Idx,2),1,dim); 
            for d=1:dim
                % dimension learning-based hunting update
                X_DLH(t,d) = Rimepop(t,d) + rand .* (Rimepop(Idx(random_Idx_neighbor(d)),d) - Rimepop(r1(t),d)); 
            end
            % Boundary Control
            X_DLH(t,:) = reflectBoundary(X_DLH(t, :), Lb, Ub);
            % Evaluate fitness of new population
            Fit_DLH(t) = fobj(X_DLH(t,:)); 
            FEs=FEs+1;
        end
        for i = 1:N
             % Boundary Control
            newRimepop(i,:) = reflectBoundary(newRimepop(i, :), Lb, Ub);                
            % Evaluate fitness of new population
            newRime_rates(1, i) = fobj(newRimepop(i, :));
            FEs = FEs + 1;
            % positive greedy selection mechanism
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
        neighbor = zeros(N,N);
        % Store convergence curve
        Convergence_curve(Time) = Best_rime_rate;
        Time = Time + 1;
    end
end