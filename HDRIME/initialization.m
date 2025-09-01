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
function [pop, pop_fitness, FEs] = initialization(N, dim, ub, lb, fobj, FEs)
    % Step 1: 生成一个均匀随机种群 PR
    PR = repmat(lb, N, 1) + rand(N, dim) .* repmat(ub - lb, N, 1); 
    % 计算PR的适应度，找出最大与最小适应度值
    fitness_values_PR = zeros(N, 1);
    for i = 1:N
        fitness_values_PR(i) = fobj(PR(i, :));
        FEs = FEs + 1;
    end
%     [~, sorted_indices_PR] = sort(fitness_values_PR, 1);
    % 对适应度值进行排序，并直接获取最小值和最大值
    [sorted_fitness_values_PR, sorted_indices_PR] = sort(fitness_values_PR, 1);
    fmin = sorted_fitness_values_PR(1); % 最小值是排序后的第一个元素
    fmax = sorted_fitness_values_PR(end); % 最大值是排序后的最后一个元素
    Weight = [];
    a = 0.1;
    b = 0.9;
    for i = 1:N
        Weight(i) = a + (b-a)*(fmax-fitness_values_PR(i))/(fmax-fmin);
    end

    % 计算聚类的数量 k
    k = round(sqrt(N));    
    % Step 2: 对 PR 进行聚类以获得中心 Z_R
%     [~, ZR] = kmeans(PR, k);  
    % 自定义加权 k-means
    max_iter = 100; % 最大迭代次数
    ZR = PR(randperm(N, k), :); % 随机初始化聚类中心
    for iter = 1:max_iter
        % 计算加权距离
        distances = pdist2(PR, ZR, 'squaredeuclidean'); % 计算平方欧氏距离
        weighted_distances = distances .* Weight'; % 加入权重
        
        % 分配样本点到最近的聚类中心
        [~, labels] = min(weighted_distances, [], 2);
        
        % 更新聚类中心
        new_ZR = zeros(k, size(PR, 2));
        for j = 1:k
            cluster_points = PR(labels == j, :);
            if ~isempty(cluster_points)
                new_ZR(j, :) = mean(cluster_points, 1); % 更新中心
            end
        end
        
        % 判断是否收敛
        if norm(new_ZR - ZR, 'fro') < 1e-5
            break;
        end
        ZR = new_ZR;
    end
    % 初始化空集合 PC
    PC = [];
    % 备份ZR表
    ZT = ZR;
    
    % Step 3: 为剩余的 N - k 个个体生成新个体
    for j = 1:(N - k)
        % 随机选择一个中心 ZP
        center_index = randi(k);  % 记录 ZP 的索引
        ZP = ZR(center_index, :); % 选择中心点 ZP
        
        % 使用柯西变异生成新个体 vj
        vj = ZP + cauchyrnd(0, 1, [1, dim]);
        PC = [PC; vj];
        
        % 计算适应度
        vj_rate = fobj(vj);
        FEs = FEs + 1;
        ZP_rate = fobj(ZP);
        FEs = FEs + 1;

        % 如果 vj 的适应度更好，则替换 ZR 中对应的 ZP 为 vj
        if vj_rate < ZP_rate
            ZR(center_index, :) = vj;  % 更新中心点 ZR表
        end
    end
    
    % Step 10: 从 为未改过的Z_R表 ∪ 新个体表PC 中选择 N/2 个最好的个体作为初始种群的一部分
    pop_candidates = [ZT; PC];
    fitness_values = zeros(N, 1);
    for i = 1:N
        fitness_values(i) = fobj(pop_candidates(i, :));
        FEs = FEs + 1;
    end
    
    [~, sorted_indices] = sort(fitness_values, 1);
    pop = pop_candidates(sorted_indices(1:N/2), :);
    
    % Step 10: 从 PR 中选择另外 N/2 个最好的个体作为初始种群的另一部分
    pop = [pop; PR(sorted_indices_PR(1:N/2), :)];
    
    % 合并两部分个体的适应度值，确保返回行向量
    pop_fitness = [fitness_values(sorted_indices(1:N/2)); fitness_values_PR(sorted_indices_PR(1:N/2))]';
end

% 生成柯西分布随机数的函数
function rnd = cauchyrnd(mu, sigma, sz)
    % 生成柯西分布的随机数
    rnd = mu + sigma * tan(pi * (rand(sz) - 0.5));
end