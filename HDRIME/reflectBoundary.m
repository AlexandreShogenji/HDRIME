function x_reflected = reflectBoundary(x, Lb, Ub)
    x_reflected = x;
    for i = 1:length(x)
        if x_reflected(i) < Lb(i)
            x_reflected(i) = Lb(i) + (Lb(i) - x_reflected(i));
        elseif x_reflected(i) > Ub(i)
            x_reflected(i) = Ub(i) - (x_reflected(i) - Ub(i));
        end
    end
end
