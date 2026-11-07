function solve_BAOW_ODE(p,u0,tspan,masses)

    # make owl mortality allometric
    m_owl = mo .* masses .^ mu
    GGE = 0.001
    
        
    p = [m_owl, GGE, a_owl, h_owl]

    function BAOW_model(du, u, p, t, masses)

        # allometric parameters
        r_max = r0 .* masses .^ rho
        K = ko .* masses .^ kapp

        # current states
        R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, O = u

        # parameters
        m_owl, GGE, a_owl, h_owl = p 

        # sum up all the prey
        R_all = R1 + R2 + R3 + R4 + R5 + R6 + R7 + R8 + R9 + R10

        # universal functional response denominator
        O_denominator = 1 + a_owl * h_owl * R_all

        # Evaluate differential equations

        # prey equations
        du[1] =  r_max[1] * R1   * (1 - (R1 + 0.5  * (R_all - R1))  / K[1])  - a_owl * R1 * O / O_denominator
        du[2] =  r_max[2] * R2   * (1 - (R2 + 0.5  * (R_all - R2))  / K[2])  - a_owl * R2 * O / O_denominator
        du[3] =  r_max[3] * R3   * (1 - (R3 + 0.5  * (R_all - R3))  / K[3])  - a_owl * R3 * O / O_denominator
        du[4] =  r_max[4] * R4   * (1 - (R4 + 0.5  * (R_all - R4))  / K[4])  - a_owl * R4 * O / O_denominator
        du[5] =  r_max[5] * R5   * (1 - (R5 + 0.5  * (R_all - R5))  / K[5])  - a_owl * R5 * O / O_denominator
        du[6] =  r_max[6] * R6   * (1 - (R6 + 0.5  * (R_all - R6))  / K[6])  - a_owl * R6 * O / O_denominator
        du[7] =  r_max[7] * R7   * (1 - (R7 + 0.5  * (R_all - R7))  / K[7])  - a_owl * R7 * O / O_denominator
        du[8] =  r_max[8] * R8   * (1 - (R8 + 0.5  * (R_all - R8))  / K[8])  - a_owl * R8 * O / O_denominator
        du[9] =  r_max[9] * R9   * (1 - (R9 + 0.5  * (R_all - R9))  / K[9])  - a_owl * R9 * O / O_denominator
        du[10] = r_max[10] * R10 * (1 - (R10 + 0.5 * (R_all - R10)) / K[10]) - a_owl * R10 * O / O_denominator

        # owl equation
        e_owl = GGE *sum(masses .* a_owl .* du[1:10])
        owl_births = e_owl * a_owl * sum(du[1:10]) * O / O_denominator
        owl_deaths = m_owl * O
        du[11] = owl_births - owl_deaths

        return nothing
    end

    prob = ODEProblem(BAOW_model, u0, tspan, p)
    sol_p = solve(prob, Tsit5(); saveat=0.1)

    return sol_p
end