function solve_BAOW_ODE(p,u0,tspa,massesn)

    function BAOW_model(du, u, p, t, masses)

        # allometric parameters
        r_max = r0 .* masses .^ rho
        K = ko .* masses .^ kapp
        m_owl = mo .* masses .^ mu

        # current states
        R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, O = u

        # parameters
        r_max, K, m_owl, GGE, a_owl, h_owl, e_owl = p 

        # universal functional response denominator
        O_denominator = 1 + a_owl*h_owl*sum(du[1:10])

        # Evaluate differential equations
        du[1] =  r_max[1] * R1   * (1 - (R1 + 0.5  * sum(du[:]) - R1 - O)  / K[1])  - a_owl * R1 * O / O_denominator
        du[2] =  r_max[2] * R2   * (1 - (R2 + 0.5  * sum(du[:]) - R2 - O)  / K[2])  - a_owl * R2 * O / O_denominator
        du[3] =  r_max[3] * R3   * (1 - (R3 + 0.5  * sum(du[:]) - R3 - O)  / K[3])  - a_owl * R3 * O / O_denominator
        du[4] =  r_max[4] * R4   * (1 - (R4 + 0.5  * sum(du[:]) - R4 - O)  / K[4])  - a_owl * R4 * O / O_denominator
        du[5] =  r_max[5] * R5   * (1 - (R5 + 0.5  * sum(du[:]) - R5 - O)  / K[5])  - a_owl * R5 * O / O_denominator
        du[6] =  r_max[6] * R6   * (1 - (R6 + 0.5  * sum(du[:]) - R6 - O)  / K[6])  - a_owl * R6 * O / O_denominator
        du[7] =  r_max[7] * R7   * (1 - (R7 + 0.5  * sum(du[:]) - R7 - O)  / K[7])  - a_owl * R7 * O / O_denominator
        du[8] =  r_max[8] * R8   * (1 - (R8 + 0.5  * sum(du[:]) - R8 - O)  / K[8])  - a_owl * R8 * O / O_denominator
        du[9] =  r_max[9] * R9   * (1 - (R9 + 0.5  * sum(du[:]) - R9 - O)  / K[9])  - a_owl * R9 * O / O_denominator
        du[10] = r_max[10] * R10 * (1 - (R10 + 0.5 * sum(du[:]) - R10 - O) / K[10]) - a_owl * R10 * O / O_denominator

        e_owl = 
        du[11] = e_owl* a_owl * sum(du[1:10]) * O /O_denominator - m_owl * O

        return nothing
    end

    prob = ODEProblem(BAOW_model, u0, tspan, p)
    sol_p = solve(prob, Tsit5(); saveat=0.1)

    return sol_p
end