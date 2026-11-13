function BAOW_model(du, u, p, t)

    # parameters
    d_owl = p.d_owl
    GGE = p.GGE
    a_owl = p.a_owl
    h_owl = p.h_owl 

    # current states
    R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, O = u

    # sum up all the prey
    R_all = R1 + R2 + R3 + R4 + R5 + R6 + R7 + R8 + R9 + R10 + R11 + R12

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
    du[11] = r_max[11] * R11 * (1 - (R11 + 0.5 * (R_all - R11)) / K[11]) - a_owl * R11 * O / O_denominator
    du[12] = r_max[12] * R12 * (1 - (R12 + 0.5 * (R_all - R12)) / K[12]) - a_owl * R12 * O / O_denominator

    # owl equation
    owl_births = (GGE/owl_mass) * a_owl * sum(p.masses .* R_all) * O / O_denominator
    owl_deaths = d_owl * O
    du[13] = owl_births - owl_deaths

    return nothing
end

