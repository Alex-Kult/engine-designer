# --- LIBRARIES ---
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simpson
import isentropic as isn

# --- FUNCTIONS ---
def solve_chamber(a_ratio, area_c, r, P_c, rho_L, R_c, gamma, T_c, T_0, method, sigma, nu_L, m_dot_f, m_dot_o, C_d, jets_f, theta, c_p, mu, k, H, Q, c_pL, T_g, T_s, n):
    
    def psi_func(tau, n):
        if tau < 0 or tau >= 1:
            print("Error: Tau out of bounds")
            return 0
        psi = (n + 1) * (1 + 1/n)**n * (1 - tau) * tau**n
        return psi

    def get_derivatives(eta, state):
        """
        Returns [dtau/deta, dchi/deta]
        state = [tau, chi]
        """
        tau, chi = state
        
        # Safety to avoid division by zero during RK4 steps
        if eta <= 0:
            eta = 1e-9
            print("Error: Eta out of bounds (1)")
        if eta >= 1:
            eta = 1 - 1e-9
            print("Error: Eta out of bounds (2)")
        
        eta_poly = eta**2 - 3*eta + 3
        psi = psi_func(tau, n)
        
        # ODE 1: dtau/deta
        dtau_term1 = (1 - eta) / (eta * eta_poly)
        dtau_term2 = (psi / (L * tau)) * chi - 3*(1 - eta) * tau
        dtau = dtau_term1 * dtau_term2
        
        # ODE 2: dchi/deta
        dchi_term1 = S / ((1 - eta) * tau)
        dchi_term2 = eta * eta_poly - chi
        dchi = dchi_term1 * dchi_term2
        
        return np.array([dtau, dchi])

    m_o = r / (r + 1) # weight concentration of oxygen

    delta_P = 0.25*P_c # pressure drop in the injector
    v_i = math.sqrt(2*delta_P/rho_L) # injection velocity

    a_c = math.sqrt(gamma*R_c*T_c) # speed of sound at end of combustion
    mach_c = isn.mach_find(a_ratio, gamma, "subsonic") # mach number at end of combustion
    v_g = mach_c*a_c # gas final velocity in combustion chamber
    
    rho_g = P_c * gamma / a_c**2

    if method == "swirling":
        SMD = 7.3 * sigma**(0.6) * nu_L**(0.2) * m_dot_f**(0.25) * delta_P**(-0.4) # Sauter Mean Diameter

    elif method == "impinging":
        area_tot = m_dot_f/(C_d*math.sqrt(2*rho_L*delta_P))
        area_orifice = area_tot / jets_f

        d_jet = math.sqrt(4 * area_orifice / math.pi)
        imp_dist = 5*d_jet
        
        We = (rho_L * v_i**2 * d_jet) / sigma
        
        theta = theta * (math.pi / 180)

        s = rho_g/rho_L
        SMD = d_jet * (2.62/64**(1/3)) * s**(-1/6) * (We * ((1 - math.cos(theta))**2) / (math.sin(theta))**3)**(-1/3) # From Ryan & Anderson Paper (turbulent mixing from impinging liquid jets)

    r_0 = SMD/2 # initial droplet radius

    # Initial Conditions
    chi_0 = v_i/v_g
    tau_0 = T_0/T_c
    psi_0 = psi_func(tau_0, n)

    # Calculate Constants
    Pr = c_p * mu / k # Prandtl number
    G = (m_dot_f + m_dot_o) / area_c # total propellant mass flux (fuel+ox combined), per Gontijo et al. 2021 (COB-2021-2105) Eq. (2):
    # "G is the propellant mass flux" (not fuel-only) -- confirmed correct, matches Spalding's L* model.
    B = (H*m_o) / (Q*r) + c_pL*(T_g - T_s)/Q # Transfer number
    L = (chi_0*psi_0) / (3*tau_0**2) # Chemical loading
    S = (9*Pr) / (2*math.log(1+B)) # Droplet drag

    # Combustion Check
    L_c = chi_0/3 * (n + 1)/(n - 1) * (1 + 1/n)**n * (1 + 1/(n - 1))**(n-2)

    if L/chi_0 < L_c/chi_0:
        print("Combustion is possible")
    else:
        print("Error: Combustion is not possible")

    # Initial small step for the analytic calculation
    h_init = 1e-4

    # Calculate first step of numerical solution
    dchi_deta_0 = -S * chi_0 / tau_0
    chi_1 = chi_0 + dchi_deta_0 * h_init

    tau_1 = tau_0 + ((S - tau_0) * (1 - tau_0)) / ((n - 2) * (1 - tau_0) - 1) * h_init

    eta_span = (h_init, 0.9999)
    step_1 = [tau_1, chi_1]

    sol = solve_ivp(
        fun=get_derivatives,
        t_span=eta_span,
        y0=step_1,
        method='RK45',      # Adaptive Runge-Kutta 4(5)
        rtol=1e-6,          # Relative tolerance
        atol=1e-8,          # Absolute tolerance
        dense_output=True   # Allow evaluating solution at any point
    )

    # Extract results
    eta_vals = np.concatenate(([0], sol.t))
    tau_vals = np.concatenate(([tau_0], sol.y[0]))
    chi_vals = np.concatenate(([chi_0], sol.y[1]))

    integrand = (chi_vals / tau_vals) * (1 - eta_vals)
    xi_star = simpson(y=integrand, x=eta_vals)

    L_star = (xi_star*a_c*r_0**2 * (2/(gamma + 1) * (1 + (gamma - 1)/2 * (G/(rho_g*a_c))**2))**((gamma + 1)/(2*(gamma - 1)))) / (k/(c_p*rho_L) * math.log(1 + B))
    
    return L_star, SMD, imp_dist
