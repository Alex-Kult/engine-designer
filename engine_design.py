# --- LIBRARIES ---
import math
import isentropic as isn
import l_star as cmb
import MOC_ref as moc

sigma = 0.0236 # surface tension [N/m]
nu_L = 1.402e-6 # kinematic viscosity [m^2/s]
c_pL = 2570 # specific heat at constant pressure on liquid phase [J/(kg*K)]
Q = 841719.12 # vaporization enthalpy [J/kg]
H = 28.2e6 # fuel calorific value [J/kg]
rho_L = 788.3 # density of liquid phase [kg/m^3]
T_s = 351.5 # boiling temperature [K]

# --- INJECTION PARAMETERS ---
of_ratio = 1.22 # oxidizer to fuel ratio --------------------------------- REVISIT WITH CEA
T_0 = 300 # initial temperature (depends on ignitition method) [K]
method = "impinging" # impinging or swirling
# Fill out if method = "impinging"
jets_f = 12 # number of fuel jets
C_d = 0.83 # discharge coefficient
theta = 40 # angle jets make with the ceterline (axial down engine) [deg]

# --- COMBUSTION PARAMETERS ---
n = 4 # integer that modifies the form of the reaction rate (n = 4 for ethanol)
T_c =  3479.61 # adiabatic flame temperature [K] ------------------------- REVISIT WITH CEA
T_g = 3300 # gas stream temperature [K] ---------------------------------- REVISIT WITH CEA
M = 0.01934 # molecular weight of the combustion gas products [kg/mol] --- REVISIT WITH CEA
gamma = 1.1442 # ratio of specific heats --------------------------------- REVISIT WITH CEA
c_p = 6496.9 # specific heat at constant pressure of combustion products - REVISIT WITH CEA
r = 2.087 # weight of oxygen required for combustion of unit weight fuel - REVISIT WITH CEA
mu = 1.0578e-4 # dynamic gas viscosity [Pa*s] ---------------------------- REVISIT WITH CEA
k = 1.60624 # average gas conductivity ----------------------------------- REVISIT WITH CEA

# --- CHAMBER PARAMETERS ---
c_ratio = 6 # contraction ratio ----------------------------------------- REVISIT
b = 30 # contraction angle [deg]
P_c = 400 # chamber pressure [psi]
L_star_fos = 1.25 # L* factor of safety

# --- ENGINE PARAMETERS
thrust = 800 # [lbs]

#################################### END INPUTS #########################################
# --- CONSTANTS & CONVERSIONS ---
R_0 = 8.31446 # universal gas constant [J/(mol*K)]
P_atm = 101325 # [Pa]
g_0 = 9.80665 # [m/s^2]

psi_to_pa = 6894.75729317 # psi to Pa
lbf_to_n = 4.44822162 # lbs to N

P_c = P_c * psi_to_pa
thrust = thrust * lbf_to_n
b = math.radians(b)

# --- ISENTROPIC RELATIONS ---
R = R_0/M # gas constant of combustion products
P_e = P_atm

mach_e = isn.press_ratio(gamma, P_e, P_c)

area_ratio = isn.a_ratio(gamma, mach_e)

T_e = isn.temp_ratio(gamma, mach_e, T_c)
a_e = math.sqrt(gamma*R*T_e)
v_e = mach_e*a_e
m_dot_tot = thrust/v_e

m_dot_o = m_dot_tot/(1 + of_ratio)*of_ratio # oxygen mass flow rate [kg/s]
m_dot_f = m_dot_tot/(1 + of_ratio) # ethanol mass flow rate [kg/s]

area_t = m_dot_tot/(P_c*math.sqrt(gamma/(R*T_c))*(2/(gamma + 1))**((gamma + 1)/(2*(gamma - 1)))) # area of throat [m^2]
area_e = area_t*area_ratio

Rt = math.sqrt(area_t/math.pi)
Re = math.sqrt(area_e/math.pi)
Rn = 0.382*Rt # Based on Georgia Tech

# --- L* ---
area_c = c_ratio * area_t # chamber area [m^2]

L_star, SMD, imp_dist = cmb.solve_chamber(c_ratio, area_c, r, P_c, rho_L, R, gamma, T_c, T_0, method, sigma, nu_L, m_dot_f, m_dot_o, C_d, jets_f, theta, c_p, mu, k, H, Q, c_pL, T_g, T_s, n)

L_star = L_star*L_star_fos
vol_cmb = L_star*area_t # volume of chamber

R2 = 2*Rt
R1 = 1.5*Rt
Rc = math.sqrt(area_c/(math.pi))

line_width = (Rc - Rt - (R1 + R2) * (1 - math.cos(b)))/math.tan(b)
L_conv = (R1 + R2) * math.sin(b) + line_width # length of converging section

vol_conv = math.pi/3 * L_conv * (Rc**2 + Rc*Rt + Rt**2) # volume of truncated cone representing converging section

vol_cyl = vol_cmb - vol_conv
L_cyl = vol_cyl / area_c + imp_dist
L_c = L_cyl + L_conv

if line_width < 0:
    print("Error: Converging geometry not solvable")
    
# --- Method of Characteristics ---
theta_max, area_ratio_moc, Re_moc, Le_moc = moc.moc_geometry(gamma, mach_e, Rt)

# --- ISP ---
isp = thrust / (m_dot_tot*g_0)

# --- OUTPUT ---
print("--- ENGINE GEOMETRY ---")
print("\nINPUTS:")
print(f"\tChamber Pressure: {P_c/psi_to_pa:.6} [psi]")
print(f"\tChamber Temperature: {T_c} [K]")
print(f"\tThrust: {thrust/lbf_to_n:.6} [lbf]")
print(f"\tO/F: {of_ratio}")
print(f"\tContraction Ratio: {c_ratio}")
print(f"\tGamma: {gamma}")
print(f"\tIsp: {isp:.6} [s]")

print("\nINJECTION PARAMETERS:")
print(f"For {jets_f} fuel orifices")
print(f"\tTotal Mass Flow Rate: {m_dot_tot:.6} [kg/s]")
print(f"\tOxidizer Mass Flow Rate: {m_dot_o:.6} [kg/s]")
print(f"\tFuel Mass Flow Rate: {m_dot_f:.6} [kg/s]")
print(f"\tSMD = {SMD*1e6:.4f} [microns]")

print("\nCHAMBER PARAMETERS:")
print(f"\tL*: {L_star*1000:.6} [mm]")
print(f"\tL_cyl: {L_cyl*1000:.6} [mm]")
print(f"\tL_c: {L_c*1000:.6} [mm]")
print(f"\tR1: {R1*1000:.6} [mm]")
print(f"\tR2: {R2*1000:.6} [mm]")
print(f"\tb: {math.degrees(b):.6} [deg]")
print(f"\tRt: {Rt*1000:.6} [mm]")
print(f"\tRc: {Rc*1000:.6} [mm]")

print("\nNOZZLE PARAMETERS:")
print(f"\tRn: {Rn*1000:.6} [mm]")
print(f"\tExit Mach: {mach_e:.6}")
print(f"\tTn: {math.degrees(theta_max):.6} [deg]")
print("\n\tIsentropic:")
print(f"\t\tArea Ratio: {area_ratio:.6}")
print(f"\t\tRe: {Re*1000:.6} [mm]")
print("\n\tMethod of Characteristics:")
print(f"\t\tArea Ratio: {area_ratio_moc:.6}")
print(f"\t\tRe: {Re_moc*1000:.6} [mm]")
print(f"\t\tLe: {Le_moc*1000:.6} [mm]")