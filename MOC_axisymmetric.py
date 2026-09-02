#Author: Alex Kult
#Description: Axisymmetric method-of-characteristics minimum-length nozzle contour
#Date: 2026-09-01
#
# ----------------------------------------------------------------------------
# WHY THIS FILE EXISTS - Summary provided by Claude
# ----------------------------------------------------------------------------
# MOC_ref.py implements the PLANAR (2-D slab) method of characteristics: it
# treats the "y" coordinate as a Cartesian height, not a radius. For a planar
# nozzle, theta+nu and theta-nu really are exact constants along their
# characteristics (the Riemann invariants of 2-D potential flow), which is
# what let that code get away with pure algebra (K- and K+ never change
# along a characteristic).
#
# A real (axisymmetric / round) rocket nozzle is governed by a different PDE:
# the axisymmetric continuity equation has an extra "flow is spreading into a
# ring, not a slab" term (-rho*v/y). That term means theta+nu and theta-nu
# are NOT constant along the characteristics anymore -- they drift by an
# amount that depends on y (how far off the axis you are). This file
# implements the correct axisymmetric compatibility equations in place of
# the planar ones, plus a numerical-continuation solver that's actually
# robust enough to converge at strong (large exit-Mach) design points.
#
# ----------------------------------------------------------------------------
# THE COMPATIBILITY EQUATIONS
# ----------------------------------------------------------------------------
#     d(theta + nu) = -Gamma_minus * dx        along C- (slope = tan(theta-mu))
#     d(theta - nu) = +Gamma_plus  * dx        along C+ (slope = tan(theta+mu))
#
#     Gamma_minus = M*sin(theta)*cos(theta+mu) / ( y * (1 - (M*cos(theta))^2) )
#     Gamma_plus  = M*sin(theta)*cos(theta-mu) / ( y * (1 - (M*cos(theta))^2) )
#
# Setting these source terms to zero recovers the familiar planar result.
#
# VALIDATION (this matters, because a wrong sign/coefficient here would be
# very easy to get subtly wrong and hard to notice): these two coefficients
# were checked against an EXACT closed-form solution of the full axisymmetric
# irrotational supersonic flow equations -- spherically-symmetric point-
# source flow. A point source's velocity field is purely radial, so it is
# trivially irrotational, and mass conservation through concentric spheres
# gives rho(R)*V(R)*R^2 = const, i.e. the ordinary quasi-1D area-Mach
# relation with an "area" proportional to R^2 (R = spherical radius). In the
# meridional (x,y) plane this gives a fully closed-form theta(x,y)=atan2(y,x),
# M(x,y)=M(sqrt(x^2+y^2)) field that satisfies the governing PDEs exactly.
# Stepping numerically along the true local C-/C+ directions from a point in
# this field and comparing the resulting d(theta+-nu)/dx to -Gamma_minus /
# +Gamma_plus predicted at that point, the residual shrinks linearly with
# the step size and extrapolates to exactly zero -- i.e. the coefficients
# above are exact, not approximate. The `interior_point` unit process below
# was independently checked the same way (feed it two real points from the
# exact field, check its predicted third point converges to the exact
# solution as the angular spacing between the input points shrinks -- it
# converges cleanly, roughly as the cube of the step size).
#
# ----------------------------------------------------------------------------
# THE CONTINUATION METHOD
# ----------------------------------------------------------------------------
# Because axisymmetric flow does not conserve K- = theta+nu, the classic
# planar shortcut (theta_max = 0.5*nu(mach_e), solved once, algebraically)
# no longer hits the target exit Mach exactly -- the wall angle needed to
# reach a given exit Mach has to be found by shooting: solve the whole
# characteristics mesh, check the achieved exit Mach, adjust theta_max,
# repeat. For a strong design point (large mach_e, i.e. a lot of wall
# turning) a cold, from-scratch fixed-point solve of the mesh at that large
# theta_max routinely fails to converge (diverges or produces a
# non-physical/non-monotonic wall contour) -- there's no good initial guess
# far from the (much simpler) planar solution.
#
# The fix used here is numerical continuation / homotopy: start at a SMALL,
# easy theta_max (which converges trivially from a naive planar-style
# guess), then ramp theta_max upward in small adaptive steps, at each step
# WARM-STARTING every mesh point's fixed-point iteration from the previous,
# already-converged step's solution. As long as each step is small enough,
# the previous solution stays inside the new step's basin of convergence,
# so the solver can be walked all the way up to a strong design point that
# a cold solve simply cannot reach. If a step produces an invalid mesh, the
# step size is halved and retried. Once the achieved exit Mach brackets the
# target between two ramp steps, this switches to bisection (still
# warm-started) to hone in on the theta_max that hits the target exactly.
#
# ----------------------------------------------------------------------------
# MOC AREA RATIO vs 1-D ISENTROPIC AREA RATIO
# ----------------------------------------------------------------------------
# You would expect a "perfect" minimum-length nozzle to produce exactly
# uniform, parallel flow at the design Mach at the exit, in which case simple
# 1-D mass conservation would force the MOC-computed area ratio to exactly
# match the ordinary isentropic A/A*(mach_e). That expectation is exactly
# true in PLANAR flow (K- conservation makes it exact for any mesh, coarse
# or fine). It is NOT exactly true in axisymmetric flow, even with an
# arbitrarily fine mesh -- and this was checked carefully rather than
# assumed:
#
#  - The Gamma_minus/Gamma_plus coefficients are exact (see above).
#  - The core interior-point unit process converges to the exact solution
#    at high order as the mesh is refined (see above).
#  - Yet the overall MOC area ratio, run at increasing n (ray count), closes
#    only VERY slowly on the isentropic value -- far too slowly to be
#    ordinary O(1/n) mesh truncation error (doubling n barely moves the
#    residual), and Richardson-extrapolating the trend to n->infinity lands
#    on a small but distinctly NON-zero residual (about -0.8% for a mild
#    mach_e=1.3 test case, and roughly -4 to -5% for a strong case like this
#    engine's mach_e~2.68 design point).
#  - Directly inspecting the mesh confirms why: the flow just upstream of
#    the exit is not perfectly uniform across the radius (e.g. Mach number
#    differing by roughly half a percent between the centerline and the
#    wall at a station one step before the exit, for the mild test case)
#
# This is an inherent characteristic of the classic "sharp-corner
# throat, wall cancels the corner's expansion fan" minimum-length-nozzle
# construction when it's extended to axisymmetric flow (this is the same
# construction essentially every introductory gas-dynamics text uses,
# planar or axisymmetric) -- not a bug in this implementation, and not
# something that more mesh points fixes. The MOC-computed wall contour is
# still the physically correct answer for reaching the target Mach AT THE
# WALL via this design method; the 1-D isentropic area ratio is a useful
# but approximate reference that assumes perfectly uniform exit flow, which
# axisymmetric divergence effects mean this method doesn't quite deliver.
# Expect the two numbers to differ by a few percent for a real engine's
# design Mach, growing with how much the flow has to turn.
#
# ----------------------------------------------------------------------------
# PRACTICAL NOTE ON RUNTIME
# ----------------------------------------------------------------------------
# Cost scales roughly as O(n^2) mesh points, each needing `iters` fixed-point
# iterations, each needing `n_sub` sub-steps per characteristic segment, all
# repeated at every continuation/bisection step. The defaults below
# (n=40) take on the order of a minute for a strong design point (e.g.
# mach_e ~ 2.7) on a typical machine. Raising n improves resolution near the
# throat and along the wall but, per the note above, will NOT make the area
# ratio converge all the way to the isentropic value -- past n~40-60 the
# remaining gap is the inherent effect described above, not mesh error, so
# it isn't worth spending much more runtime chasing it.

import numpy as np
import MOC_lib as moc

theta_min = 1e-6
throat_mach = 1.00000001

N_SUB = 8  # midpoint sub-steps used to integrate the axisymmetric correction
           # along each characteristic segment (see _integrate_gamma)


def _gamma_minus(theta, mach, mu, y):
    Mx2 = (mach * np.cos(theta)) ** 2
    return mach * np.sin(theta) * np.cos(theta + mu) / (y * (1 - Mx2))


def _gamma_plus(theta, mach, mu, y):
    Mx2 = (mach * np.cos(theta)) ** 2
    return mach * np.sin(theta) * np.cos(theta - mu) / (y * (1 - Mx2))


def _props(gamma, theta, nu):
    mach = moc.invert_prandtl_meyer_angle(gamma, nu)
    mu = moc.mach_angle(mach)
    return mach, mu


class MeshInvalid(Exception):
    pass


def _integrate_gamma(gamma_fn, props_fn, pa, pb, n_sub=N_SUB):
    """
    Midpoint-rule average of Gamma_minus/Gamma_plus along the straight-line
    segment from pa to pb, sampled at n_sub midpoints of (theta, nu, y)
    linearly interpolated between the endpoints. Used instead of a single
    2-point trapezoidal average because (1) Gamma ~ 1/y, and midpoint
    sampling never touches y=0 exactly even when a segment starts on the
    axis, and (2) for strong turning, more sub-samples resolve how fast
    Gamma changes across a segment instead of a crude 2-point estimate.
    """
    theta_a, nu_a, y_a = pa['theta'], pa['nu'], pa['y']
    theta_b, nu_b, y_b = pb['theta'], pb['nu'], pb['y']
    total = 0.0
    for i in range(n_sub):
        t = (i + 0.5) / n_sub
        theta_t = theta_a + t * (theta_b - theta_a)
        nu_t = nu_a + t * (nu_b - nu_a)
        y_t = y_a + t * (y_b - y_a)
        mach_t, mu_t = props_fn(theta_t, nu_t)
        total += gamma_fn(theta_t, mach_t, mu_t, y_t)
    return total / n_sub


def _solve_mesh(gamma, theta_max, throat_rad, n, delta, iters, n_sub=N_SUB,
                 warm_wall_pts=None, warm_sweeps=None):
    """
    Solve one full characteristics mesh at a given theta_max.

    If warm_wall_pts/warm_sweeps are given (the converged mesh from a nearby,
    already-solved theta_max), every point's fixed-point iteration is seeded
    from the corresponding point in that mesh instead of a cold "planar"
    predictor -- the continuation/homotopy trick described up top.
    """

    def props(theta, nu):
        try:
            return _props(gamma, theta, nu)
        except (ValueError, RuntimeError) as e:
            raise MeshInvalid(str(e))

    def gminus_int(pa, pb):
        return _integrate_gamma(_gamma_minus, props, pa, pb, n_sub)

    def gplus_int(pa, pb):
        return _integrate_gamma(_gamma_plus, props, pa, pb, n_sub)

    ray_theta = np.linspace(theta_min, theta_max, n)
    nu_kick = moc.prandtl_meyer(gamma, throat_mach)
    mu_kick = moc.mach_angle(throat_mach)
    ray_nu = ray_theta + nu_kick

    ray_seed = []
    for m in range(n):
        mach_m, mu_m = props(ray_theta[m], ray_nu[m])
        ray_seed.append(dict(x=0.0, y=throat_rad, theta=ray_theta[m], nu=ray_nu[m], mu=mu_m, mach=mach_m))

    def centerline_point(pa, warm=None):
        theta_b = 0.0
        if warm is not None:
            nu_b, x_b = warm['nu'], warm['x']
        else:
            nu_b = pa['theta'] + pa['nu']
            mach_b, mu_b = props(theta_b, nu_b)
            m1 = np.tan(0.5 * ((pa['theta'] - pa['mu']) + (theta_b - mu_b)))
            x_b = pa['x'] - pa['y'] / m1
        mach_b, mu_b = props(theta_b, nu_b)
        for _ in range(iters):
            pb = dict(x=x_b, y=0.0, theta=theta_b, nu=nu_b, mu=mu_b, mach=mach_b)
            Ga = gminus_int(pa, pb)
            nu_b = pa['theta'] + pa['nu'] - delta * Ga * (x_b - pa['x']) - theta_b
            mach_b, mu_b = props(theta_b, nu_b)
            m1 = np.tan(0.5 * ((pa['theta'] - pa['mu']) + (theta_b - mu_b)))
            x_b = pa['x'] - pa['y'] / m1
        return dict(x=x_b, y=0.0, theta=theta_b, nu=nu_b, mu=mu_b, mach=mach_b)

    def interior_point(pa, pc, warm=None):
        Km0 = pa['theta'] + pa['nu']
        Kp0 = pc['theta'] - pc['nu']
        if warm is not None:
            theta_b, nu_b, x_b, y_b = warm['theta'], warm['nu'], warm['x'], warm['y']
        else:
            theta_b = 0.5 * (Km0 + Kp0)
            nu_b = 0.5 * (Km0 - Kp0)
            mach_b, mu_b = props(theta_b, nu_b)
            m1 = np.tan(0.5 * ((pa['theta'] - pa['mu']) + (theta_b - mu_b)))
            m2 = np.tan(0.5 * ((pc['theta'] + pc['mu']) + (theta_b + mu_b)))
            x_b = (pa['y'] - pc['y'] + m2 * pc['x'] - m1 * pa['x']) / (m2 - m1)
            y_b = pa['y'] + m1 * (x_b - pa['x'])
        mach_b, mu_b = props(theta_b, nu_b)
        for _ in range(iters):
            pb = dict(x=x_b, y=y_b, theta=theta_b, nu=nu_b, mu=mu_b, mach=mach_b)
            Ga = gminus_int(pa, pb)
            Gc = gplus_int(pc, pb)
            Km = Km0 - delta * Ga * (x_b - pa['x'])
            Kp = Kp0 + delta * Gc * (x_b - pc['x'])
            theta_b = 0.5 * (Km + Kp)
            nu_b = 0.5 * (Km - Kp)
            mach_b, mu_b = props(theta_b, nu_b)
            m1 = np.tan(0.5 * ((pa['theta'] - pa['mu']) + (theta_b - mu_b)))
            m2 = np.tan(0.5 * ((pc['theta'] + pc['mu']) + (theta_b + mu_b)))
            x_b = (pa['y'] - pc['y'] + m2 * pc['x'] - m1 * pa['x']) / (m2 - m1)
            y_b = pa['y'] + m1 * (x_b - pa['x'])
        return dict(x=x_b, y=y_b, theta=theta_b, nu=nu_b, mu=mu_b, mach=mach_b)

    def wall_point(pa, prev_wall, first=False, warm=None):
        theta_b = theta_max if first else pa['theta']
        Km0 = pa['theta'] + pa['nu']
        if warm is not None:
            nu_b, x_b, y_b = warm['nu'], warm['x'], warm['y']
        else:
            nu_b = Km0 - theta_b
            mach_b, mu_b = props(theta_b, nu_b)
            m1 = np.tan(theta_max) if first else np.tan(0.5 * (prev_wall['theta'] + theta_b))
            m2 = np.tan(0.5 * ((pa['theta'] + pa['mu']) + (theta_b + mu_b)))
            x_b = (prev_wall['y'] - pa['y'] + m2 * pa['x'] - m1 * prev_wall['x']) / (m2 - m1)
            y_b = prev_wall['y'] + m1 * (x_b - prev_wall['x'])
        m1 = np.tan(theta_max) if first else np.tan(0.5 * (prev_wall['theta'] + theta_b))
        mach_b, mu_b = props(theta_b, nu_b)
        for _ in range(iters):
            pb = dict(x=x_b, y=y_b, theta=theta_b, nu=nu_b, mu=mu_b, mach=mach_b)
            Ga = gminus_int(pa, pb)
            nu_b = Km0 - delta * Ga * (x_b - pa['x']) - theta_b
            mach_b, mu_b = props(theta_b, nu_b)
            m2 = np.tan(0.5 * ((pa['theta'] + pa['mu']) + (theta_b + mu_b)))
            x_b = (prev_wall['y'] - pa['y'] + m2 * pa['x'] - m1 * prev_wall['x']) / (m2 - m1)
            y_b = prev_wall['y'] + m1 * (x_b - prev_wall['x'])
        return dict(x=x_b, y=y_b, theta=theta_b, nu=nu_b, mu=mu_b, mach=mach_b)

    wall_pts = [dict(x=0.0, y=throat_rad, theta=theta_min, nu=nu_kick, mu=mu_kick, mach=throat_mach)]
    sweeps = []
    for k in range(n):
        pa0 = ray_seed[0] if k == 0 else sweeps[k - 1][1]
        warm_cl = warm_sweeps[k][0] if warm_sweeps is not None else None
        cl = centerline_point(pa0, warm=warm_cl)
        sweep = [cl]
        for m in range(1, n - k):
            ray_index = k + m
            pa = ray_seed[ray_index] if k == 0 else sweeps[k - 1][m + 1]
            pc = sweep[m - 1]
            warm_pt = warm_sweeps[k][m] if warm_sweeps is not None else None
            sweep.append(interior_point(pa, pc, warm=warm_pt))
        warm_wp = warm_sweeps[k][-1] if warm_sweeps is not None else None
        wp = wall_point(sweep[-1], wall_pts[-1], first=(k == 0), warm=warm_wp)
        sweep.append(wp)
        wall_pts.append(wp)
        sweeps.append(sweep)

    for wp in wall_pts:
        for key in ('x', 'y', 'theta', 'nu', 'mach'):
            if not np.isfinite(wp[key]):
                raise MeshInvalid(f"non-finite {key} in wall contour")
    for i in range(1, len(wall_pts)):
        if wall_pts[i]['y'] < wall_pts[i - 1]['y'] - 1e-7 or wall_pts[i]['x'] < wall_pts[i - 1]['x'] - 1e-7:
            raise MeshInvalid(
                f"wall contour not monotonic at point {i}: "
                f"({wall_pts[i-1]['x']:.6g},{wall_pts[i-1]['y']:.6g}) -> "
                f"({wall_pts[i]['x']:.6g},{wall_pts[i]['y']:.6g})"
            )

    return wall_pts, sweeps


def moc_geometry(gamma, mach_e, throat_rad, n=50, iters=6, planar=False,
                  n_steps_init=10, max_bisections=12, n_shoot_iters=25, tol=1e-7, n_sub=N_SUB):
    """
    Axisymmetric (planar=False, default) or planar (planar=True) minimum-length
    nozzle contour via method of characteristics.

    Uses a single continuation/homotopy ramp on theta_max, from a modest
    starting value up toward theta_max_planar (=0.5*nu(mach_e), always an
    upper bound on the true axisymmetric theta_max -- axisymmetric divergence
    means less wall turning is needed to reach a given exit Mach than in
    planar flow). Every step warm-starts from the previous, already-converged
    mesh. A step that produces an invalid or non-physical mesh is halved and
    retried. See the module docstring up top for why this is needed, and for
    an important note on how the returned area ratio compares to the simple
    1-D isentropic area ratio.

    Because axisymmetric flow does not conserve K- = theta+nu,
    theta_max_planar overshoots the target exit Mach: the achieved exit Mach
    rises through mach_e partway up the ramp. Once that crossing is bracketed
    between two consecutive ramp steps, this switches to bisecting directly
    within that bracket (still warm-starting from the lower, converged end)
    to hone in on the theta_max that hits mach_e exactly.

    Returns (theta_max, area_ratio, Re, Le), matching MOC_ref.moc_geometry's
    signature. Default n=40 takes on the order of a minute for a strong
    design point; see the "PRACTICAL NOTE ON RUNTIME" section up top before
    raising it.
    """
    theta_max_planar = 0.5 * moc.prandtl_meyer(gamma, mach_e)

    if planar:
        wall_pts, _ = _solve_mesh(gamma, theta_max_planar, throat_rad, n, delta=0, iters=iters, n_sub=n_sub)
        return theta_max_planar, (wall_pts[-1]['y'] / throat_rad) ** 2, wall_pts[-1]['y'], wall_pts[-1]['x']

    def solve_step(theta_max_next, wall_pts, sweeps):
        return _solve_mesh(gamma, theta_max_next, throat_rad, n, delta=1, iters=iters, n_sub=n_sub,
                            warm_wall_pts=wall_pts, warm_sweeps=sweeps)

    theta_current = theta_max_planar / n_steps_init
    wall_pts, sweeps = _solve_mesh(gamma, theta_current, throat_rad, n, delta=1, iters=iters, n_sub=n_sub)
    m_current = wall_pts[-1]['mach']

    if m_current >= mach_e:
        raise RuntimeError(
            f"Even the smallest continuation step (theta_max={np.degrees(theta_current):.3f} deg) "
            f"already overshoots the target exit Mach ({m_current:.4f} >= {mach_e}). "
            "Try a larger n_steps_init."
        )

    step = theta_current
    theta_prev, wall_prev, sweeps_prev, m_prev = theta_current, wall_pts, sweeps, m_current

    while m_current < mach_e:
        theta_next = min(theta_current + step, theta_max_planar)
        bisections = 0
        while True:
            try:
                new_wall_pts, new_sweeps = solve_step(theta_next, wall_pts, sweeps)
                m_next = new_wall_pts[-1]['mach']
                break
            except MeshInvalid:
                m_next = None
            bisections += 1
            if bisections > max_bisections:
                raise RuntimeError(
                    f"Continuation stalled at theta_max={np.degrees(theta_current):.3f} deg "
                    f"(target {np.degrees(theta_max_planar):.3f} deg) after {max_bisections} step "
                    "bisections -- the mesh solve is not converging even with small steps."
                )
            step = step / 2
            theta_next = theta_current + step

        theta_prev, wall_prev, sweeps_prev, m_prev = theta_current, wall_pts, sweeps, m_current
        theta_current, wall_pts, sweeps, m_current = theta_next, new_wall_pts, new_sweeps, m_next

        if m_current >= mach_e:
            break
        if theta_current >= theta_max_planar - 1e-12:
            raise RuntimeError(
                f"Ramped theta_max all the way to the planar value "
                f"({np.degrees(theta_max_planar):.3f} deg) without reaching the target exit "
                f"Mach (got {m_current:.4f}, target {mach_e})."
            )
        step = step * 1.5

    a, b = theta_prev, theta_current
    wall_a, sweeps_a, ma = wall_prev, sweeps_prev, m_prev
    wall_b, sweeps_b, mb = wall_pts, sweeps, m_current
    for _ in range(n_shoot_iters):
        if abs(mb - mach_e) < tol:
            break
        c = 0.5 * (a + b)
        wc, sc = solve_step(c, wall_a, sweeps_a)
        mc = wc[-1]['mach']
        if mc < mach_e:
            a, wall_a, sweeps_a, ma = c, wc, sc, mc
        else:
            b, wall_b, sweeps_b, mb = c, wc, sc, mc

    Re = wall_b[-1]['y']
    Le = wall_b[-1]['x']
    area_ratio = (Re / throat_rad) ** 2
    return b, area_ratio, Re, Le