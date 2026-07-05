import math
import taichi as ti

# -------------------------------------------------------------
# Black Hole Parameters (0-D fields to allow dynamic modification from CPU/GUI)
# -------------------------------------------------------------
spin = ti.field(dtype=ti.f32, shape=())     # Dimensionless spin parameter a in [0, 0.999]
mass = ti.field(dtype=ti.f32, shape=())     # Mass M of the black hole (typically scaled to 1.0)
r_hor = ti.field(dtype=ti.f32, shape=())    # Event horizon radius r+
r_isco = ti.field(dtype=ti.f32, shape=())   # Innermost Stable Circular Orbit (ISCO) radius

# Test fields for physics validation
test_out = ti.field(dtype=ti.f32, shape=2)   # [0] = captured, [1] = max Hamiltonian drift

# -------------------------------------------------------------
# Particle System & Ray Experiments (Phase 3)
# -------------------------------------------------------------
num_particles = 1500
particle_pos = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)   # BL coordinates (r, th, ph)
particle_speed = ti.field(dtype=ti.f32, shape=num_particles)            # Orbital velocity (dphi/dt)
particle_color = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)  # Particle RGB color

# Forward Ray experiments
ray_path = ti.Vector.field(3, dtype=ti.f32, shape=500)   # Boyer-Lindquist coordinates of path
ray_path_len = ti.field(dtype=ti.i32, shape=())          # Current active length of path

def init_physics_parameters():
    spin[None] = 0.0   # Schwarzschild by default
    mass[None] = 1.0
    r_hor[None] = 2.0  # Schwarzschild horizon is at 2M
    r_isco[None] = 6.0 # Schwarzschild ISCO is at 6M
    ray_path_len[None] = 0

def update_parameters(a: float, m: float = 1.0):
    a = min(max(a, 0.0), 0.999)
    spin[None] = a
    mass[None] = m
    # Horizon r+ = M + sqrt(M^2 - a^2)
    r_hor[None] = m + math.sqrt(max(m * m - a * a, 0.0))
    # Bardeen ISCO for prograde orbit
    z1 = 1.0 + (1.0 - a * a) ** (1 / 3) * ((1.0 + a) ** (1 / 3) + (1.0 - a) ** (1 / 3))
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    r_isco[None] = m * (3.0 + z2 - math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2)))

# -------------------------------------------------------------
# Planckian locus approximation (helper for particle colors)
# -------------------------------------------------------------
@ti.func
def blackbody_rgb(T: ti.f32):
    t = ti.math.clamp(T, 1000.0, 40000.0) / 100.0
    r, g, b = 1.0, 1.0, 1.0
    if t <= 66.0:
        g = ti.math.clamp((99.4708025861 * ti.log(t) - 161.1195681661) / 255.0, 0.0, 1.0)
        b = 0.0
        if t > 19.0:
            b = ti.math.clamp((138.5177312231 * ti.log(t - 10.0) - 305.0447927307) / 255.0, 0.0, 1.0)
    else:
        r = ti.math.clamp(329.698727446 * (t - 60.0) ** -0.1332047592 / 255.0, 0.0, 1.0)
        g = ti.math.clamp(288.1221695283 * (t - 60.0) ** -0.0755148492 / 255.0, 0.0, 1.0)
    return ti.Vector([r, g, b])

# -------------------------------------------------------------
# Particle Initialization & Physics Updates
# -------------------------------------------------------------
@ti.kernel
def init_particles():
    # Spawns particles in 3D orbit around the black hole
    for i in particle_pos:
        r = ti.random() * 14.0 + 5.5
        ph = ti.random() * 2.0 * math.pi
        
        # Tilt particle orbits slightly to create a thick accretion dust halo
        th = (math.pi / 2.0) + (ti.random() - 0.5) * 0.12
        
        particle_pos[i] = ti.Vector([r, th, ph])
        
        # Keplerian orbital angular velocity
        om = 1.0 / (r ** 1.5 + spin[None])
        particle_speed[i] = om
        
        # Map color to temperature: hot blue close, warm orange far
        temp = 9000.0 * (r / 6.0) ** -0.75
        particle_color[i] = blackbody_rgb(temp)

@ti.kernel
def update_particles(dt: ti.f32):
    a = spin[None]
    for i in particle_pos:
        r = particle_pos[i].x
        th = particle_pos[i].y
        ph = particle_pos[i].z
        
        # Angular velocity
        om = 1.0 / (r ** 1.5 + a)
        
        # Orbit precession and frame-dragging effects
        ph += om * dt
        if ph > 2.0 * math.pi:
            ph -= 2.0 * math.pi
            
        particle_pos[i].z = ph

# -------------------------------------------------------------
# Geodesics in Kerr Spacetime (Boyer-Lindquist coordinates)
# State vector: y = [r, theta, phi, p_r, p_theta]
# -------------------------------------------------------------
@ti.func
def geodesic_rhs(y, E: ti.f32, L: ti.f32, a: ti.f32):
    r, th, pr, pth = y[0], y[1], y[3], y[4]
    
    # Avoid coordinate singularities near poles and division by zero
    s = ti.sin(th)
    c = ti.cos(th)
    s = ti.max(ti.abs(s), 1e-5) * (1.0 if s >= 0 else -1.0)
    
    # Kerr metric functions
    sigma = r * r + a * a * c * c
    delta = r * r - 2.0 * r + a * a
    
    P = E * (r * r + a * a) - a * L
    Wt = L / s - a * E * s

    # Hamiltonian gauge (H = 0 for photon)
    F = delta * pr * pr + pth * pth - P * P / delta + Wt * Wt
    Hc = F / (2.0 * sigma)

    # Hamilton's equations for position coordinates
    dr = delta * pr / sigma
    dth = pth / sigma
    dph = (a * P / delta + Wt / s) / sigma

    # Hamilton's equations for conjugate momenta coordinates
    ddel = 2.0 * r - 2.0
    Fr = ddel * pr * pr - (2.0 * P * (2.0 * E * r) * delta - P * P * ddel) / (delta * delta)
    Fth = 2.0 * Wt * (-L * c / (s * s) - a * E * c)
    
    dpr = -(Fr - 2.0 * Hc * (2.0 * r)) / (2.0 * sigma)
    dpth = -(Fth - 2.0 * Hc * (-2.0 * a * a * s * c)) / (2.0 * sigma)
    
    return ti.Vector([dr, dth, dph, dpr, dpth])

@ti.func
def rk4_step(y, h: ti.f32, E: ti.f32, L: ti.f32, a: ti.f32):
    k1 = geodesic_rhs(y, E, L, a)
    k2 = geodesic_rhs(y + 0.5 * h * k1, E, L, a)
    k3 = geodesic_rhs(y + 0.5 * h * k2, E, L, a)
    k4 = geodesic_rhs(y + h * k3, E, L, a)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

# -------------------------------------------------------------
# ZAMO Orthonormal Tetrad Camera Projection
# Converts pixel look vectors (nr, nth, nph) to constants (E, L)
# and initial momenta (p_r, p_th).
# -------------------------------------------------------------
@ti.func
def camera_ray(nr: ti.f32, nth: ti.f32, nph: ti.f32, r: ti.f32, th: ti.f32, a: ti.f32):
    s = ti.sin(th)
    c = ti.cos(th)
    s = ti.max(ti.abs(s), 1e-5) * (1.0 if s >= 0 else -1.0)
    sigma = r * r + a * a * c * c
    delta = r * r - 2.0 * r + a * a
    A2 = (r * r + a * a) ** 2 - a * a * delta * s * s
    g_tt = -(1.0 - 2.0 * r / sigma)
    g_tp = -2.0 * a * r * s * s / sigma
    g_pp = A2 * s * s / sigma
    omega = 2.0 * a * r / A2                       # Frame-dragging rate
    alpha = ti.sqrt(delta * sigma / A2)            # Lapse

    pt = 1.0 / alpha
    pp = omega / alpha + nph / ti.sqrt(g_pp)
    E = -(g_tt * pt + g_tp * pp)
    L = g_tp * pt + g_pp * pp
    p_r = nr * ti.sqrt(sigma / delta)
    p_th = nth * ti.sqrt(sigma)
    return E, L, p_r, p_th

# -------------------------------------------------------------
# Ray Experiment Kernel (Phase 3 Sandbox)
# -------------------------------------------------------------
@ti.kernel
def trace_launched_ray(rc: ti.f32, thc: ti.f32, phc: ti.f32, lth: ti.f32, lph: ti.f32, a: ti.f32):
    # Setup ray trajectory starting from the camera look vector
    n = ti.Vector([ti.cos(lth), ti.sin(lth) * ti.cos(lph), ti.sin(lth) * ti.sin(lph)])
    
    # We trace FORWARD into the scene
    E, L, pr0, pth0 = camera_ray(n.x, n.y, n.z, rc, thc, a)
    y = ti.Vector([rc, thc, phc, pr0, pth0])
    
    rstop = r_hor[None] + 0.02
    step = 0
    ray_path_len[None] = 0
    
    while step < 500:
        ray_path[step] = ti.Vector([y[0], y[1], y[2]])
        ray_path_len[None] += 1
        
        # Adaptive integration steps
        hs = 0.07 * ti.math.clamp(y[0] * 0.25, 0.04, 2.2)
        hs *= ti.math.clamp((y[0] - rstop) * 2.0, 0.05, 1.0)
        yn = rk4_step(y, hs, E, L, a)
        
        # Terminate if swallowed or escaped far
        if yn[0] < rstop or yn[0] > 100.0:
            break
            
        y = yn
        step += 1

# -------------------------------------------------------------
# Self-Testing Geodesic Simulation Kernels
# -------------------------------------------------------------
@ti.kernel
def trace_test_ray(b: ti.f32, a: ti.f32):
    for _one in range(1):
        r0 = 200.0
        nph = b / r0
        n = ti.Vector([-ti.sqrt(1.0 - nph * nph), 0.0, nph])
        E, L, pr0, pth0 = camera_ray(n.x, n.y, n.z, r0, math.pi / 2, a)
        y = ti.Vector([r0, math.pi / 2, 0.0, pr0, pth0])
        rstop = 1.0 + ti.sqrt(max(1.0 - a * a, 0.0)) + 0.02
        captured = 0.0
        hmax = 0.0
        
        for _ in range(40000):
            if y[0] < rstop:
                captured = 1.0
                break
            if y[0] > 250.0:
                break
            hs = 0.05 * ti.math.clamp(y[0] * 0.25, 0.02, 2.0)
            y = rk4_step(y, hs, E, L, a)
            
            # Gauge H (should stay 0 along the geodesic path)
            if y[0] > 3.0:
                s = ti.sin(y[1])
                s = ti.max(ti.abs(s), 1e-5) * (1.0 if s >= 0 else -1.0)
                sg = y[0] * y[0] + a * a * ti.cos(y[1]) ** 2
                dl = y[0] * y[0] - 2.0 * y[0] + a * a
                P = E * (y[0] * y[0] + a * a) - a * L
                Wt = L / s - a * E * s
                Hh = (dl * y[3] ** 2 + y[4] ** 2 - P * P / dl + Wt * Wt) / (2.0 * sg)
                hmax = ti.max(hmax, ti.abs(Hh))
                
        test_out[0] = captured
        test_out[1] = hmax

def isco_radius(a: float) -> float:
    z1 = 1.0 + (1.0 - a * a) ** (1 / 3) * ((1.0 + a) ** (1 / 3) + (1.0 - a) ** (1 / 3))
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    return 3.0 + z2 - math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))

def run_physics_checks():
    # Headless metric validity tests
    ok = True
    print("\n[PHYSICS SELF-TEST] Starting general relativity metric checks...")
    
    # 1. Schwarzschild critical impact parameter test (b_crit = 3*sqrt(3) ~= 5.196)
    for b, expect in [(4.9, 1.0), (5.5, 0.0)]:
        trace_test_ray(b, 0.0)
        got, hdrift = test_out[0], test_out[1]
        status = "PASS" if got == expect else "FAIL"
        ok &= (got == expect)
        print(f"[{status}] a=0 ray, b={b}M -> {'captured' if got else 'escaped'} "
              f"(expected {'captured' if expect else 'escaped'}), max|H| drift {hdrift:.2e}")
        ok &= (hdrift < 5e-3)
        
    # 2. Spinning Kerr hole asymmetry tests (a=0.95)
    # Dragging widens retrograde capture, saves prograde
    trace_test_ray(-6.3, 0.95)
    ret_got = test_out[0]
    ok &= (ret_got == 1.0)
    print(f"[{'PASS' if ret_got == 1.0 else 'FAIL'}] a=0.95 retrograde b=-6.3M captured (dragging widens retrograde shadow)")
    
    trace_test_ray(3.2, 0.95)
    pro_got = test_out[0]
    ok &= (pro_got == 0.0)
    print(f"[{'PASS' if pro_got == 0.0 else 'FAIL'}] a=0.95 prograde b=+3.2M escaped (dragging saves prograde ray)")
    
    print(f"ISCO check: a=0 -> {isco_radius(0.0):.4f}M (expect 6.0), a=0.998 -> {isco_radius(0.998):.4f}M (expect ~1.237)")
    
    if ok:
        print("[PHYSICS SELF-TEST] All checks successfully PASSED.\n")
    else:
        print("[PHYSICS SELF-TEST] ERROR: Some checks FAILED.\n")
    return ok
