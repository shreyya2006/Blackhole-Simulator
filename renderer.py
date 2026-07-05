import math
import taichi as ti
import physics

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
W = 1280
H = 800
MAX_STEPS = 360
R_ESCAPE = 60.0
DISK_OUT = 18.0
T_PEAK = 7500.0   # Disk peak temperature in Kelvin
H0 = 0.16         # Ray tracing step size factor

vec2 = ti.math.vec2
vec3 = ti.math.vec3

# -------------------------------------------------------------
# Camera Fields
# -------------------------------------------------------------
cam_r = ti.field(ti.f32, shape=())
cam_th = ti.field(ti.f32, shape=())
cam_ph = ti.field(ti.f32, shape=())

# Camera look direction angles in local ZAMO frame
cam_look_th = ti.field(ti.f32, shape=())
cam_look_ph = ti.field(ti.f32, shape=())

fov = ti.field(ti.f32, shape=())
exposure = ti.field(ti.f32, shape=())
disk_t = ti.field(ti.f32, shape=())

# -------------------------------------------------------------
# Screen Buffer Fields
# -------------------------------------------------------------
col_new = ti.Vector.field(3, ti.f32, shape=(W, H))
col_acc = ti.Vector.field(3, ti.f32, shape=(W, H))
bloom_a = ti.Vector.field(3, ti.f32, shape=(W, H))
img = ti.Vector.field(3, ti.f32, shape=(W, H))

# Render Mode: 0 = Cinematic, 1 = Physics Viz, 2 = View Mode (minimal UI)
render_mode = ti.field(ti.i32, shape=())

def init_renderer_parameters():
    cam_r[None] = 24.0
    cam_th[None] = math.radians(84.0)  # Lower camera for a more dramatic, massive scale
    cam_ph[None] = 0.0
    cam_look_th[None] = math.pi        # Straight at black hole
    cam_look_ph[None] = 0.0
    fov[None] = math.radians(55.0)
    exposure[None] = 0.95              # Cinematic brightness
    disk_t[None] = 0.0
    render_mode[None] = 1              # Default to Physics Viz for learning

# -------------------------------------------------------------
# Procedural Noise & Utilities
# -------------------------------------------------------------
@ti.func
def hash21(p):
    return ti.math.fract(ti.sin(p.dot(vec2(127.1, 311.7))) * 43758.5453)

@ti.func
def hash31(p):
    return ti.math.fract(ti.sin(p.dot(vec3(17.1, 31.7, 61.3))) * 43758.5453)

@ti.func
def vnoise(p):
    i = ti.floor(p)
    f = ti.math.fract(p)
    u = f * f * (3.0 - 2.0 * f)
    a = hash21(i)
    b = hash21(i + vec2(1, 0))
    c = hash21(i + vec2(0, 1))
    d = hash21(i + vec2(1, 1))
    return ti.math.mix(ti.math.mix(a, b, u.x), ti.math.mix(c, d, u.x), u.y)

@ti.func
def fbm(p):
    v = 0.0
    amp = 0.5
    q = p
    for _ in range(4):
        v += amp * vnoise(q)
        q = vec2(0.8 * q.x - 0.6 * q.y, 0.6 * q.x + 0.8 * q.y) * 2.03
        amp *= 0.5
    return v

# -------------------------------------------------------------
# Accretion Disk Physics (Newtonian & Relativistic Profiles)
# -------------------------------------------------------------
@ti.func
def disk_temperature(r: ti.f32, rin: ti.f32) -> ti.f32:
    x = ti.max(r / rin, 1.0001)
    f = x ** -0.75 * (1.0 - ti.sqrt(1.0 / x)) ** 0.25
    return T_PEAK * f / 0.4879

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
    return vec3(r, g, b)

@ti.func
def shade_disk(r: ti.f32, ph: ti.f32, E: ti.f32, L: ti.f32, a: ti.f32,
               rin: ti.f32, t: ti.f32, mode: ti.i32):
    # circular-orbit Keplerian velocity Omega in Kerr metric
    om = 1.0 / (r ** 1.5 + a)
    
    # Equatorial metric components (theta = pi/2)
    g_tt = -(1.0 - 2.0 / r)
    g_tp = -2.0 * a / r
    g_pp = r * r + a * a + 2.0 * a * a / r
    
    ut2 = -(g_tt + 2.0 * om * g_tp + om * om * g_pp)
    col = vec3(0.0)
    
    if ut2 > 1e-6:
        ut = 1.0 / ti.sqrt(ut2)
        
        # Redshift factor (Doppler + Gravitational)
        gfac = ti.min(1.0 / (ut * (E - om * L)), 4.0)
        T_em = disk_temperature(r, rin)
        
        # Sheared spiral noise coordinates (differential rotation)
        ph_sheared = ph - om * t + 3.8 * ti.sin(r * 0.16 - 0.12 * t)
        pco = vec2(r * ti.cos(ph_sheared), r * ti.sin(ph_sheared))
        
        # Domain Warping for high-fidelity plasma turbulence
        warp = vec2(fbm(pco * 1.8), fbm(pco * 1.8 + vec2(1.7, 3.1)))
        tex = 0.18 + 1.45 * fbm(pco * 1.4 + warp * 0.85) ** 2
        
        # Relativistic beaming scale
        inten = gfac ** 4 * (T_em / T_PEAK) ** 4 * tex
        
        if mode != 1:  # Cinematic View
            col = blackbody_rgb(gfac * T_em) * inten * 1.8
        else:          # Physics Viz Mode
            base_col = blackbody_rgb(gfac * T_em) * inten * 0.95
            
            # Map redshift factor to colors (blueshift = cyan, redshift = orange-red)
            shift_col = vec3(0.0)
            if gfac > 1.02:
                shift_col = vec3(0.08, 0.42, 1.0) * (gfac - 1.0) * 2.2
            elif gfac < 0.98:
                shift_col = vec3(1.0, 0.15, 0.03) * (1.0 - gfac) * 2.6
            else:
                shift_col = vec3(0.85, 0.85, 0.85)
                
            col = ti.math.mix(base_col, shift_col, 0.65)
            
            # Draw coordinate grid lines
            r_grid = ti.abs(r - ti.round(r)) < 0.038 and ti.round(r) % 2 == 0
            ph_pos = ph if ph >= 0 else ph + 2.0 * math.pi
            ph_deg = ph_pos * 180.0 / math.pi
            ph_grid = ti.abs(ph_deg - ti.round(ph_deg / 30.0) * 30.0) < 0.8
            
            if r_grid or ph_grid:
                col = ti.math.mix(col, vec3(0.0, 1.0, 0.85), 0.8)
                
            # Draw yellow ISCO orbit circle
            if ti.abs(r - rin) < 0.045:
                col = ti.math.mix(col, vec3(1.0, 0.85, 0.0), 0.9)
                
    return col

# -------------------------------------------------------------
# Sky Background (High-Fidelity Space Environment)
# -------------------------------------------------------------
@ti.func
def sky_color(d):
    # Base dark space gradient
    col = vec3(0.001, 0.0012, 0.0025)
    
    # Coordinates in 2D projected sphere
    n_coord = vec2(d.x + 2.0 * d.z, d.y) * 2.3
    
    # Cinematic colored nebulae gas layers
    neb1 = fbm(n_coord)
    neb2 = fbm(n_coord + vec2(4.8, -1.8))
    dust = fbm(n_coord * 1.8 + vec2(-2.1, 3.2))
    
    # 1. Purple/Magenta gas lanes
    col += vec3(0.016, 0.003, 0.026) * ti.max(neb1 - 0.22, 0.0) * 2.2
    # 2. Cyan/Teal nebula clouds
    col += vec3(0.003, 0.015, 0.02) * ti.max(neb2 - 0.28, 0.0) * 2.4
    # 3. Dense dark-orange dust band
    col += vec3(0.024, 0.01, 0.002) * ti.max(dust - 0.24, 0.0) * 1.6
    
    # Star density variation (star clusters and empty voids)
    star_density = fbm(n_coord * 0.45)
    
    p = d * 64.0
    cell = ti.floor(p)
    rnd = hash31(cell)
    
    threshold = 0.90 + 0.07 * (1.0 - star_density)
    if rnd > threshold:
        sp = cell + 0.5 + 0.6 * (vec3(hash31(cell + 1.3), hash31(cell + 2.7), hash31(cell + 4.1)) - 0.5)
        sd = sp.normalized()
        ang = ti.acos(ti.math.clamp(d.dot(sd), -1.0, 1.0))
        
        # Color temperature tint (hot blue to cool red-yellow)
        tint = ti.math.mix(vec3(0.68, 0.82, 1.0), vec3(1.0, 0.84, 0.65), hash31(cell + 9.9))
        
        # Map star size and intensity
        brightness = (rnd - threshold) / (1.0 - threshold)
        col += tint * brightness * 18.0 * ti.exp(-(ang / 0.0026) ** 2)
        
    return col

# -------------------------------------------------------------
# Ray Tracing Volumetric Core Kernel
# -------------------------------------------------------------
@ti.kernel
def render(t: ti.f32, jx: ti.f32, jy: ti.f32):
    a = physics.spin[None]
    rc, thc, phc = cam_r[None], cam_th[None], cam_ph[None]
    tanf = ti.tan(fov[None] * 0.5)
    rin = physics.r_isco[None]
    rstop = physics.r_hor[None] + 0.02
    mode = render_mode[None]
    
    lth = cam_look_th[None]
    lph = cam_look_ph[None]
    
    # Orthonormal camera frame
    n = vec3(ti.cos(lth), ti.sin(lth) * ti.cos(lph), ti.sin(lth) * ti.sin(lph))
    v = vec3(-ti.sin(lth), ti.cos(lth) * ti.cos(lph), ti.cos(lth) * ti.sin(lph))
    u = vec3(0.0, -ti.sin(lph), ti.cos(lph))
    
    for i, j in col_new:
        scr_u = (i + 0.5 + jx - 0.5 * W) / H
        scr_v = (j + 0.5 + jy - 0.5 * H) / H
        
        rd = (n + scr_u * tanf * u + scr_v * tanf * v).normalized()
        
        # Boyer-Lindquist initial momenta
        E, L, pr0, pth0 = physics.camera_ray(rd.x, rd.y, rd.z, rc, thc, a)
        y = ti.Vector([rc, thc, phc, pr0, pth0])

        accum_color = vec3(0.0)
        transparency = 1.0
        done = 0
        step = 0
        
        while done == 0 and step < MAX_STEPS:
            hs = H0 * ti.math.clamp(y[0] * 0.25, 0.04, 2.2)
            hs *= ti.math.clamp((y[0] - rstop) * 2.0, 0.05, 1.0)
            hs *= ti.math.clamp(ti.abs(ti.sin(y[1])) * 10.0, 0.05, 1.0)
            
            # Step null geodesic path backward
            yn = physics.rk4_step(y, hs, E, L, a)

            # Volumetric Accretion Disk and Plunging Region integration
            r_mid = 0.5 * (y[0] + yn[0])
            th_mid = 0.5 * (y[1] + yn[1])
            ph_mid = 0.5 * (y[2] + yn[2])
            
            if r_mid >= rstop and r_mid <= DISK_OUT:
                cth = ti.cos(th_mid)
                density = 0.0
                emission_col = vec3(0.0)
                
                # 1. Main Accretion Disk (outside ISCO)
                if r_mid >= rin:
                    h_scale = 0.06 # Thickness ratio (z/r)
                    # Gaussian vertical profile, density falls off with radius (r^-1.2)
                    density = ti.exp(-0.5 * (cth / h_scale) ** 2) * (r_mid ** -1.2)
                    emission_col = shade_disk(r_mid, ph_mid, E, L, a, rin, t, mode)
                
                # 2. Plunging Region & Photon Ring (inside ISCO, down to horizon)
                else:
                    h_scale = 0.015 # Thinner, compressed region
                    # High-temperature gas falling towards the horizon
                    density = 0.085 * ti.exp(-0.5 * (cth / h_scale) ** 2) * (r_mid ** -0.5)
                    # Blue-white high-energy blackbody corona emission
                    emission_col = vec3(1.4, 1.75, 2.2) * 1.6 * (r_mid ** -2.0)
                    
                # Single-step absorption and emission accumulation
                dtau = density * 22.0 * hs
                if dtau > 1e-5:
                    step_trans = ti.exp(-dtau)
                    accum_color += transparency * emission_col * (1.0 - step_trans)
                    transparency *= step_trans

            # Lensed 3D Ergosphere Boundary (Physics Viz Mode Only)
            if mode == 1 and a > 0.02:
                re_0 = 1.0 + ti.sqrt(1.0 - a * a * ti.cos(y[1]) ** 2)
                re_n = 1.0 + ti.sqrt(1.0 - a * a * ti.cos(yn[1]) ** 2)
                if (y[0] - re_0) * (yn[0] - re_n) < 0.0:
                    accum_color += transparency * vec3(0.08, 0.015, 0.22)  # Glowing purple

            # Swallowed by event horizon
            if yn[0] < rstop:
                done = 2
                break
                
            # Escaped to space
            if yn[0] > R_ESCAPE and yn[0] > y[0]:
                d = physics.geodesic_rhs(yn, E, L, a)
                sth, cth = ti.sin(yn[1]), ti.cos(yn[1])
                sph, cph = ti.sin(yn[2]), ti.cos(yn[2])
                r1 = yn[0]
                dx = d[0] * sth * cph + r1 * cth * cph * d[1] - r1 * sth * sph * d[2]
                dy = d[0] * sth * sph + r1 * cth * sph * d[1] + r1 * sth * cph * d[2]
                dz = d[0] * cth - r1 * sth * d[1]
                
                bg_col = sky_color(vec3(dx, dy, dz).normalized())
                accum_color += transparency * bg_col
                transparency = 0.0
                done = 3
                break
                
            # Early out when the ray is fully opaque
            if transparency < 0.015:
                done = 1
                break
                
            y = yn
            step += 1
            
        if done == 0:  # Escape due to step limit
            d = physics.geodesic_rhs(y, E, L, a)
            sth, cth = ti.sin(y[1]), ti.cos(y[1])
            sph, cph = ti.sin(y[2]), ti.cos(y[2])
            r1 = y[0]
            dx = d[0] * sth * cph + r1 * cth * cph * d[1] - r1 * sth * sph * d[2]
            dy = d[0] * sth * sph + r1 * cth * sph * d[1] + r1 * sth * cph * d[2]
            dz = d[0] * cth - r1 * sth * d[1]
            bg_col = sky_color(vec3(dx, dy, dz).normalized())
            accum_color += transparency * bg_col
            
        col_new[i, j] = accum_color

# -------------------------------------------------------------
# Point Coordinate Camera Projection & Draw Overlays
# -------------------------------------------------------------
@ti.func
def project_point(r: ti.f32, th: ti.f32, ph: ti.f32, rc: ti.f32, thc: ti.f32, phc: ti.f32,
                  lth: ti.f32, lph: ti.f32, a: ti.f32, fov_val: ti.f32):
    # Convert camera position to quasi-Cartesian
    xc = ti.sqrt(rc * rc + a * a) * ti.sin(thc) * ti.cos(phc)
    yc = ti.sqrt(rc * rc + a * a) * ti.sin(thc) * ti.sin(phc)
    zc = rc * ti.cos(thc)
    
    # Convert point position to quasi-Cartesian
    xp = ti.sqrt(r * r + a * a) * ti.sin(th) * ti.cos(ph)
    yp = ti.sqrt(r * r + a * a) * ti.sin(th) * ti.sin(ph)
    zp = r * ti.cos(th)
    
    # Spherical basis vectors at camera position
    er_x = ti.sin(thc) * ti.cos(phc)
    er_y = ti.sin(thc) * ti.sin(phc)
    er_z = ti.cos(thc)
    
    eth_x = ti.cos(thc) * ti.cos(phc)
    eth_y = ti.cos(thc) * ti.sin(phc)
    eth_z = -ti.sin(thc)
    
    eph_x = -ti.sin(phc)
    eph_y = ti.cos(phc)
    eph_z = 0.0
    
    # Orthonormal camera frame in Cartesian space
    nx = ti.cos(lth) * er_x + ti.sin(lth) * ti.cos(lph) * eth_x + ti.sin(lth) * ti.sin(lph) * eph_x
    ny = ti.cos(lth) * er_y + ti.sin(lth) * ti.cos(lph) * eth_y + ti.sin(lth) * ti.sin(lph) * eph_y
    nz = ti.cos(lth) * er_z + ti.sin(lth) * ti.cos(lph) * eth_z + ti.sin(lth) * ti.sin(lph) * eph_z
    
    vx = -ti.sin(lth) * er_x + ti.cos(lth) * ti.cos(lph) * eth_x + ti.cos(lth) * ti.sin(lph) * eph_x
    vy = -ti.sin(lth) * er_y + ti.cos(lth) * ti.cos(lph) * eth_y + ti.cos(lth) * ti.sin(lph) * eph_y
    vz = -ti.sin(lth) * er_z + ti.cos(lth) * ti.cos(lph) * eth_z + ti.cos(lth) * ti.sin(lph) * eph_z
    
    ux = -ti.sin(lph) * eth_x + ti.cos(lph) * eph_x
    uy = -ti.sin(lph) * eth_y + ti.cos(lph) * eph_y
    uz = -ti.sin(lph) * eth_z + ti.cos(lph) * eph_z
    
    dx = xp - xc
    dy = yp - yc
    dz = zp - zc
    
    z_cam = dx * nx + dy * ny + dz * nz
    x_cam = dx * ux + dy * uy + dz * uz
    y_cam = dx * vx + dy * vy + dz * vz
    
    screen_pos = ti.Vector([-10.0, -10.0])
    if z_cam > 0.05:
        tanf = ti.tan(fov_val * 0.5)
        u_scr = x_cam / (z_cam * tanf)
        v_scr = y_cam / (z_cam * tanf)
        screen_pos = ti.Vector([u_scr, v_scr])
    return screen_pos

@ti.kernel
def draw_particles_overlay(rc: ti.f32, thc: ti.f32, phc: ti.f32,
                           lth: ti.f32, lph: ti.f32, a: ti.f32, fov_val: ti.f32):
    for i in range(physics.num_particles):
        pt = physics.particle_pos[i]
        col = physics.particle_color[i]
        scr_pos = project_point(pt[0], pt[1], pt[2], rc, thc, phc, lth, lph, a, fov_val)
        if scr_pos.x > -9.0:
            px = ti.cast(0.5 * W + 0.5 * H * scr_pos.x, ti.i32)
            py = ti.cast(0.5 * H + 0.5 * H * scr_pos.y, ti.i32)
            if px >= 0 and px < W and py >= 0 and py < H:
                img[px, py] = ti.math.mix(img[px, py], col, 0.72)

@ti.kernel
def draw_path_overlay(rc: ti.f32, thc: ti.f32, phc: ti.f32,
                      lth: ti.f32, lph: ti.f32, a: ti.f32, fov_val: ti.f32):
    num_pts = physics.ray_path_len[None]
    for step in range(num_pts):
        pt = physics.ray_path[step]
        scr_pos = project_point(pt[0], pt[1], pt[2], rc, thc, phc, lth, lph, a, fov_val)
        if scr_pos.x > -9.0:
            px = ti.cast(0.5 * W + 0.5 * H * scr_pos.x, ti.i32)
            py = ti.cast(0.5 * H + 0.5 * H * scr_pos.y, ti.i32)
            
            # Draw line segments (3x3 dots)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    xx = px + dx
                    yy = py + dy
                    if xx >= 0 and xx < W and yy >= 0 and yy < H:
                        img[xx, yy] = ti.math.mix(img[xx, yy], vec3(0.12, 1.0, 0.32), 0.75)  # Glowing green

# -------------------------------------------------------------
# Image Blending and Post Processing
# -------------------------------------------------------------
@ti.kernel
def blend(alpha: ti.f32):
    for i, j in col_acc:
        col_acc[i, j] = ti.math.mix(col_acc[i, j], col_new[i, j], alpha)

@ti.kernel
def bloom_extract_h():
    for i, j in bloom_a:
        acc = vec3(0.0)
        wsum = 0.0
        for k in ti.static(range(-6, 7)):
            x = ti.math.clamp(i + k * 2, 0, W - 1)
            w = ti.exp(-0.5 * (k / 3.0) ** 2)
            c = col_acc[x, j] * exposure[None]
            acc += ti.max(c - 1.0, 0.0) * w
            wsum += w
        bloom_a[i, j] = acc / wsum

@ti.kernel
def bloom_v_compose():
    for i, j in img:
        acc = vec3(0.0)
        wsum = 0.0
        for k in ti.static(range(-6, 7)):
            y = ti.math.clamp(j + k * 2, 0, H - 1)
            w = ti.exp(-0.5 * (k / 3.0) ** 2)
            acc += bloom_a[i, y] * w
            wsum += w
        
        # High contrast cinematic exposure adjustment
        c = col_acc[i, j] * exposure[None] + 0.85 * acc / wsum
        c = ti.pow(c, 1.08)  # Boost contrast slightly
        
        # ACES Filmic Tone Mapping Curve
        m = c * (2.51 * c + 0.03) / (c * (2.43 * c + 0.59) + 0.14)
        
        # Gamma Correction 2.2
        img[i, j] = ti.math.clamp(m, 0.0, 1.0) ** (1.0 / 2.2)
