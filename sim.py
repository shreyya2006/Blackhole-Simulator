import argparse
import sys
import time
import math
import taichi as ti

# -------------------------------------------------------------
# Command Line Argument Parsing
# -------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true", help="run physics self-tests headless and exit")
parser.add_argument("--quality", choices=["low", "med", "high"], default="med")
args, unknown = parser.parse_known_args()

# -------------------------------------------------------------
# Taichi Initialization (Must happen before module imports)
# -------------------------------------------------------------
try:
    ti.init(arch=ti.gpu)
except Exception:
    ti.init(arch=ti.cpu)
    print("WARNING: GPU backend failed. Running on CPU. Expect lower FPS.")

# -------------------------------------------------------------
# Module Imports
# -------------------------------------------------------------
import physics
import renderer
import ui

# Apply quality settings
if args.quality == "low":
    renderer.W = 800
    renderer.H = 500
    renderer.MAX_STEPS = 200
elif args.quality == "high":
    renderer.W = 1440
    renderer.H = 900
    renderer.MAX_STEPS = 500

# -------------------------------------------------------------
# Main Loop & Interactions
# -------------------------------------------------------------
def print_controls():
    print("""
=============================================================
         INTERACTIVE KERR BLACK HOLE EXPLORATION LAB
=============================================================
CONTROLS:
  [Mouse Drag]
    - In Orbit Mode (default): Drag to orbit the camera.
    - In Explorer Mode: Drag to look around (steer camera).
    
  [Keyboard Navigation]
    - In Orbit Mode:
        W / S  -> Zoom camera closer / further.
    - In Explorer Mode:
        W / S  -> Fly forward / backward along look direction.
        A / D  -> Strafe left / right.
        
  [Simulation Controls]
    - SPACE    -> Play / Pause time advancement for disk.
    - UP / DOWN-> Speed up / Slow down time scaling.
    - R        -> Reset black hole and camera settings.
    - ESC      -> Exit simulator.
=============================================================
""")

def move_camera_free(dr_local, dth_local, dph_local):
    """
    Moves the camera in local ZAMO coordinates.
    Translates local displacements to Boyer-Lindquist coordinate changes.
    """
    r = renderer.cam_r[None]
    th = renderer.cam_th[None]
    ph = renderer.cam_ph[None]
    a = physics.spin[None]
    
    # Avoid mathematical coordinate pole singularities
    th = max(min(th, math.pi - 0.015), 0.015)
    
    s = math.sin(th)
    c = math.cos(th)
    sigma = r * r + a * a * c * c
    delta = r * r - 2.0 * r + a * a
    A2 = (r * r + a * a) ** 2 - a * a * delta * s * s
    
    # Scale coordinates by the Boyer-Lindquist metric coefficients
    scale_r = math.sqrt(delta / sigma)
    scale_th = 1.0 / math.sqrt(sigma)
    scale_ph = math.sqrt(sigma) / (s * math.sqrt(A2))
    
    # Update camera coordinates
    r_new = r + dr_local * scale_r
    th_new = th + dth_local * scale_th
    ph_new = ph + dph_local * scale_ph
    
    # Prevent the camera from falling through the event horizon (add 0.15 safety margin)
    r_min = physics.r_hor[None] + 0.15
    renderer.cam_r[None] = max(r_new, r_min)
    renderer.cam_th[None] = max(min(th_new, math.pi - 0.015), 0.015)
    renderer.cam_ph[None] = ph_new

def main():
    print_controls()
    
    # Initialize values
    physics.init_physics_parameters()
    physics.update_parameters(0.0)  # Start with Schwarzschild
    physics.init_particles()
    renderer.init_renderer_parameters()
    
    try:
        window = ti.ui.Window("Astrophysics Lab: Black Hole Simulator", (renderer.W, renderer.H))
    except Exception as e:
        print("Could not open a GUI window. Vulkan/OpenGL drivers missing? Error:", e)
        sys.exit(1)
        
    canvas = window.get_canvas()
    
    paused = False
    frame = 0
    last_mouse = None
    t0 = time.time()
    last_time = time.time()
    
    while window.running:
        moved = False
        
        # Calculate coordinate delta time
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        # Limit dt to prevent giant jumps during window drags or lags
        dt = min(dt, 0.08)
        
        # 1. Process Event Queue
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == ti.ui.SPACE:
                paused = not paused
            elif e.key in [ti.ui.UP, "Up", "arrow_up", "up"]:
                ui.time_scale = min(ui.time_scale + 0.15, 3.0)
                moved = True
            elif e.key in [ti.ui.DOWN, "Down", "arrow_down", "down"]:
                ui.time_scale = max(ui.time_scale - 0.15, 0.0)
                moved = True
            elif e.key == "r":
                # Reset parameters
                physics.update_parameters(0.0)
                physics.ray_path_len[None] = 0
                physics.init_particles()
                renderer.init_renderer_parameters()
                ui.time_scale = 1.0
                ui.active_camera_type = 0
                ui.journey_stage = 0
                moved = True
                
        # 2. Camera Controls & Navigation
        step_speed = 8.0 * dt  # Frame-rate independent movement speed
        
        # Determine movement input
        move_forward = window.is_pressed("w")
        move_back = window.is_pressed("s")
        move_left = window.is_pressed("a")
        move_right = window.is_pressed("d")
        
        # ZAMO Orbit Camera Mode (locked during Guided Journey transitions)
        if ui.active_camera_type == 0 and ui.journey_stage == 0:
            if move_forward:
                renderer.cam_r[None] = max(physics.r_hor[None] * 2.2, renderer.cam_r[None] * (1.0 - 0.9 * dt))
                moved = True
            if move_back:
                renderer.cam_r[None] = min(55.0, renderer.cam_r[None] * (1.0 + 0.9 * dt))
                moved = True
                
        # Free Explorer Camera Mode
        elif ui.journey_stage == 0:
            if move_forward or move_back or move_left or move_right:
                lth = renderer.cam_look_th[None]
                lph = renderer.cam_look_ph[None]
                
                n_r = math.cos(lth)
                n_th = math.sin(lth) * math.cos(lph)
                n_ph = math.sin(lth) * math.sin(lph)
                
                u_th = -math.sin(lph)
                u_ph = math.cos(lph)
                
                dr_loc, dth_loc, dph_loc = 0.0, 0.0, 0.0
                if move_forward:
                    dr_loc += step_speed * n_r
                    dth_loc += step_speed * n_th
                    dph_loc += step_speed * n_ph
                if move_back:
                    dr_loc -= step_speed * n_r
                    dth_loc -= step_speed * n_th
                    dph_loc -= step_speed * n_ph
                if move_left:
                    dth_loc -= step_speed * u_th
                    dph_loc -= step_speed * u_ph
                if move_right:
                    dth_loc += step_speed * u_th
                    dph_loc += step_speed * u_ph
                    
                move_camera_free(dr_loc, dth_loc, dph_loc)
                moved = True

        # Mouse Orbit / Look Interaction
        if window.is_pressed(ti.ui.LMB):
            mx, my = window.get_cursor_pos()
            if last_mouse is not None:
                dx = mx - last_mouse[0]
                dy = my - last_mouse[1]
                if abs(dx) + abs(dy) > 1e-5:
                    if ui.active_camera_type == 0 and ui.journey_stage == 0:
                        renderer.cam_ph[None] += dx * 3.2
                        new_th = renderer.cam_th[None] - dy * 2.2
                        renderer.cam_th[None] = min(max(new_th, 0.05), math.pi - 0.05)
                    elif ui.journey_stage == 0:
                        renderer.cam_look_ph[None] += dx * 3.2
                        new_lth = renderer.cam_look_th[None] - dy * 2.2
                        renderer.cam_look_th[None] = min(max(new_lth, 0.1), math.pi - 0.1)
                    moved = True
            last_mouse = (mx, my)
        else:
            last_mouse = None

        # 3. Update Guided Journey animations
        if ui.journey_stage > 0:
            ui.update_journey(dt)
            moved = True  # Keep clearing temporal AA accumulation during transitions

        # 4. Advance Time and Particles
        if not paused:
            renderer.disk_t[None] += 0.22 * ui.time_scale
            # Scale DT to match visual velocity
            physics.update_particles(dt * 12.0 * ui.time_scale)

        # 5. Rendering Pipeline
        jx = 0.0
        jy = 0.0
        if not moved:
            jx = (hash(frame * 2654435761 % 1013) % 1000) / 1000.0 - 0.5
            jy = (hash(frame * 40503 % 1013) % 1000) / 1000.0 - 0.5
            
        renderer.render(renderer.disk_t[None], jx * 0.9, jy * 0.9)
        renderer.blend(0.55 if moved else 0.16)
        renderer.bloom_extract_h()
        renderer.bloom_v_compose()
        
        # 6. Apply Overlays on final screen-space pixel buffer
        # Render lensed dust particles
        if ui.learn_mode or renderer.render_mode[None] == 1:
            renderer.draw_particles_overlay(
                renderer.cam_r[None], renderer.cam_th[None], renderer.cam_ph[None],
                renderer.cam_look_th[None], renderer.cam_look_ph[None],
                physics.spin[None], renderer.fov[None]
            )
            
        # Render active ray trajectories
        if physics.ray_path_len[None] > 0:
            renderer.draw_path_overlay(
                renderer.cam_r[None], renderer.cam_th[None], renderer.cam_ph[None],
                renderer.cam_look_th[None], renderer.cam_look_ph[None],
                physics.spin[None], renderer.fov[None]
            )
            
        # Display image
        canvas.set_image(renderer.img)
        
        # 7. Draw HUD (translucent Dear ImGui windows)
        ui.draw_hud(window)
        
        window.show()
        frame += 1
        
        if frame % 120 == 0:
            fps = frame / (time.time() - t0)
            print(f"FPS: {fps:.1f} | Spin a: {physics.spin[None]:.3f} | Cam r: {renderer.cam_r[None]:.1f}M", end="\r", flush=True)

if __name__ == "__main__":
    if args.check:
        physics.init_physics_parameters()
        physics.run_physics_checks()
    else:
        main()
