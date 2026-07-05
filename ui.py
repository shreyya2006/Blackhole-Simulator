import math
import taichi as ti
import physics
import renderer

# -------------------------------------------------------------
# UI & Guided Journey State (CPU side)
# -------------------------------------------------------------
learn_mode = True
time_scale = 1.0
active_camera_type = 0  # 0 = ZAMO Orbit, 1 = Free-Flight Explorer

# Guided Journey state machine
# 0 = Free Sandbox, 1 = Lensing, 2 = Beaming, 3 = Frame Dragging, 4 = Complete
journey_stage = 0
journey_timer = 0.0

stage_texts = {
    1: (
        "STAGE 1: GRAVITATIONAL LENSING\n"
        "-------------------------------------\n"
        "Look at the starfield around the dark void.\n"
        "Curved spacetime acts as a lens. Light paths from\n"
        "stars behind the black hole are bent, wrapping them\n"
        "into Einstein rings and distorted double-images.\n"
        "We are viewing a static Schwarzschild black hole (spin a = 0)."
    ),
    2: (
        "STAGE 2: DOPlER & RELATIVISTIC BEAMING\n"
        "-------------------------------------\n"
        "Look at the color difference on the accretion disk:\n"
        "Gas on the left side is orbiting toward us at nearly\n"
        "light speed, shifting its color blue (blueshift) and\n"
        "boosting its brightness (g^4 beaming). The gas on\n"
        "the right moves away, redshifting and dimming."
    ),
    3: (
        "STAGE 3: FRAME DRAGGING & ERGOSPHERE\n"
        "-------------------------------------\n"
        "We spin the black hole up to a = 0.95 and zoom in.\n"
        "Rotating spacetime drags everything around it.\n"
        "The violet translucent bubble is the Ergosphere.\n"
        "Within this shell, space is dragged so strongly that\n"
        "nothing can stand still, even if moving at light speed!"
    ),
    4: (
        "TOUR COMPLETED: SANDBOX UNLOCKED!\n"
        "-------------------------------------\n"
        "You have finished the guided tour. You are now free\n"
        "to explore. Toggle Cinematic / Physics visual modes,\n"
        "change cameras, orbit, fly close to the horizon,\n"
        "slide parameters, and launch experimental light rays!"
    )
}

def start_journey():
    global journey_stage, journey_timer
    journey_stage = 1
    journey_timer = 0.0
    renderer.render_mode[None] = 1 # Physics viz
    physics.update_parameters(0.0) # Schwarzschild
    renderer.cam_r[None] = 42.0
    renderer.cam_th[None] = math.radians(72.0)
    renderer.cam_ph[None] = 0.0

def update_journey(dt):
    global journey_stage, journey_timer
    if journey_stage == 0:
        return
        
    journey_timer += dt
    t = min(journey_timer / 4.0, 1.0) # 4 second transition duration
    
    if journey_stage == 1:
        # Interpolate camera zoom (42M -> 24M)
        renderer.cam_r[None] = 42.0 * (1.0 - t) + 24.0 * t
        renderer.cam_th[None] = math.radians(72.0)
        physics.update_parameters(0.0)
        
    elif journey_stage == 2:
        # Interpolate inclination to near edge-on (72 deg -> 85 deg)
        renderer.cam_r[None] = 24.0
        renderer.cam_th[None] = math.radians(72.0) * (1.0 - t) + math.radians(84.0) * t
        physics.update_parameters(0.0)
        
    elif journey_stage == 3:
        # Zoom closer and spin up black hole (a: 0.0 -> 0.95)
        renderer.cam_r[None] = 24.0 * (1.0 - t) + 12.0 * t
        renderer.cam_th[None] = math.radians(84.0)
        spin_val = 0.95 * t
        physics.update_parameters(spin_val)

def draw_hud(window):
    global learn_mode, time_scale, active_camera_type, journey_stage, journey_timer
    
    gui = window.get_gui()
    
    # If in Cinematic View Mode, hide the entire HUD except for a small, transparent switch button
    if not learn_mode:
        with gui.sub_window("System", 0.88, 0.015, 0.105, 0.05) as sw:
            if sw.button("Learn Mode"):
                learn_mode = True
                renderer.render_mode[None] = 1 # Switch to physics viz
        return
    
    # 1. Main Mode Selector Window
    with gui.sub_window("Simulation Control", 0.015, 0.015, 0.27, 0.16) as sw:
        sw.text("BLACK HOLE LAB v2.0")
        
        # View Mode / Learn Mode Toggles
        if sw.button("View Mode (Cinematic)"):
            learn_mode = False
            renderer.render_mode[None] = 0 # Switch to cinematic rendering style
            journey_stage = 0              # Cancel active journey
                
        # Camera Mode Selector
        cam_label = "Orbit Cam (ZAMO)" if active_camera_type == 0 else "Free Explorer"
        if sw.button("Camera: " + cam_label):
            active_camera_type = 1 - active_camera_type
            renderer.cam_look_th[None] = math.pi
            renderer.cam_look_ph[None] = 0.0

        # Rendering style selector (Physics color vs. Cinematic color)
        if renderer.render_mode[None] == 1:
            if sw.button("Style: Relativistic Redshift Map"):
                renderer.render_mode[None] = 0 # switch to cinematic
        else:
            if sw.button("Style: Cinematic Blackbody"):
                renderer.render_mode[None] = 1 # switch to physics viz

    # 2. Guided Journey HUD
    with gui.sub_window("Guided Journey Tour", 0.015, 0.185, 0.27, 0.23) as tour:
        if journey_stage == 0:
            tour.text("Take a structured educational approach:")
            if tour.button("Start Guided Journey"):
                start_journey()
        else:
            tour.text(f"Guided Tour Stage: {journey_stage} / 4")
            if journey_stage < 4:
                if tour.button("Next Stage"):
                    journey_stage += 1
                    journey_timer = 0.0
            else:
                if tour.button("Unlock Sandbox Mode"):
                    journey_stage = 0 # Back to sandbox
            
            if tour.button("Cancel Tour"):
                journey_stage = 0
                
            tour.text(f"Transition: {min(journey_timer / 4.0 * 100, 100):.0f}%")

    # 3. Parameters Panel (Adaptive: disabled during transitions, full in sandbox)
    with gui.sub_window("Physics Sandbox & Experiments", 0.015, 0.425, 0.27, 0.555) as ps:
        if journey_stage in [1, 2, 3]:
            ps.text("SANDBOX: Locked during Tour.")
            ps.text(f"Current spin: {physics.spin[None]:.2f}")
            ps.text(f"Distance: {renderer.cam_r[None]:.1f}M")
        else:
            ps.text("BLACK HOLE MASS & SPIN")
            # Spin slider (Kerr)
            new_a = ps.slider_float("Spin a", physics.spin[None], 0.0, 0.999)
            if abs(new_a - physics.spin[None]) > 1e-4:
                physics.update_parameters(new_a)
            
            ps.text(f"Horizon r+ = {physics.r_hor[None]:.2f} M")
            ps.text(f"ISCO Radius = {physics.r_isco[None]:.2f} M")
            
            ps.text("CAMERA ADJUSTMENTS")
            new_r = ps.slider_float("Distance r_c", renderer.cam_r[None], 5.0, 55.0)
            if abs(new_r - renderer.cam_r[None]) > 1e-3:
                renderer.cam_r[None] = new_r
                
            new_th = ps.slider_float("Inclination", math.degrees(renderer.cam_th[None]), 3.0, 177.0)
            if abs(new_th - math.degrees(renderer.cam_th[None])) > 1e-2:
                renderer.cam_th[None] = math.radians(new_th)
                
            new_fov = ps.slider_float("FOV", math.degrees(renderer.fov[None]), 15.0, 90.0)
            if abs(new_fov - math.degrees(renderer.fov[None])) > 1e-2:
                renderer.fov[None] = math.radians(new_fov)
                
            renderer.exposure[None] = ps.slider_float("Exposure", renderer.exposure[None], 0.2, 4.0)

            ps.text("TIME INTEGRATION")
            time_scale = ps.slider_float("Speed", time_scale, 0.0, 3.0)
            
            ps.text("EXPERIMENTS (Forward Tracing)")
            if ps.button("Launch Light Ray"):
                physics.trace_launched_ray(
                    renderer.cam_r[None], renderer.cam_th[None], renderer.cam_ph[None],
                    renderer.cam_look_th[None], renderer.cam_look_ph[None], physics.spin[None]
                )
            if ps.button("Clear Trajectories"):
                physics.ray_path_len[None] = 0

    # 4. Learning Explanation Window (Adaptive Text)
    with gui.sub_window("Scientific Core", 0.71, 0.015, 0.275, 0.44) as edu:
        if journey_stage > 0:
            # Guided tour explanation text
            text_lines = stage_texts[journey_stage].split("\n")
            for line in text_lines:
                edu.text(line)
        else:
            # Sandbox default explanation text
            edu.text("ACCRETION DISK & precessions:")
            edu.text("The accretion disk consists of swirling hot dust.")
            edu.text("Particles orbit in Boyer-Lindquist coordinates,")
            edu.text("precessing around the event horizon.")
            edu.text("---------------------------------------")
            edu.text("SPACETIME METRIC BOUNDARIES:")
            edu.text("Horizon (r+): solid black boundary.")
            edu.text("ISCO ring: the bright neon yellow orbit.")
            edu.text("Ergosphere shell (lensed purple shell):")
            edu.text("space itself revolves here.")
            edu.text("---------------------------------------")
            edu.text("EXPERIMENTS Sandbox:")
            edu.text("Position the camera, click 'Launch Light Ray'")
            edu.text("to fire a laser from the lens and observe")
            edu.text("orbital precession, frame dragging, or capture!")

    # 5. Interactive Keyboard Reference
    with gui.sub_window("Interactive Commands", 0.71, 0.46, 0.275, 0.20) as ctrl:
        ctrl.text("CONTROLS:")
        ctrl.text("Mouse Drag   - Orbit (ZAMO) / Look (Explorer)")
        if active_camera_type == 0:
            ctrl.text("W / S        - Zoom In / Out")
        else:
            ctrl.text("W/S/A/D      - First-person flying controls")
        ctrl.text("SPACE        - Pause / Resume time scaling")
        ctrl.text("R            - Reset Lab Parameters")
