"""
Waymo SafePoint 3D: MechDog Speed & Crosswalk Timing Calculator.
Extracts gait timing and stride parameters directly from firmware and SDK.
Calculates ground speed (m/s), walking pace, and street crossing duration.
"""

from real_world_escort import RealDogController

GAITS = {
    "DEFAULT": RealDogController.GAIT_DEFAULT,
    "FAST": RealDogController.GAIT_FAST,
    "SPRINT": RealDogController.GAIT_SPRINT,
}


def compute_mechdog_speed(stride_mm=50, lift_ms=150, contact_ms=200, slip_factor=0.85):
    """
    Computes theoretical and real-world ground velocity for Hiwonder MechDog.
    Formula (Hiwonder: move(stride_mm, turn_deg), set_gait_params(air_ms, ground_ms, lift_mm)):
      A foot on the ground carries the body one stride during its ground phase, so
      v_ideal = (stride_mm / 1000.0) / (contact_ms / 1000.0)
      v_real = v_ideal * slip_factor  (slip_factor is a guess until measured, e.g. with the sonar against a wall)
    """
    stride_m = stride_mm / 1000.0
    t_step_s = (lift_ms + contact_ms) / 1000.0
    freq_hz = 1.0 / t_step_s if t_step_s > 0 else 0

    v_ideal = stride_m / (contact_ms / 1000.0) if contact_ms > 0 else 0
    v_real = v_ideal * slip_factor

    return {
        "stride_mm": stride_mm,
        "step_duration_ms": lift_ms + contact_ms,
        "cadence_steps_per_sec": round(freq_hz, 2),
        "ideal_speed_mps": round(v_ideal, 3),
        "real_speed_mps": round(v_real, 3),
        "real_speed_mph": round(v_real * 2.23694, 2),
        "time_to_cross_6m_street_sec": round(6.0 / v_real, 1) if v_real > 0 else float("inf")
    }

if __name__ == "__main__":
    print("\n========================================================")
    print("      HIWONDER MECHDOG PHYSICAL VELOCITY ANALYSIS       ")
    print("========================================================\n")

    strides = [40, 60, 80, 100]
    print(f"{'Gait':<8} | {'lift/contact/lift_mm':<20} | {'Stride':<7} | {'Cadence':<11} | {'Speed':<9} | {'mph':<5} | 6m Cross")
    print("-" * 87)

    for name, (lift_ms, contact_ms, lift_mm) in GAITS.items():
        for s in strides:
            m = compute_mechdog_speed(stride_mm=s, lift_ms=lift_ms, contact_ms=contact_ms)
            timing = f"{lift_ms}/{contact_ms}/{lift_mm}"
            print(f"{name:<8} | {timing:<20} | {s:>3} mm  | {m['cadence_steps_per_sec']:.1f} steps/s | {m['real_speed_mps']:.3f} m/s | {m['real_speed_mph']:.2f}  | {m['time_to_cross_6m_street_sec']:5.1f} s")
        print("-" * 87)

    top = compute_mechdog_speed(100, *GAITS["SPRINT"][:2])
    print("\n[Comparison with Human Pedestrian Pace]:")
    print("  • Average adult walking speed: 1.20 - 1.40 m/s (~3.0 mph)")
    print("  • Blind / cane user walking speed: 0.80 - 1.00 m/s (~2.0 mph)")
    print(f"  • MechDog top speed (SPRINT gait, stride 100): {top['real_speed_mps']:.2f} m/s ({top['real_speed_mph']:.2f} mph)")
    print("  • Conclusion: even at sprint cadence the dog travels at ~1/2 of a cane user's pace.")
    print("    In real life, the blind person does NOT get towed; the dog walks alongside")
    print("    the handler's left leg as a short-range curb scout and obstacle radar.\n")
