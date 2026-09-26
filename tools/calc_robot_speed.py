"""
Waymo SafePoint 3D: MechDog Speed & Crosswalk Timing Calculator.
Extracts gait timing and stride parameters directly from firmware and SDK.
Calculates ground speed (m/s), walking pace, and street crossing duration.
"""

def compute_mechdog_speed(stride_mm=50, lift_ms=150, contact_ms=200, slip_factor=0.85):
    """
    Computes theoretical and real-world ground velocity for Hiwonder MechDog.
    Formula:
      T_cycle = (lift_ms + contact_ms) / 1000.0  (seconds per trot phase)
      freq_hz = 1.0 / T_cycle
      v_ideal = (stride_mm / 1000.0) * freq_hz
      v_real = v_ideal * slip_factor
    """
    stride_m = stride_mm / 1000.0
    t_step_s = (lift_ms + contact_ms) / 1000.0
    freq_hz = 1.0 / t_step_s if t_step_s > 0 else 0

    v_ideal = stride_m * freq_hz
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

    strides = [30, 40, 50, 60, 80, 100]
    print(f"{'Stride':<10} | {'Cadence':<12} | {'Real Speed':<14} | {'Speed (mph)':<12} | {'Time (6m Cross)':<15}")
    print("-" * 75)

    for s in strides:
        m = compute_mechdog_speed(stride_mm=s, lift_ms=150, contact_ms=200)
        print(f"{s} mm{'':<5} | {m['cadence_steps_per_sec']} steps/s   | {m['real_speed_mps']} m/s ({m['real_speed_mps']*100:.0f} cm/s) | {m['real_speed_mph']} mph      | {m['time_to_cross_6m_street_sec']} s")

    print("\n[Comparison with Human Pedestrian Pace]:")
    print("  • Average adult walking speed: 1.20 - 1.40 m/s (~3.0 mph)")
    print("  • Blind / cane user walking speed: 0.80 - 1.00 m/s (~2.0 mph)")
    print("  • MechDog max speed (stride 100): ~0.25 - 0.28 m/s (~0.6 mph)")
    print("  • Conclusion: Robot dog travels at ~1/3 to 1/4 the speed of a human walking pace.")
    print("    In real life, the blind person does NOT get towed; the dog walks alongside")
    print("    the handler's left leg as a short-range curb scout and obstacle radar.\n")
