# MechDog WiFi bridge configuration. Edit before uploading with the Hiwonder Python Editor.

# "ap": MechDog creates its own network (most reliable at a demo venue).
# "sta": MechDog joins an existing network (set STA_SSID / STA_PASSWORD).
# "none": USB serial only. No radio at all, so nothing can fail at boot (use this away from the hotspot).
# "ble": no WiFi; the bridge talks over Hiwonder's Bluetooth LE serial service instead (host: --serial ble).
#        Use this when WiFi can't send: once the bridge runs, WiFi is left ~3 KB of IDF heap and TX fails.
WIFI_MODE = "none"
BLE_NAME = "MechDog"

AP_SSID = "MechDog_wifi"
AP_PASSWORD = "12345678"

STA_SSID = "YourMiamiHome24"
STA_PASSWORD = "Welcome2miami"
STA_CONNECT_TIMEOUT_S = 20

PORT = 5005
WATCHDOG_MS = 1500
LOW_BATTERY_V = 6.6
# Hiwonder's own continuous low-voltage beep. False silences it; walking still stops below LOW_BATTERY_V.
LOW_POWER_BEEP = False
OBSTACLE_STOP_CM = 20
OBSTACLE_GUARD = False     # 2026-09-27: disabled; false short echoes stopped the walk every step
# Unit returned by Hiwonder_IIC.I2CSonar.getDistance(). Confirmed cm by reading the stock main.py off this
# device (2026-09-08): it does `_distance = round(_SONER_DISTANCE * 10)` before reporting mm to the app, and
# compares raw readings against 10 and 40 as cm thresholds. Reported in hello.caps.
SONAR_UNIT = "cm"
SONAR_MEDIAN_N = 3        # median of the last N valid reads: one glitch cannot trigger the obstacle guard
SONAR_DROPOUT_POLLS = 5   # invalid/out-of-range reads in a row before the distance is reported as unknown
FALL_DEG = 999            # 2026-09-27: fall stop disabled; walking shake read as falls and cut the legs
FALL_CLEAR_DEG = 30       # back below this on both axes = upright again, walking allowed

TELEMETRY_HZ_DEFAULT = 0
SENSOR_POLL_MS = 100   # sonar/IMU/battery are polled at this rate regardless of telemetry subscription
LOOP_MS = 10

FW_VERSION = "mechdog-bridge-0.1"
