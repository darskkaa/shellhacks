# MechDog WiFi bridge configuration. Edit before uploading with the Hiwonder Python Editor.

# "ap": MechDog creates its own network (most reliable at a demo venue).
# "sta": MechDog joins an existing network (set STA_SSID / STA_PASSWORD).
WIFI_MODE = "sta"

AP_SSID = "MechDog_wifi"
AP_PASSWORD = "12345678"

STA_SSID = "YourMiamiHome24"
STA_PASSWORD = "Welcome2miami"
STA_CONNECT_TIMEOUT_S = 20

PORT = 5005
WATCHDOG_MS = 1500
LOW_BATTERY_V = 6.6
OBSTACLE_STOP_CM = 20
OBSTACLE_GUARD = True
# Unit returned by Hiwonder_IIC.I2CSonar.getDistance(). Confirmed cm by reading the stock main.py off this
# device (2026-09-08): it does `_distance = round(_SONER_DISTANCE * 10)` before reporting mm to the app, and
# compares raw readings against 10 and 40 as cm thresholds. Reported in hello.caps.
SONAR_UNIT = "cm"
SONAR_MEDIAN_N = 3        # median of the last N valid reads: one glitch cannot trigger the obstacle guard
SONAR_DROPOUT_POLLS = 5   # invalid/out-of-range reads in a row before the distance is reported as unknown
FALL_DEG = 50             # |roll| or |pitch| above this = knocked over / fallen (Hiwonder IOT lesson impact rule)
FALL_CLEAR_DEG = 30       # back below this on both axes = upright again, walking allowed

TELEMETRY_HZ_DEFAULT = 0
SENSOR_POLL_MS = 100   # sonar/IMU/battery are polled at this rate regardless of telemetry subscription
LOOP_MS = 10

FW_VERSION = "mechdog-bridge-0.1"
