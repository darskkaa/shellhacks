// Route energy estimate for a Waymo Jaguar I-PACE and Pareto ranking over (risk, time, energy).
// Pure functions with no env or I/O, so check-rank.mjs can import them without API keys.

const G = 9.81;
// I-PACE curb mass (~2,130 kg) plus Waymo sensors and a passenger or two.
const MASS_KG = 2200;
// Published I-PACE drag coefficient.
const DRAG_CD = 0.29;
// I-PACE frontal area.
const FRONTAL_AREA_M2 = 2.6;
// Sea-level air density; Miami is at sea level.
const AIR_DENSITY = 1.2;
// Typical low-rolling-resistance EV tire on asphalt.
const ROLLING_CRR = 0.011;
// Battery-to-wheel efficiency (inverter, motors, gearing).
const DRIVETRAIN_EFF = 0.9;
// Share of braking or descent energy the motors recover into the battery.
const REGEN_RECOVERY = 0.6;
// ponytail: stops per km = this / average km/h (2 stops/km at 30 km/h, 0.6 at 100); swap for signal counts per route.
const STOP_RATE_KMH = 60;
// Speed the car gets back up to after each urban stop (~30 mph); faster routes use their average instead.
const URBAN_CRUISE_MS = 13.4;
// ponytail: flat 1.5 kW for A/C in Miami heat plus Waymo's sensors and compute; measure per vehicle if it matters.
const AUX_KW = 1.5;
const J_PER_KWH = 3.6e6;

// Traction plus auxiliary energy in kWh for one route: rolling, drag at average speed, stop-and-go
// re-acceleration, and elevation gain/loss from [lat, lng, meters] samples when present.
export function routeEnergyKwh({ distanceM, durationSec, elevationProfile }) {
  const v = durationSec > 0 ? distanceM / durationSec : URBAN_CRUISE_MS;
  const km = distanceM / 1000;
  const rolling = ROLLING_CRR * MASS_KG * G * distanceM;
  const drag = 0.5 * AIR_DENSITY * DRAG_CD * FRONTAL_AREA_M2 * v * v * distanceM;
  const peak = Math.max(v, URBAN_CRUISE_MS);
  const stops = (STOP_RATE_KMH / Math.max(v * 3.6, 1)) * km;
  const stopGo = stops * 0.5 * MASS_KG * peak * peak * (1 - REGEN_RECOVERY);
  let gain = 0,
    loss = 0;
  for (let i = 1; i < (elevationProfile?.length ?? 0); i++) {
    const dh = elevationProfile[i][2] - elevationProfile[i - 1][2];
    if (dh > 0) gain += dh;
    else loss -= dh;
  }
  const hills = MASS_KG * G * (gain / DRIVETRAIN_EFF - loss * REGEN_RECOVERY);
  const aux = AUX_KW * 1000 * Math.max(durationSec, 0);
  const joules = (rolling + drag + stopGo) / DRIVETRAIN_EFF + hills + aux;
  return Number((Math.max(joules, 0) / J_PER_KWH).toFixed(2));
}

// Risk points within which routes count as equally safe: max(3, 5% of the lowest risk).
const RISK_TIE_MIN = 3;
const RISK_TIE_SHARE = 0.05;
// Durations within 30 s count as equally fast, so energy decides.
const DURATION_TIE_SEC = 30;

const dominates = (a, b) =>
  a.risk.total <= b.risk.total &&
  a.durationSec <= b.durationSec &&
  a.energyKwh <= b.energyKwh &&
  (a.risk.total < b.risk.total || a.durationSec < b.durationSec || a.energyKwh < b.energyKwh);
const byRisk = (a, b) => a.risk.total - b.risk.total || a.durationSec - b.durationSec || a.energyKwh - b.energyKwh;

// Routes need risk.total, durationSec, energyKwh. Returns copies ordered recommended, other frontier routes by risk,
// then dominated routes by risk, each with frontier, dominatedBy (letter), badges, recommended; plus the rule applied.
export function rankRoutes(routes) {
  const minRisk = Math.min(...routes.map((r) => r.risk.total));
  const riskTie = Number(Math.max(RISK_TIE_MIN, RISK_TIE_SHARE * minRisk).toFixed(2));
  const safe = routes.filter((r) => r.risk.total <= minRisk + riskTie);
  const quickest = Math.min(...safe.map((r) => r.durationSec));
  const [best] = safe
    .filter((r) => r.durationSec <= quickest + DURATION_TIE_SEC)
    .sort((a, b) => a.energyKwh - b.energyKwh || byRisk(a, b));
  const isFrontier = (r) => !routes.some((o) => dominates(o, r));
  const rest = routes.filter((r) => r !== best);
  const ordered = [best, ...rest.filter(isFrontier).sort(byRisk), ...rest.filter((r) => !isFrontier(r)).sort(byRisk)];
  const min = (key) => Math.min(...routes.map(key));
  const [minR, minT, minE] = [min((r) => r.risk.total), min((r) => r.durationSec), min((r) => r.energyKwh)];
  const letter = (i) => String.fromCharCode(65 + i);
  return {
    routes: ordered.map((r) => {
      const by = ordered.findIndex((o) => dominates(o, r));
      return {
        ...r,
        frontier: by === -1,
        dominatedBy: by === -1 ? null : letter(by),
        badges: [
          r.risk.total === minR && "safest",
          r.durationSec === minT && "fastest",
          r.energyKwh === minE && "efficient",
        ].filter(Boolean),
        recommended: r === best,
      };
    }),
    ranking: {
      rule:
        `Safety first: among routes within ${riskTie} risk points of the safest, the fastest wins; ` +
        `if times are within ${DURATION_TIE_SEC} s, the one using the least energy.`,
      riskTie,
    },
  };
}
