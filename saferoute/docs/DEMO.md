# Demo script (about 3 minutes)

Start: `npm start`, open http://localhost:3000. The page loads straight into Design District → Legion Park.

1. **Hook (Design District → Legion Park, loads automatically).** Point at the verdict card: Google's fastest passes 8 flood reports that the recommended route avoids for about 8 s. When Google offers only one route, SafeRoute generates its own detours, so the safest card may read "(SafeRoute detour)". Then point at the green line against the red one and the risk bars. Numbers shift with Google traffic.
2. **Live conditions (top tiles).** Today's king tide at Virginia Key, the NWS flood alert (storm mode doubles or quadruples flood weight), and the number of crashes Miami-Dade Police are handling right now.
3. **Venue trip (FIU → Brickell button).** The safest route is also the fastest here, but the SW 24th St and Tamiami alternatives cross 12 flood reports and active FDOT work zones. Read one 🚧 project name aloud.
4. **Same time, safer (UM → Brickell button).** Both routes take about 17 minutes, but Route A has 40 fewer crashes and 2 fewer fatal ones. Safety costs nothing here.
5. **Flood simulator.** Pick FIU → Brickell. Drag the water slider up to 9 ft: road that would be under water turns blue on the map, the gauge fills, and the elevation chart floods (hover it to move a marker along the route; tap a red dot for a real crash record). Then press "Find a drier route": the plan changes from FL-836 to a SW 24th St detour and the blue banner says why.
6. **Hurricane simulator.** On FIU → Brickell set Conditions to "Category 3 hurricane": SafeRoute reroutes from FL-836 to an inland detour via NW 36th St (+8 min, 6.4 fewer miles in surge zones on Sep 26), shown in green against Google's dashed fastest route through the shaded zones. Then try Category 4 on Coconut Grove → Wynwood. The map shades evacuation zones A-D, the cards show miles through each zone and NOAA's worst-case surge depth (up to 9 ft from Coconut Grove), and risk jumps. Little Havana to Lincoln Road shows 4 ft over the causeway at Category 3. The "Nearest Atlantic storm" tile shows the real NHC picture (TS Fay, about 2,257 mi away on Sep 26).
7. **Close.** 34,817 hazards loaded into memory at startup (from MongoDB Atlas, or local GeoJSON without it), all from public data (FDOT, Miami-Dade 311, Miami-Dade Police, NOAA, NWS, Google Maps). Gemini writes the plain-English summary from our numbers only.

Numbers above are from Sep 26 2026 runs. Live tiles, live crashes, and Gemini wording change from run to run.
