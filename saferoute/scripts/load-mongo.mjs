// Loads data/*.geojson into MongoDB Atlas: point hazards into `saferoute.hazards`, evacuation zone polygons
// into `saferoute.evac_zones`, both with 2dsphere indexes.
// Run: npm run load-mongo
import { readFile } from "node:fs/promises";
import { MongoClient } from "mongodb";

const KINDS = {
  crash: "data/crashes.geojson",
  flood: "data/flooding.geojson",
  school: "data/schools.geojson",
  construction: "data/construction.geojson",
};

if (!process.env.MONGODB_URI) throw new Error("MONGODB_URI missing from .env");
const client = new MongoClient(process.env.MONGODB_URI);
try {
  const hazards = client.db("saferoute").collection("hazards");
  for (const [kind, file] of Object.entries(KINDS)) {
    const { features } = JSON.parse(await readFile(file, "utf8"));
    const docs = features.map(({ geometry, properties }) => ({ kind, geometry, properties }));
    // Replace per kind so re-running after fetch-data never duplicates rows.
    await hazards.deleteMany({ kind });
    await hazards.insertMany(docs, { ordered: false });
    console.log(`${kind}: ${docs.length}`);
  }
  await hazards.createIndex({ geometry: "2dsphere", kind: 1 });
  console.log(`total: ${await hazards.countDocuments()}`);

  // Polygons live in their own collection: route scoring asks "which zones does this line cross", not "which points are near".
  const zonesCol = client.db("saferoute").collection("evac_zones");
  const { features: zones } = JSON.parse(await readFile("data/evac-zones.geojson", "utf8"));
  await zonesCol.deleteMany({});
  await zonesCol.createIndex({ geometry: "2dsphere" });
  let rejected = 0;
  // Insert one at a time so a single self-intersecting county polygon (which 2dsphere rejects) is skipped, not fatal.
  for (const { geometry, properties } of zones) {
    try {
      await zonesCol.insertOne({ geometry, category: properties.CATEGORY, zone: properties.ZONEID });
    } catch {
      rejected++;
    }
  }
  console.log(`evac zones: ${zones.length - rejected} loaded, ${rejected} rejected as invalid geometry`);
} finally {
  await client.close();
}
