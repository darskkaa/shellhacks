// Fails if an inline <script> in public/index.html has a syntax error; prettier and eslint do not parse it.
import { readFileSync } from "node:fs";

const html = readFileSync(new URL("../public/index.html", import.meta.url), "utf8");
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
if (!scripts.length) throw new Error("no inline <script> found in public/index.html");
for (const code of scripts) new Function(code);
console.log(`page checks passed (${scripts.length} inline script)`);
