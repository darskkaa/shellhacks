import js from "@eslint/js";
import globals from "globals";

export default [
  { ignores: ["node_modules/", "data/"] },
  js.configs.recommended,
  { languageOptions: { ecmaVersion: 2024, sourceType: "module", globals: globals.node } },
  // Rest-destructuring is how fields are dropped before sending or prompting.
  { rules: { "no-unused-vars": ["error", { ignoreRestSiblings: true }] } },
];
