import { resolve } from "path";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

const svelteImportPattern = /import\s+([\w*{},\s]+)\s+from\s+['"](svelte|svelte\/internal)['"];?/g;
const bareSvelteImportPattern = /import\s+['"]svelte(\/\w+)*['"];?/g;

function gradioSvelteRuntime() {
  return {
    name: "gradio-svelte-runtime",
    enforce: "post",
    transform(code) {
      if (!code.includes("svelte")) return null;
      const next = code
        .replace(svelteImportPattern, (_, imported) => `const ${imported.trim().replace(" as ", ": ")} = window.__gradio__svelte__internal;`)
        .replace(bareSvelteImportPattern, "");
      return next === code ? null : { code: next, map: null };
    }
  };
}

export default defineConfig({
  plugins: [svelte({ compilerOptions: { accessors: true, discloseVersion: false } }), gradioSvelteRuntime()],
  build: {
    lib: {
      entry: resolve(__dirname, "Index.svelte"),
      formats: ["es"],
      fileName: () => "index.js"
    },
    outDir: "../templates/component",
    emptyOutDir: true,
    cssCodeSplit: false,
    rollupOptions: {
      output: {
        assetFileNames: () => "style.css"
      }
    }
  }
});
