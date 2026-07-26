import { readdirSync, readFileSync } from "fs";
import { createRequire } from "module";
import { resolve } from "path";
import { fileURLToPath } from "url";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import sveltePreprocess from "svelte-preprocess";
import { defineConfig } from "vite";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const require = createRequire(import.meta.url);
const gradioFrontendRoot = "C:/Users/Marc/anaconda3/envs/py311/Lib/site-packages/gradio/_frontend_code";
const localNodeModules = resolve(__dirname, "node_modules");
const svelteImportPattern = /import\s+([\w*{},\s]+)\s+from\s+['"](svelte|svelte\/internal)['"]/g;
const localRuntimePackages = new Set([
  "amuchina",
  "dompurify",
  "github-slugger",
  "katex",
  "marked",
  "marked-gfm-heading-id",
  "marked-highlight",
  "mermaid",
  "pixi.js",
  "prismjs",
  "svelte-i18n",
  "tinycolor2"
]);

function gradioPackages() {
  const packages = new Map();
  for (const dirname of readdirSync(gradioFrontendRoot, { withFileTypes: true })) {
    if (!dirname.isDirectory()) continue;
    const packagePath = resolve(gradioFrontendRoot, dirname.name, "package.json");
    let pkg;
    try {
      pkg = JSON.parse(readFileSync(packagePath, "utf8"));
    } catch {
      continue;
    }
    if (!pkg.name?.startsWith("@gradio/")) continue;
    if (pkg.name === "@gradio/client") continue;
    packages.set(pkg.name, { dir: resolve(gradioFrontendRoot, dirname.name), pkg });
  }
  return packages;
}

const gradioPackageMap = gradioPackages();

function pickExportEntry(exportValue) {
  if (!exportValue || typeof exportValue === "string") return exportValue;
  return exportValue.browser?.gradio || exportValue.browser?.svelte || exportValue.browser?.import || exportValue.browser?.default || exportValue.gradio || exportValue.svelte || exportValue.import || exportValue.default?.gradio || exportValue.default?.svelte || exportValue.default?.import || exportValue.default?.default || exportValue.default;
}

function gradioSvelteRuntime() {
  return {
    name: "gradio-svelte-runtime",
    enforce: "post",
    transform(code) {
      const newCode = code.replace(svelteImportPattern, (_match, imports) => `const ${imports.replace(/\* as /, "").replace(/ as /g, ": ")} = window.__gradio__svelte__internal;`);
      return newCode === code ? null : { code: newCode, map: null };
    }
  };
}

function localRuntimeResolver() {
  return {
    name: "local-runtime-resolver",
    enforce: "pre",
    resolveId(source) {
      const packageName = source.split("/")[0];
      if (localRuntimePackages.has(packageName)) return require.resolve(source);
      return null;
    }
  };
}

function gradioSourceResolver() {
  return {
    name: "gradio-source-resolver",
    enforce: "pre",
    resolveId(source) {
      if (source === "@gradio/client") {
        return resolve(localNodeModules, "@gradio/client/dist/index.js");
      }
      if (source === "@gradio/client/package.json") {
        return resolve(localNodeModules, "@gradio/client/package.json");
      }
      if (!source.startsWith("@gradio/")) return null;
      const parts = source.split("/");
      const packageName = `${parts[0]}/${parts[1]}`;
      const packageInfo = gradioPackageMap.get(packageName);
      if (!packageInfo) return null;
      const subpath = parts.length > 2 ? `./${parts.slice(2).join("/")}` : ".";
      const exportValue = packageInfo.pkg.exports?.[subpath];
      const entry = pickExportEntry(exportValue) || (subpath === "." ? packageInfo.pkg.main : subpath);
      return resolve(packageInfo.dir, entry);
    }
  };
}

export default defineConfig({
  plugins: [localRuntimeResolver(), gradioSourceResolver(), svelte({ preprocess: sveltePreprocess({ typescript: { tsconfigFile: resolve(__dirname, "tsconfig.json") } }), compilerOptions: { accessors: true, immutable: true, discloseVersion: false } }), gradioSvelteRuntime()],
  resolve: {
    conditions: ["gradio", "svelte", "browser", "import"]
  },
  optimizeDeps: {
    exclude: Array.from(gradioPackageMap.keys())
  },
  build: {
    target: "es2020",
    sourcemap: true,
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
        entryFileNames: "index.js",
        chunkFileNames: "wangp-[name]-[hash].js",
        assetFileNames: (assetInfo) => assetInfo.name?.endsWith(".css") ? "style.css" : "wangp-[name]-[hash][extname]"
      }
    }
  }
});
