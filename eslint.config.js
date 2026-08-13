import eslint from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist/**", "node_modules/**", "benchmark/data/**", "benchmark/.venv/**"] },
  eslint.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["src/**/*.ts", "tests/**/*.ts", "benchmark/**/*.ts"],
    languageOptions: {
      parserOptions: { project: "./tsconfig.json" },
      globals: {
        chrome: "readonly",
        document: "readonly",
        window: "readonly",
        navigator: "readonly",
        indexedDB: "readonly",
        crypto: "readonly",
        fetch: "readonly",
        Image: "readonly",
        HTMLImageElement: "readonly",
        ImageData: "readonly",
        createImageBitmap: "readonly",
        requestAnimationFrame: "readonly",
        cancelAnimationFrame: "readonly",
        setTimeout: "readonly",
        clearTimeout: "readonly",
        URL: "readonly",
        Blob: "readonly",
        FileReader: "readonly",
        MutationObserver: "readonly",
        IntersectionObserver: "readonly",
        ResizeObserver: "readonly",
        Node: "readonly",
        Element: "readonly",
        HTMLElement: "readonly",
        CustomEvent: "readonly"
      }
    },
    rules: {
      "@typescript-eslint/consistent-type-imports": "error",
      "@typescript-eslint/no-explicit-any": "error"
    }
  },
  {
    files: ["scripts/**/*.mjs", "eslint.config.js"],
    languageOptions: {
      globals: { console: "readonly", process: "readonly", Buffer: "readonly", URL: "readonly", chrome: "readonly" }
    }
  }
);
