import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#F7F5F2",
        primary: "#1A1A1A",
        secondary: "#6B6B6B",
        accent: "#1C3F5E",
        approve: "#2D6A4F",
        deny: "#9B2335",
        escalate: "#B45309",
        border: "#E2DDD8",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "sans-serif"],
        mono: ["var(--font-jetbrains)", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
