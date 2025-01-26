import type { Config } from "tailwindcss";

export default {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: "#213362",
        },
        secondary: {
          DEFAULT: "#cc034d",
        },
        background: {
          light: "#f9fafb",
          white: "#ffffff",
        },
        text: {
          primary: "#1e293b",
          secondary: "#64748b",
          white: "#ffffff",
        },
      },
      fontFamily: {
        sans: ["var(--font-gt-america)", "sans-serif"],
        canela: ["var(--font-canela-deck)", "serif"],
      },
      backgroundImage: {
        website: "url('/website.png')",
      },
    },
  },
  plugins: [],
} satisfies Config;
