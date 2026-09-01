/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Design tokens pulled from the reference screenshot, named for what they do
        // rather than raw hex, so components read as intent ("sidebar", "accent")
        // instead of magic numbers.
        sidebar: {
          DEFAULT: "#241b35",   // main sidebar background
          hover: "#322544",     // nav item hover/active background
          border: "#242c42",
        },
        accent: {
          DEFAULT: "#8b6fd6",   // primary blue -- buttons, active nav, user avatar
          hover: "#7a5cc4",
        },
        canvas: "#f6f3fb",       // main chat area background
        muted: "#9691a8",        // section labels, secondary text
        ink: "#241b35",          // primary text color
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      borderRadius: {
        xl2: "1rem",
      },
    },
  },
  plugins: [],
};
