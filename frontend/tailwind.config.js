/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",

    // Or if using `src` directory:
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        air: "#7c9eb2",
        violetlight: "#52528C",
        violetdark: "#372554",
        emerald: "#b1fcc6",
      },
      fontFamily: {
        monofett: ["Monofett", "cursive"],
        sharetech: ["Share Tech Mono", "monospace"],
      },
    },
  },
  plugins: [require("@tailwindcss/forms")],
};
