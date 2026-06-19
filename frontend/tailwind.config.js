/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        darkBg: '#09090b',
        panelBg: '#18181b',
        borderBg: '#27272a',
        accentBg: '#ff4444'
      }
    },
  },
  plugins: [],
}
