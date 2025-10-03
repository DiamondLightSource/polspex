// import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ThemeProvider, DiamondTheme } from "@diamondlightsource/sci-react-ui";
// import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  // <StrictMode>
  //   <App />
  // </StrictMode>,
  <ThemeProvider theme={DiamondTheme}>
    <App />
  </ThemeProvider>,
)
