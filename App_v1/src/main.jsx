import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import Cadastro from "./pages/Cadastro.jsx";
import "./index.css";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import Monitoramento from "./pages/Monitoramento.jsx";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
  },
  {
    path: "/cadastro",
    element: <Cadastro />,
  },
  {
    path: "/monitoramento",
    element: <Monitoramento />,
  },
]);

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>
);
